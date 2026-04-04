"""
Unit tests for src/tools/geo_matcher.py

All tests use MockGeocoder so no real HTTP calls are made.
Distance expectations are approximate (±5 miles) because geodesic distances
from the registry coordinates may differ slightly from published values.
"""

from __future__ import annotations

import pytest

from src.api.models import Location, Trial
from src.tools.geo_matcher import GeoMatchResult, GeoMatchSummary, GeoMatcher
from tests.conftest import COORDS_REGISTRY, MockGeocoder, make_location, make_trial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def geo(registry: dict | None = None) -> GeoMatcher:
    """Return a GeoMatcher with a mock geocoder (no real HTTP, no delay)."""
    return GeoMatcher(geocoder=MockGeocoder(registry), delay_seconds=0)


# ---------------------------------------------------------------------------
# _location_to_string
# ---------------------------------------------------------------------------


class TestLocationToString:
    def test_city_state_country(self):
        loc = make_location("Boston", "MA", "United States")
        assert GeoMatcher._location_to_string(loc) == "Boston, MA, United States"

    def test_city_and_state_only(self):
        loc = Location(city="Boston", state="MA")
        assert GeoMatcher._location_to_string(loc) == "Boston, MA"

    def test_zip_takes_precedence_over_city(self):
        loc = make_location("Boston", "MA", zip_code="02101")
        result = GeoMatcher._location_to_string(loc)
        assert result.startswith("02101")

    def test_zip_with_country(self):
        loc = Location(zip_code="02101", country="United States")
        assert GeoMatcher._location_to_string(loc) == "02101, United States"

    def test_zip_without_country(self):
        loc = Location(zip_code="02101")
        assert GeoMatcher._location_to_string(loc) == "02101"

    def test_country_only(self):
        loc = Location(country="United States")
        assert GeoMatcher._location_to_string(loc) == "United States"

    def test_all_none_returns_empty_string(self):
        loc = Location()
        assert GeoMatcher._location_to_string(loc) == ""


# ---------------------------------------------------------------------------
# _geocode (with caching)
# ---------------------------------------------------------------------------


class TestGeocode:
    def test_known_location_returns_coords(self):
        matcher = geo()
        result = matcher._geocode("Boston, MA, United States")
        assert result == COORDS_REGISTRY["boston, ma, united states"]

    def test_unknown_location_returns_none(self):
        matcher = geo()
        result = matcher._geocode("Nowhere City, ZZ")
        assert result is None

    def test_empty_string_returns_none(self):
        matcher = geo()
        assert matcher._geocode("") is None
        assert matcher._geocode("   ") is None

    def test_result_is_cached(self):
        geocoder = MockGeocoder()
        original_geocode = geocoder.geocode
        call_count = 0

        def counting_geocode(query, timeout=10):
            nonlocal call_count
            call_count += 1
            return original_geocode(query, timeout)

        geocoder.geocode = counting_geocode
        matcher = GeoMatcher(geocoder=geocoder, delay_seconds=0)

        matcher._geocode("Boston, MA, United States")
        matcher._geocode("Boston, MA, United States")  # second call

        assert call_count == 1, "Geocoder should only be called once per unique location"

    def test_cache_is_per_instance(self):
        m1 = geo()
        m2 = geo()
        m1._geocode("Boston, MA, United States")
        # m2 should have an empty cache independent of m1
        assert len(m2._geocache) == 0


# ---------------------------------------------------------------------------
# match — basic filtering
# ---------------------------------------------------------------------------


class TestMatchFiltering:
    def test_returns_trial_within_radius(self, trial_cambridge):
        matcher = geo()
        summary = matcher.match([trial_cambridge], "Boston, MA, United States", radius_miles=50)
        assert len(summary.results) == 1
        assert summary.results[0].trial.nct_id == "NCT0000001"

    def test_excludes_trial_outside_radius(self, trial_chicago):
        matcher = geo()
        summary = matcher.match([trial_chicago], "Boston, MA, United States", radius_miles=50)
        assert len(summary.results) == 0

    def test_skips_trial_with_no_locations(self, trial_no_locations):
        matcher = geo()
        summary = matcher.match([trial_no_locations], "Boston, MA, United States")
        assert len(summary.results) == 0
        assert summary.trials_with_locations == 0

    def test_mixed_list_filters_correctly(self, boston_trials):
        # Cambridge (~4 mi) and Worcester (~40 mi) should be within 50 miles
        # Providence (~50 mi) is borderline — include if ≤ radius
        # Hartford (~100 mi) and Chicago (~980 mi) should be excluded
        matcher = geo()
        summary = matcher.match(boston_trials, "Boston, MA, United States", radius_miles=50)
        nct_ids = {r.trial.nct_id for r in summary.results}
        assert "NCT0000001" in nct_ids  # Cambridge
        assert "NCT0000002" in nct_ids  # Worcester
        assert "NCT0000004" not in nct_ids  # Hartford
        assert "NCT0000005" not in nct_ids  # Chicago

    def test_empty_trial_list(self):
        matcher = geo()
        summary = matcher.match([], "Boston, MA, United States")
        assert summary.results == []
        assert summary.total_trials_input == 0

    def test_radius_zero_returns_nothing(self, trial_cambridge):
        matcher = geo()
        summary = matcher.match([trial_cambridge], "Boston, MA, United States", radius_miles=0)
        assert len(summary.results) == 0


# ---------------------------------------------------------------------------
# match — sorting
# ---------------------------------------------------------------------------


class TestMatchSorting:
    def test_results_sorted_by_distance_ascending(self, boston_trials):
        matcher = geo()
        summary = matcher.match(boston_trials, "Boston, MA, United States", radius_miles=150)
        distances = [r.distance_miles for r in summary.results]
        assert distances == sorted(distances)

    def test_nearest_trial_is_first(self, trial_cambridge, trial_worcester):
        matcher = geo()
        # Pass Worcester first to ensure sorting is not input-order dependent
        summary = matcher.match(
            [trial_worcester, trial_cambridge],
            "Boston, MA, United States",
        )
        assert summary.results[0].trial.nct_id == "NCT0000001"  # Cambridge is nearer

    def test_single_trial_returned_without_error(self, trial_cambridge):
        matcher = geo()
        summary = matcher.match([trial_cambridge], "Boston, MA, United States")
        assert len(summary.results) == 1


# ---------------------------------------------------------------------------
# match — multi-site trials
# ---------------------------------------------------------------------------


class TestMultiSiteTrials:
    def test_uses_nearest_site(self, trial_multi_site):
        """
        trial_multi_site has sites in Chicago (~980 mi) and Cambridge (~4 mi).
        Within a 50-mile radius from Boston, it should appear with the Cambridge distance.
        """
        matcher = geo()
        summary = matcher.match([trial_multi_site], "Boston, MA, United States", radius_miles=50)
        assert len(summary.results) == 1
        result = summary.results[0]
        assert result.city == "Cambridge"
        assert result.distance_miles < 10  # Cambridge is ~4 miles from Boston

    def test_multi_site_excluded_when_all_sites_out_of_range(self, trial_chicago):
        matcher = geo()
        summary = matcher.match([trial_chicago], "Boston, MA, United States", radius_miles=50)
        assert len(summary.results) == 0


# ---------------------------------------------------------------------------
# match — geocoding failures
# ---------------------------------------------------------------------------


class TestGeocodingFailures:
    def test_unknown_user_location_returns_error(self, trial_cambridge):
        matcher = geo()
        summary = matcher.match([trial_cambridge], "Fictional City, ZZ")
        assert len(summary.results) == 0
        assert len(summary.errors) == 1
        assert "Fictional City" in summary.errors[0]
        assert summary.user_coords is None

    def test_trial_with_ungeocodeable_location_is_skipped(self):
        trial = make_trial(
            nct_id="NCT_UNKNOWN",
            title="Nowhere Trial",
            locations=[Location(city="Unknown City", state="ZZ", country="Atlantis")],
        )
        matcher = geo()
        summary = matcher.match([trial], "Boston, MA, United States", radius_miles=200)
        assert len(summary.results) == 0
        assert summary.trials_with_locations == 1  # location was present but couldn't geocode

    def test_partial_geocoding_failure_still_returns_good_trials(
        self, trial_cambridge, trial_chicago
    ):
        # Make a trial with an invalid location
        bad_trial = make_trial(
            nct_id="NCT_BAD",
            title="Bad Location Trial",
            locations=[Location(city="Ghost Town", state="ZZ")],
        )
        matcher = geo()
        summary = matcher.match(
            [bad_trial, trial_cambridge], "Boston, MA, United States", radius_miles=50
        )
        nct_ids = {r.trial.nct_id for r in summary.results}
        assert "NCT0000001" in nct_ids
        assert "NCT_BAD" not in nct_ids


# ---------------------------------------------------------------------------
# match — summary metadata
# ---------------------------------------------------------------------------


class TestSummaryMetadata:
    def test_total_trials_input_is_correct(self, boston_trials):
        matcher = geo()
        summary = matcher.match(boston_trials, "Boston, MA, United States")
        assert summary.total_trials_input == len(boston_trials)

    def test_trials_with_locations_counts_correctly(self, boston_trials, trial_no_locations):
        matcher = geo()
        summary = matcher.match(boston_trials, "Boston, MA, United States")
        # boston_trials has 5 trials with locations + 1 without
        assert summary.trials_with_locations == 5

    def test_user_coords_populated_on_success(self, trial_cambridge):
        matcher = geo()
        summary = matcher.match([trial_cambridge], "Boston, MA, United States")
        assert summary.user_coords is not None
        lat, lon = summary.user_coords
        assert 42.0 < lat < 43.0
        assert -72.0 < lon < -70.0

    def test_result_contains_facility_info(self, trial_cambridge):
        matcher = geo()
        summary = matcher.match([trial_cambridge], "Boston, MA, United States")
        result = summary.results[0]
        assert result.city == "Cambridge"
        assert result.state == "MA"
        assert result.country == "United States"

    def test_trials_within_radius_matches_results_length(self, boston_trials):
        matcher = geo()
        summary = matcher.match(boston_trials, "Boston, MA, United States", radius_miles=50)
        assert summary.trials_within_radius == len(summary.results)


# ---------------------------------------------------------------------------
# distance_between
# ---------------------------------------------------------------------------


class TestDistanceBetween:
    def test_boston_to_cambridge_is_short(self):
        matcher = geo()
        dist = matcher.distance_between(
            "Boston, MA, United States", "Cambridge, MA, United States"
        )
        assert dist is not None
        assert dist < 10

    def test_boston_to_chicago_is_large(self):
        matcher = geo()
        dist = matcher.distance_between(
            "Boston, MA, United States", "Chicago, IL, United States"
        )
        assert dist is not None
        assert dist > 800

    def test_unknown_location_returns_none(self):
        matcher = geo()
        result = matcher.distance_between("Boston, MA, United States", "Fictional, ZZ")
        assert result is None

    def test_same_location_returns_zero_or_near_zero(self):
        matcher = geo()
        dist = matcher.distance_between(
            "Boston, MA, United States", "Boston, MA, United States"
        )
        assert dist == 0.0

    def test_distance_is_symmetric(self):
        matcher = geo()
        d_ab = matcher.distance_between(
            "Boston, MA, United States", "Chicago, IL, United States"
        )
        d_ba = matcher.distance_between(
            "Chicago, IL, United States", "Boston, MA, United States"
        )
        assert d_ab == d_ba


# ---------------------------------------------------------------------------
# GeoMatcher repr
# ---------------------------------------------------------------------------


def test_repr_shows_cache_size():
    matcher = geo()
    assert "GeoMatcher" in repr(matcher)
    matcher._geocode("Boston, MA, United States")
    assert "1" in repr(matcher)
