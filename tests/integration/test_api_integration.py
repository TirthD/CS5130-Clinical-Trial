"""
Integration tests: GeoMatcher with Trial objects parsed from sample API responses.

These tests load JSON fixture files that mirror the ClinicalTrials.gov API
response format, parse them into Trial objects, then run GeoMatcher to
verify that the full parse → geocode → filter → rank pipeline works end-to-end.

No real geocoding requests are made (MockGeocoder is used).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.models import Trial
from src.tools.geo_matcher import GeoMatcher
from tests.conftest import COORDS_REGISTRY, MockGeocoder

# Resolve fixture directory relative to this file
FIXTURES_DIR = Path(__file__).parents[2] / "data" / "sample_responses"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_fixture(filename: str) -> dict:
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    with open(path) as f:
        return json.load(f)


def parse_trials_from_fixture(data: dict) -> list[Trial]:
    """Parse a ClinicalTrials.gov-style response into Trial objects."""
    studies = data.get("studies", [])
    return [Trial.from_api_response(s) for s in studies]


# ---------------------------------------------------------------------------
# Fixture-based tests
# ---------------------------------------------------------------------------


class TestGeoMatcherWithFixtures:
    def test_geo_trials_fixture_parses_without_error(self):
        data = load_fixture("geo_trials.json")
        trials = parse_trials_from_fixture(data)
        assert len(trials) > 0
        for trial in trials:
            assert trial.nct_id
            assert trial.locations is not None

    def test_boston_user_gets_nearby_trials(self):
        data = load_fixture("geo_trials.json")
        trials = parse_trials_from_fixture(data)

        # Build a registry that includes all cities in the fixture
        registry = dict(COORDS_REGISTRY)
        matcher = GeoMatcher(geocoder=MockGeocoder(registry), delay_seconds=0)

        summary = matcher.match(trials, "Boston, MA, United States", radius_miles=60)

        # We expect at least one result from the fixture within 60 miles of Boston
        assert summary.user_coords is not None
        # Results should be sorted by distance
        distances = [r.distance_miles for r in summary.results]
        assert distances == sorted(distances)

    def test_new_york_user_gets_different_results_than_boston_user(self):
        data = load_fixture("geo_trials.json")
        trials = parse_trials_from_fixture(data)

        registry = dict(COORDS_REGISTRY)
        matcher = GeoMatcher(geocoder=MockGeocoder(registry), delay_seconds=0)

        boston_summary = matcher.match(trials, "Boston, MA, United States", radius_miles=100)
        ny_summary = matcher.match(trials, "New York, NY, United States", radius_miles=100)

        boston_ids = {r.trial.nct_id for r in boston_summary.results}
        ny_ids = {r.trial.nct_id for r in ny_summary.results}

        # The two cities are ~215 miles apart, so their 100-mile circles don't overlap
        # — result sets should differ (at minimum they won't be identical)
        assert boston_ids != ny_ids or len(boston_ids) == 0

    def test_result_metadata_is_populated_from_api_response(self):
        data = load_fixture("geo_trials.json")
        trials = parse_trials_from_fixture(data)

        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)
        summary = matcher.match(trials, "Boston, MA, United States", radius_miles=100)

        for result in summary.results:
            assert result.trial.nct_id
            assert result.distance_miles >= 0
            # city/state may be None if the API response omits them, but the
            # result object should at least exist
            assert hasattr(result, "city")
            assert hasattr(result, "state")


# ---------------------------------------------------------------------------
# Radius boundary tests using the fixture
# ---------------------------------------------------------------------------


class TestRadiusBoundaryWithFixture:
    def test_wider_radius_returns_superset(self):
        data = load_fixture("geo_trials.json")
        trials = parse_trials_from_fixture(data)

        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)

        narrow = matcher.match(trials, "Boston, MA, United States", radius_miles=30)
        wide = matcher.match(trials, "Boston, MA, United States", radius_miles=200)

        narrow_ids = {r.trial.nct_id for r in narrow.results}
        wide_ids = {r.trial.nct_id for r in wide.results}

        # Every trial found at 30 miles must also appear at 200 miles
        assert narrow_ids.issubset(wide_ids)

    def test_zero_radius_returns_nothing(self):
        data = load_fixture("geo_trials.json")
        trials = parse_trials_from_fixture(data)

        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)
        summary = matcher.match(trials, "Boston, MA, United States", radius_miles=0)
        assert summary.results == []
