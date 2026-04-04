"""
Integration tests: GeoMatcher inside the agent pipeline.

These tests exercise GeoMatcher together with the TrialSearcher to verify
that the "distance" sort mode correctly delegates to geo_matcher and that
the end-to-end pipeline produces correctly ordered, radius-filtered results.

All network calls are mocked — no real API or geocoding requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.models import SearchResult as ApiSearchResult
from src.tools.geo_matcher import GeoMatcher
from src.tools.trial_searcher import SearchParams, TrialSearcher
from tests.conftest import MockGeocoder, make_location, make_trial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_api_client(trials):
    """Return a mock API client that returns a fixed list of trials."""
    client = MagicMock()
    client.search_trials.return_value = ApiSearchResult(
        total_count=len(trials),
        trials=trials,
    )
    return client


# ---------------------------------------------------------------------------
# Pipeline: TrialSearcher → GeoMatcher
# ---------------------------------------------------------------------------


class TestTrialSearcherGeoMatcherPipeline:
    """
    Verify that results from TrialSearcher can be piped into GeoMatcher
    and that the combined output is correctly filtered and sorted by distance.
    """

    def test_distance_sort_filters_and_orders_results(self):
        cambridge_trial = make_trial(
            "NCT_CAM", "Cambridge Study",
            locations=[make_location("Cambridge", "MA")],
        )
        worcester_trial = make_trial(
            "NCT_WOR", "Worcester Study",
            locations=[make_location("Worcester", "MA")],
        )
        chicago_trial = make_trial(
            "NCT_CHI", "Chicago Study",
            locations=[make_location("Chicago", "IL")],
        )

        client = _make_api_client([chicago_trial, worcester_trial, cambridge_trial])
        searcher = TrialSearcher(api_client=client)

        params = SearchParams(
            condition="diabetes",
            location="Boston, MA",
            status="RECRUITING",
            max_results=10,
            sort_by="distance",  # signals that geo_matcher should be applied
        )

        # Searcher returns all three (status filter passes, no demographics filter)
        search_result = searcher.search(params)
        assert len(search_result.trials) == 3

        # Now pipe into geo_matcher
        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)
        geo_summary = matcher.match(
            search_result.trials,
            user_location="Boston, MA, United States",
            radius_miles=50,
        )

        # Only Cambridge and Worcester should survive the 50-mile filter
        nct_ids = [r.trial.nct_id for r in geo_summary.results]
        assert "NCT_CAM" in nct_ids
        assert "NCT_WOR" in nct_ids
        assert "NCT_CHI" not in nct_ids

        # Cambridge (~4 mi) should rank before Worcester (~40 mi)
        assert nct_ids.index("NCT_CAM") < nct_ids.index("NCT_WOR")

    def test_no_trials_with_locations_returns_empty_geo_result(self):
        no_loc_trial = make_trial("NCT_NOLOC", "Remote Study", locations=[])
        client = _make_api_client([no_loc_trial])
        searcher = TrialSearcher(api_client=client)

        search_result = searcher.search(
            SearchParams(condition="asthma", status="RECRUITING")
        )

        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)
        geo_summary = matcher.match(
            search_result.trials,
            user_location="Boston, MA, United States",
            radius_miles=50,
        )

        assert geo_summary.results == []
        assert geo_summary.trials_with_locations == 0

    def test_api_error_propagates_before_geo_matching(self):
        from src.api.exceptions import ClinicalTrialsAPIError

        client = MagicMock()
        client.search_trials.side_effect = ClinicalTrialsAPIError("timeout")
        searcher = TrialSearcher(api_client=client)

        result = searcher.search(SearchParams(condition="cancer", status="RECRUITING"))

        assert result.trials == []
        assert len(result.errors) == 1

        # Geo matcher should handle the empty list gracefully
        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)
        geo_summary = matcher.match(
            result.trials, "Boston, MA, United States", radius_miles=50
        )
        assert geo_summary.results == []


# ---------------------------------------------------------------------------
# Geo-aware search simulation
# ---------------------------------------------------------------------------


class TestGeoAwareSearchSimulation:
    """
    Simulate what the agent would do for a geo-filtered search:
    search → filter by status → geo-filter → return ranked results.
    """

    def test_full_pipeline_rank_by_distance(self):
        trials = [
            make_trial("NCT001", "Trial A", [make_location("Chicago", "IL")]),
            make_trial("NCT002", "Trial B", [make_location("Worcester", "MA")]),
            make_trial("NCT003", "Trial C", [make_location("Cambridge", "MA")]),
            make_trial("NCT004", "Trial D", [make_location("Providence", "RI")]),
        ]
        client = _make_api_client(trials)
        searcher = TrialSearcher(api_client=client)

        result = searcher.search(SearchParams(condition="heart disease", status="RECRUITING"))
        assert len(result.trials) == 4

        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)
        geo_summary = matcher.match(result.trials, "Boston, MA, United States", radius_miles=60)

        ids_in_order = [r.trial.nct_id for r in geo_summary.results]
        # Expected order by distance from Boston (approximate):
        # NCT003 Cambridge ~4 mi, NCT002 Worcester ~40 mi, NCT004 Providence ~50 mi
        # NCT001 Chicago ~980 mi → excluded
        assert ids_in_order[0] == "NCT003"
        assert "NCT001" not in ids_in_order

    def test_large_radius_includes_more_trials(self):
        trials = [
            make_trial("NCT001", "Boston-area", [make_location("Cambridge", "MA")]),
            make_trial("NCT002", "NY-area", [make_location("New York", "NY")]),
        ]
        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)

        narrow = matcher.match(trials, "Boston, MA, United States", radius_miles=50)
        wide = matcher.match(trials, "Boston, MA, United States", radius_miles=300)

        assert len(narrow.results) < len(wide.results)
        assert len(wide.results) == 2

    def test_increasing_radius_never_removes_results(self):
        trials = [
            make_trial("NCT001", "Cambridge Study", [make_location("Cambridge", "MA")]),
            make_trial("NCT002", "Worcester Study", [make_location("Worcester", "MA")]),
            make_trial("NCT003", "Hartford Study", [make_location("Hartford", "CT")]),
        ]
        matcher = GeoMatcher(geocoder=MockGeocoder(), delay_seconds=0)

        prev_count = 0
        for radius in [10, 50, 120, 500]:
            summary = matcher.match(trials, "Boston, MA, United States", radius_miles=radius)
            assert len(summary.results) >= prev_count
            prev_count = len(summary.results)
