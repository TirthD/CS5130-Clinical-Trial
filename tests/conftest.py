"""
Shared pytest fixtures for unit, integration, and evaluation tests.

Fixtures here are available to all test files without explicit imports.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.api.models import Contact, Eligibility, Intervention, Location, Trial


# ---------------------------------------------------------------------------
# Mock geocoder — used in any test that touches GeoMatcher
# ---------------------------------------------------------------------------

# Known coordinates used across the test suite (lat, lon)
COORDS_REGISTRY: dict[str, tuple[float, float]] = {
    # Boston area
    "boston, ma, united states": (42.3601, -71.0589),
    "cambridge, ma, united states": (42.3736, -71.1097),
    "worcester, ma, united states": (42.2626, -71.8023),
    "02101, united states": (42.3601, -71.0589),  # Boston zip
    # Providence is ~50 miles from Boston
    "providence, ri, united states": (41.8240, -71.4128),
    # Hartford is ~100 miles from Boston
    "hartford, ct, united states": (41.7658, -72.6851),
    # New York is ~215 miles from Boston
    "new york, ny, united states": (40.7128, -74.0060),
    # Chicago is ~980 miles from Boston
    "chicago, il, united states": (41.8781, -87.6298),
    # Los Angeles is ~2,980 miles from Boston
    "los angeles, ca, united states": (34.0522, -118.2437),
}


class MockGeocodeResult:
    """Minimal stand-in for a geopy Location result object."""

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude = latitude
        self.longitude = longitude


class MockGeocoder:
    """
    Deterministic geocoder that resolves strings against COORDS_REGISTRY.

    Returning None for any location not in the registry lets tests verify
    graceful degradation when geocoding fails.
    """

    def __init__(self, registry: dict[str, tuple[float, float]] | None = None) -> None:
        self._registry = {k.lower(): v for k, v in (registry or COORDS_REGISTRY).items()}

    def geocode(self, query: str, timeout: int = 10) -> Optional[MockGeocodeResult]:
        coords = self._registry.get(query.strip().lower())
        if coords:
            return MockGeocodeResult(*coords)
        return None


@pytest.fixture
def mock_geocoder() -> MockGeocoder:
    """A MockGeocoder pre-loaded with COORDS_REGISTRY."""
    return MockGeocoder()


# ---------------------------------------------------------------------------
# Location / Trial factories
# ---------------------------------------------------------------------------


def make_location(
    city: str,
    state: str,
    country: str = "United States",
    facility: str | None = None,
    zip_code: str | None = None,
    status: str = "RECRUITING",
) -> Location:
    return Location(
        facility=facility or f"{city} Medical Center",
        city=city,
        state=state,
        country=country,
        zip_code=zip_code,
        status=status,
    )


def make_trial(
    nct_id: str,
    title: str,
    locations: list[Location] | None = None,
    status: str = "RECRUITING",
) -> Trial:
    return Trial(
        nct_id=nct_id,
        brief_title=title,
        overall_status=status,
        locations=locations or [],
        eligibility=Eligibility(gender="ALL", minimum_age="18 Years"),
    )


# ---------------------------------------------------------------------------
# Reusable trial fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trial_cambridge() -> Trial:
    """Trial with a single site in Cambridge, MA (~4 miles from Boston)."""
    return make_trial(
        nct_id="NCT0000001",
        title="Cambridge Diabetes Study",
        locations=[make_location("Cambridge", "MA")],
    )


@pytest.fixture
def trial_worcester() -> Trial:
    """Trial with a single site in Worcester, MA (~40 miles from Boston)."""
    return make_trial(
        nct_id="NCT0000002",
        title="Worcester Heart Study",
        locations=[make_location("Worcester", "MA")],
    )


@pytest.fixture
def trial_providence() -> Trial:
    """Trial with a site in Providence, RI (~50 miles from Boston)."""
    return make_trial(
        nct_id="NCT0000003",
        title="Providence Oncology Trial",
        locations=[make_location("Providence", "RI")],
    )


@pytest.fixture
def trial_hartford() -> Trial:
    """Trial with a site in Hartford, CT (~100 miles from Boston)."""
    return make_trial(
        nct_id="NCT0000004",
        title="Hartford Neurology Study",
        locations=[make_location("Hartford", "CT")],
    )


@pytest.fixture
def trial_chicago() -> Trial:
    """Trial with a site in Chicago, IL (~980 miles from Boston)."""
    return make_trial(
        nct_id="NCT0000005",
        title="Chicago Cardiology Trial",
        locations=[make_location("Chicago", "IL")],
    )


@pytest.fixture
def trial_no_locations() -> Trial:
    """Trial with no location data."""
    return make_trial(nct_id="NCT0000099", title="Remote-Only Study", locations=[])


@pytest.fixture
def trial_multi_site(trial_cambridge, trial_worcester) -> Trial:
    """Trial that has sites in both Cambridge (4 mi) and Chicago (980 mi)."""
    return make_trial(
        nct_id="NCT0000006",
        title="Multi-Site Cardiology Trial",
        locations=[
            make_location("Chicago", "IL"),
            make_location("Cambridge", "MA"),  # nearest to Boston user
        ],
    )


@pytest.fixture
def boston_trials(
    trial_cambridge,
    trial_worcester,
    trial_providence,
    trial_hartford,
    trial_chicago,
    trial_no_locations,
) -> list[Trial]:
    """A mixed list of trials at varying distances from Boston."""
    return [
        trial_cambridge,
        trial_worcester,
        trial_providence,
        trial_hartford,
        trial_chicago,
        trial_no_locations,
    ]
