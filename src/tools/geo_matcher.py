"""
Geo Matcher Tool
================
Matches clinical trials to a user's location by:
  1. Geocoding the user's input (city/state or zip code) to lat/lon
  2. Geocoding each trial site's address to lat/lon
  3. Computing geodesic distances between user and each site
  4. Filtering trials whose nearest site is within a search radius
  5. Returning results sorted by distance ascending

Uses geopy's Nominatim geocoder (free, no API key) and geodesic distance.
Results are cached in-process to stay within Nominatim's 1 req/sec policy.

Usage:
    matcher = GeoMatcher()
    summary = matcher.match(trials, user_location="Boston, MA", radius_miles=50)
    for r in summary.results:
        print(f"{r.trial.brief_title}: {r.distance_miles:.1f} mi  ({r.city}, {r.state})")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from geopy.distance import geodesic
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from src.api.models import Location, Trial

logger = logging.getLogger(__name__)

_NOMINATIM_USER_AGENT = "clinical-trial-finder/1.0"
# Nominatim usage policy: no faster than 1 request per second
_GEOCODE_DELAY_SECONDS = 1.1


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class GeoMatchResult:
    """A trial paired with its nearest site and distance from the user."""

    trial: Trial
    nearest_location: Location
    distance_miles: float
    facility_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


@dataclass
class GeoMatchSummary:
    """Aggregate output of a single geo-matching run."""

    user_location: str
    user_coords: Optional[tuple[float, float]]
    radius_miles: float
    total_trials_input: int
    trials_with_locations: int = 0
    trials_within_radius: int = 0
    results: list[GeoMatchResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------


class GeoMatcher:
    """
    Filters and ranks clinical trials by proximity to a user's location.

    Inject a custom geocoder in tests to avoid real HTTP calls:

        mock_geocoder = MockGeocoder(registry={"Boston, MA": (42.36, -71.06)})
        matcher = GeoMatcher(geocoder=mock_geocoder)
    """

    def __init__(
        self,
        geocoder=None,
        delay_seconds: float = _GEOCODE_DELAY_SECONDS,
    ):
        """
        Args:
            geocoder: Any geopy geocoder with a ``geocode(query, timeout)`` method.
                      Defaults to Nominatim.
            delay_seconds: Sleep between consecutive geocoding calls.
                           Set to 0 in tests when using a mock geocoder.
        """
        self._geocoder = geocoder or Nominatim(user_agent=_NOMINATIM_USER_AGENT)
        self._delay = delay_seconds
        # Keyed by normalized location string; value is (lat, lon) or None
        self._geocache: dict[str, Optional[tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        trials: list[Trial],
        user_location: str,
        radius_miles: float = 50.0,
    ) -> GeoMatchSummary:
        """
        Filter and rank trials by proximity to *user_location*.

        Args:
            trials:        Trials to filter (each may have a ``locations`` list).
            user_location: Free-text location — city/state, zip code, etc.
            radius_miles:  Maximum distance for a trial site to be included.

        Returns:
            GeoMatchSummary with ``results`` sorted by distance ascending.
        """
        summary = GeoMatchSummary(
            user_location=user_location,
            user_coords=None,
            radius_miles=radius_miles,
            total_trials_input=len(trials),
        )

        # Step 1: Resolve the user's location to coordinates
        user_coords = self._geocode(user_location)
        if user_coords is None:
            msg = f"Could not geocode user location: '{user_location}'"
            logger.error(msg)
            summary.errors.append(msg)
            return summary

        summary.user_coords = user_coords
        logger.info("User '%s' → %s", user_location, user_coords)

        # Step 2: Find the nearest site for every trial
        results: list[GeoMatchResult] = []

        for trial in trials:
            if not trial.locations:
                continue
            summary.trials_with_locations += 1

            match_pair = self._nearest_location(trial, user_coords)
            if match_pair is None:
                continue  # none of this trial's sites could be geocoded

            loc, dist_miles = match_pair
            if dist_miles <= radius_miles:
                results.append(
                    GeoMatchResult(
                        trial=trial,
                        nearest_location=loc,
                        distance_miles=round(dist_miles, 1),
                        facility_name=loc.facility,
                        city=loc.city,
                        state=loc.state,
                        country=loc.country,
                    )
                )

        # Step 3: Sort closest first
        results.sort(key=lambda r: r.distance_miles)

        summary.results = results
        summary.trials_within_radius = len(results)

        logger.info(
            "Geo match: %d/%d trials within %.0f miles of '%s'",
            len(results),
            len(trials),
            radius_miles,
            user_location,
        )
        return summary

    def distance_between(self, location_a: str, location_b: str) -> Optional[float]:
        """
        Return the geodesic distance in miles between two free-text locations.
        Returns None if either location cannot be geocoded.
        """
        coords_a = self._geocode(location_a)
        coords_b = self._geocode(location_b)
        if coords_a is None or coords_b is None:
            return None
        return round(geodesic(coords_a, coords_b).miles, 2)

    # ------------------------------------------------------------------
    # Internal geocoding
    # ------------------------------------------------------------------

    def _geocode(self, location_str: str) -> Optional[tuple[float, float]]:
        """Return (lat, lon) for a location string, using the in-process cache."""
        if not location_str or not location_str.strip():
            return None

        key = location_str.strip().lower()
        if key in self._geocache:
            return self._geocache[key]

        coords = self._geocode_with_retry(location_str)
        self._geocache[key] = coords
        return coords

    def _geocode_with_retry(
        self, location_str: str, retries: int = 2
    ) -> Optional[tuple[float, float]]:
        """Geocode with exponential back-off on timeout."""
        for attempt in range(retries + 1):
            try:
                time.sleep(self._delay * (attempt + 1) if attempt > 0 else self._delay)
                result = self._geocoder.geocode(location_str, timeout=10)
                if result:
                    return (result.latitude, result.longitude)
                logger.debug("No geocode result for: '%s'", location_str)
                return None
            except GeocoderTimedOut:
                logger.warning(
                    "Geocoding timed out for '%s' (attempt %d/%d)",
                    location_str, attempt + 1, retries + 1,
                )
            except GeocoderServiceError as exc:
                logger.warning("Geocoding service error for '%s': %s", location_str, exc)
                return None

        logger.error(
            "Geocoding failed after %d attempts for: '%s'", retries + 1, location_str
        )
        return None

    # ------------------------------------------------------------------
    # Internal distance computation
    # ------------------------------------------------------------------

    def _nearest_location(
        self,
        trial: Trial,
        user_coords: tuple[float, float],
    ) -> Optional[tuple[Location, float]]:
        """
        Find the trial site closest to the user.

        Returns (Location, distance_miles) or None if no site could be geocoded.
        """
        best_loc: Optional[Location] = None
        best_dist = float("inf")

        for loc in trial.locations or []:
            loc_str = self._location_to_string(loc)
            if not loc_str:
                continue

            coords = self._geocode(loc_str)
            if coords is None:
                continue

            dist = geodesic(user_coords, coords).miles
            if dist < best_dist:
                best_dist = dist
                best_loc = loc

        if best_loc is None:
            return None
        return (best_loc, best_dist)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _location_to_string(loc: Location) -> str:
        """
        Build a geocodable query string from a Location model.

        Prefers zip + country (most precise) and falls back to city, state, country.
        """
        if loc.zip_code:
            # zip + country gives the geocoder enough context
            country_part = f", {loc.country}" if loc.country else ""
            return f"{loc.zip_code}{country_part}"

        parts = [p for p in [loc.city, loc.state, loc.country] if p]
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"GeoMatcher(cached_locations={len(self._geocache)})"
