# Tool Specification: GeoMatcher

## Overview

`GeoMatcher` filters and ranks clinical trials by geographic proximity to a user-supplied location. It is the third stage in the agent pipeline, running after `TrialSearcher` and before the final response is assembled.

```
User query → TrialSearcher → [list of Trial objects]
                                        ↓
                               GeoMatcher.match()
                                        ↓
                          [GeoMatchResult list, sorted by distance]
```

---

## Location: `src/tools/geo_matcher.py`

---

## Public Interface

### `GeoMatcher(geocoder=None, delay_seconds=1.1)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geocoder` | geopy geocoder | Nominatim | Any geocoder with `.geocode(query, timeout)`. Inject a mock in tests. |
| `delay_seconds` | float | 1.1 | Sleep between geocoding requests. Set to `0` in tests with a mock geocoder. |

---

### `GeoMatcher.match(trials, user_location, radius_miles=50.0) → GeoMatchSummary`

Filters and ranks a list of trials by proximity to the user.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `trials` | `list[Trial]` | Trials from `TrialSearcher`. Each `Trial.locations` contains site addresses. |
| `user_location` | `str` | Free-text location: `"Boston, MA"`, `"02101"`, `"New York, NY, United States"`. |
| `radius_miles` | `float` | Maximum distance from user to trial site. Default: `50`. Config: `DEFAULT_SEARCH_RADIUS_MILES`. |

**Returns: `GeoMatchSummary`**

| Field | Type | Description |
|-------|------|-------------|
| `user_location` | str | The input string as provided. |
| `user_coords` | `tuple[float, float] \| None` | Resolved lat/lon, or `None` on geocoding failure. |
| `radius_miles` | float | The radius used. |
| `total_trials_input` | int | Number of trials passed in. |
| `trials_with_locations` | int | Subset that had at least one location. |
| `trials_within_radius` | int | Subset whose nearest site is within `radius_miles`. |
| `results` | `list[GeoMatchResult]` | Matched results, sorted by distance ascending. |
| `errors` | `list[str]` | Non-fatal error messages (e.g., geocoding failure). |

**Each `GeoMatchResult` contains:**

| Field | Description |
|-------|-------------|
| `trial` | The `Trial` object. |
| `nearest_location` | The `Location` model of the closest site. |
| `distance_miles` | Rounded geodesic distance to the nearest site. |
| `facility_name` | Name of the trial site facility. |
| `city`, `state`, `country` | Location identifiers for display. |

---

### `GeoMatcher.distance_between(location_a, location_b) → float | None`

Utility: geodesic distance in miles between two free-text locations. Returns `None` if either cannot be geocoded.

---

## Pipeline Behaviour

### Step 1 — Geocode user location
The user's input is normalized (trimmed, lowercased) and passed to the geocoder. The result is cached. If geocoding fails, an error is recorded and an empty summary is returned immediately.

### Step 2 — Find nearest site per trial
For each trial, every `Location` in `Trial.locations` is converted to a geocodable string via `_location_to_string()`:

- If `zip_code` is present → `"<zip>, <country>"` (most precise)
- Otherwise → `"<city>, <state>, <country>"` (parts joined by `, `, skipping `None`)

The geodesic distance from the user's coordinates to each site is computed using `geopy.distance.geodesic`. The closest site is selected.

### Step 3 — Filter and sort
Trials whose nearest site exceeds `radius_miles` are excluded. The remaining results are sorted by `distance_miles` ascending.

---

## Geocoding Policy (Nominatim)

Nominatim is a free geocoder backed by OpenStreetMap. Its usage policy requires:

- No more than 1 request per second → enforced via `delay_seconds=1.1`
- A descriptive `User-Agent` header → set to `"clinical-trial-finder/1.0"`
- Results are cached in-process to minimize request volume

For production use at scale, replace Nominatim with a commercial geocoder (Google Maps, HERE, Mapbox) and pass it as `geocoder=`.

---

## Caching

Results are cached in a dict on the `GeoMatcher` instance:

```python
self._geocache: dict[str, Optional[tuple[float, float]]]
```

The cache key is the normalized (stripped, lowercased) location string. `None` is also cached to avoid retrying locations that previously returned no result.

**Cache scope:** per-instance, in-process only. There is no persistent disk cache.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| User location cannot be geocoded | Empty `results`, error in `summary.errors` |
| Trial location cannot be geocoded | Trial is silently skipped |
| `GeocoderTimedOut` | Retried up to 2 times with exponential back-off |
| `GeocoderServiceError` | Logged; location treated as ungeocodeable |
| Trial has no `locations` | Counted in `total_trials_input`, not in `trials_with_locations` |
| Empty `trials` list | Returns an empty `GeoMatchSummary` immediately |

---

## Agent Integration Example

```python
from src.tools.trial_searcher import TrialSearcher, SearchParams
from src.tools.geo_matcher import GeoMatcher
from src.api.client import ClinicalTrialsClient

client = ClinicalTrialsClient()
searcher = TrialSearcher(api_client=client)
matcher = GeoMatcher()

# Step 1: search
search_result = searcher.search(SearchParams(
    condition="type 2 diabetes",
    status="RECRUITING",
    max_results=50,
    sort_by="distance",
))

# Step 2: geo-filter and rank
geo_summary = matcher.match(
    search_result.trials,
    user_location="Boston, MA",
    radius_miles=50,
)

# Step 3: present results
for r in geo_summary.results:
    print(f"{r.trial.brief_title}  —  {r.distance_miles:.1f} mi  ({r.city}, {r.state})")
```

---

## Configuration

Defaults live in `config.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| `DEFAULT_SEARCH_RADIUS_MILES` | 50 | Default radius when none is specified |
| `DEFAULT_COUNTRY` | "United States" | Appended to locations missing a country field |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `geopy >= 2.4.0` | Geocoding and geodesic distance |
| `src.api.models.Trial` | Input type (contains `locations: list[Location]`) |
| `src.api.models.Location` | Location model (`city`, `state`, `country`, `zip_code`) |
