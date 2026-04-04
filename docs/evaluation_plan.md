# Evaluation Plan: GeoMatcher

## Goal

Verify that `GeoMatcher` correctly filters clinical trials by geographic proximity and ranks them by distance, across a range of user locations, radius values, and edge cases.

---

## What We Evaluate

| Dimension | Question |
|-----------|----------|
| **Filtering correctness** | Does the tool include exactly the trials within the radius and exclude all others? |
| **Ranking correctness** | Are included trials ordered closest-first? |
| **Distance accuracy** | How close is the reported distance to the true geodesic distance? |
| **Edge-case robustness** | Does the tool behave correctly for empty lists, missing locations, and bad user input? |

---

## Metrics

All metrics are defined in `tests/evaluation/metrics.py`.

### Precision @ radius
> Of the trials returned, what fraction truly belong within the radius?

```
precision = |returned ∩ expected| / |returned|
```

A low precision means the tool is including too many distant trials (false positives).

### Recall @ radius
> Of the trials that belong within the radius, what fraction did the tool return?

```
recall = |returned ∩ expected| / |expected|
```

A low recall means the tool is missing trials that are close enough (false negatives).

### F1 @ radius
> Harmonic mean of precision and recall. Primary composite metric.

```
F1 = 2 * precision * recall / (precision + recall)
```

**Minimum acceptable F1: 0.80**

### Ranking Accuracy
> Fraction of consecutive result pairs ordered correctly (closer trial first).

```
ranking_accuracy = correct_pairs / total_evaluated_pairs
```

A pair is "correct" if both trials appear in the expected order list and their order in the output matches. Pairs where either trial is not in the expected list are skipped.

**Minimum acceptable ranking accuracy: 0.80**

### Distance MAE (Mean Absolute Error)
> Average absolute difference between the tool's reported distance and the approximate true distance.

```
MAE = (1/n) * Σ |reported_distance_i - expected_distance_i|
```

Reported as informational only (no hard threshold) because expected distances in `test_cases.json` are approximate.

---

## Test Cases

Defined in `tests/evaluation/test_cases.json`. Each case specifies:

| Field | Description |
|-------|-------------|
| `id` | Unique case identifier |
| `description` | Human-readable summary |
| `user_location` | User's location string |
| `radius_miles` | Search radius |
| `coords_registry` | Known lat/lon for each location string (used by MockGeocoder) |
| `trials` | List of trials with locations and approximate expected distances |
| `expected_within_radius` | NCT IDs that should be returned |
| `expected_order` | Correct ordering of the returned trials by distance |
| `expected_excluded` | NCT IDs that should NOT be returned |

### Current test cases

| ID | Scenario |
|----|----------|
| `tc_boston_50mi` | Boston user, 50-mile radius — Cambridge and Worcester pass, Hartford and Chicago are excluded |
| `tc_boston_100mi` | Boston user, 100-mile radius — adds Hartford to results |
| `tc_multi_site_trial` | Trial has sites in both Cambridge (nearby) and Chicago (far) — nearest site wins |
| `tc_no_locations` | Trial with no location data — skipped entirely |
| `tc_bad_user_location` | User location cannot be geocoded — error returned, empty results |
| `tc_ny_user` | New York user, 50-mile radius — only local NY trial passes |

---

## How to Run

```bash
# From the project root
python -m tests.evaluation.eval_runner

# With per-case detail
python -m tests.evaluation.eval_runner --verbose
```

Exit codes:
- `0` — all cases pass (F1 ≥ 0.80, ranking accuracy ≥ 0.80)
- `1` — one or more cases below the quality bar
- `2` — fatal error (missing fixture file, etc.)

---

## How to Add a New Test Case

1. Add a new entry to `tests/evaluation/test_cases.json` with a unique `id`.
2. Populate `coords_registry` with all location strings the case uses, mapped to `[lat, lon]`.
3. Set `expected_within_radius`, `expected_order`, and `expected_excluded` based on known distances.
4. Run the eval runner to confirm the new case passes.

---

## MockGeocoder vs Real Geocoding

All evaluation runs use `MockGeocoder` (defined in `tests/conftest.py`), which resolves location strings against a hard-coded registry of coordinates. This means:

- **Evaluation is deterministic and offline** — no network calls, no rate-limit concerns.
- **Coverage is limited** — only location strings present in the registry can be tested.
- **Distance accuracy** — `geopy.distance.geodesic` is used on real WGS84 coordinates, so computed distances match true geodesic values for the registered locations.

For a higher-fidelity smoke test against real geocoding, run the integration tests with the real Nominatim geocoder:

```bash
pytest tests/integration/ -k "not fixture" --run-real-geo
```

(Requires a `--run-real-geo` flag or env var to avoid accidental network usage in CI.)

---

## Quality Bar Summary

| Metric | Minimum |
|--------|---------|
| F1 @ radius | ≥ 0.80 |
| Ranking accuracy | ≥ 0.80 |
| Distance MAE | informational |
| All edge-case scenarios | no crashes, correct error handling |
