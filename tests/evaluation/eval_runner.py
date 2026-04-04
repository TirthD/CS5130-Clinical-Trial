"""
Evaluation runner for GeoMatcher.

Loads test cases from test_cases.json, runs GeoMatcher with a deterministic
MockGeocoder (no real HTTP calls), then computes precision, recall, F1,
ranking accuracy, and distance MAE for each case and in aggregate.

Usage
-----
Run from the project root:

    python -m tests.evaluation.eval_runner
    # or
    python tests/evaluation/eval_runner.py [--verbose]

Exit codes
----------
0   All cases pass the minimum quality bar (F1 >= 0.80, ranking accuracy >= 0.80)
1   One or more cases fall below the quality bar
2   Evaluation could not complete (e.g., missing fixture file)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as a script from the project root
PROJECT_ROOT = Path(__file__).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.models import Location, Trial
from src.tools.geo_matcher import GeoMatcher
from tests.conftest import MockGeocoder
from tests.evaluation.metrics import (
    CaseMetrics,
    aggregate,
    distance_mae,
    f1_score,
    precision_at_radius,
    ranking_accuracy,
    recall_at_radius,
)

logger = logging.getLogger(__name__)

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"

# Minimum acceptable metric values
MIN_F1 = 0.80
MIN_RANKING_ACCURACY = 0.80


# ---------------------------------------------------------------------------
# Trial construction from test-case JSON
# ---------------------------------------------------------------------------


def _build_trials_from_case(case: dict) -> list[Trial]:
    """Convert the 'trials' list in a test case to Trial model objects."""
    trials = []
    for t in case.get("trials", []):
        locs = [
            Location(
                city=loc.get("city"),
                state=loc.get("state"),
                country=loc.get("country"),
                zip_code=loc.get("zip_code"),
                facility=loc.get("facility"),
            )
            for loc in t.get("locations", [])
        ]
        trial = Trial(
            nct_id=t["nct_id"],
            brief_title=t.get("brief_title", ""),
            overall_status="RECRUITING",
            locations=locs,
        )
        trials.append(trial)
    return trials


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


def run_case(case: dict) -> CaseMetrics:
    """Evaluate GeoMatcher against a single test case."""
    case_id = case["id"]
    user_location = case["user_location"]
    radius_miles = case.get("radius_miles", 50)
    expected_within = case.get("expected_within_radius", [])
    expected_order = case.get("expected_order", [])

    # Build a registry limited to this test case's known coords
    registry = {k.lower(): tuple(v) for k, v in case.get("coords_registry", {}).items()}
    geocoder = MockGeocoder(registry=registry)
    matcher = GeoMatcher(geocoder=geocoder, delay_seconds=0)

    trials = _build_trials_from_case(case)
    summary = matcher.match(trials, user_location, radius_miles=radius_miles)

    returned_ids = [r.trial.nct_id for r in summary.results]
    returned_distances = [r.distance_miles for r in summary.results]

    # Build expected distance map from approximate values in the test case
    exp_dist_map: dict[str, float] = {}
    for t in case.get("trials", []):
        approx = t.get("expected_distance_miles_approx")
        if approx is not None:
            exp_dist_map[t["nct_id"]] = float(approx)

    prec = precision_at_radius(returned_ids, expected_within)
    rec = recall_at_radius(returned_ids, expected_within)
    f1 = f1_score(prec, rec)
    ra = ranking_accuracy(returned_ids, expected_order)
    mae = distance_mae(returned_ids, returned_distances, exp_dist_map)

    return CaseMetrics(
        case_id=case_id,
        precision=prec,
        recall=rec,
        f1=f1,
        ranking_accuracy=ra,
        distance_mae=mae,
        returned_ids=returned_ids,
        expected_ids=expected_within,
        errors=summary.errors,
    )


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------


def run_evaluation(verbose: bool = False) -> int:
    """
    Run all test cases and print a report.

    Returns 0 on success, 1 if quality bar not met, 2 on fatal error.
    """
    if not TEST_CASES_PATH.exists():
        print(f"ERROR: test cases file not found: {TEST_CASES_PATH}", file=sys.stderr)
        return 2

    with open(TEST_CASES_PATH) as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    if not cases:
        print("No test cases found.", file=sys.stderr)
        return 2

    case_results: list[CaseMetrics] = []
    failures: list[str] = []

    for case in cases:
        result = run_case(case)
        case_results.append(result)

        passed = result.f1 >= MIN_F1 and result.ranking_accuracy >= MIN_RANKING_ACCURACY
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures.append(result.case_id)

        print(
            f"[{status}] {result.case_id:30s}  "
            f"P={result.precision:.2f}  R={result.recall:.2f}  "
            f"F1={result.f1:.2f}  Rank={result.ranking_accuracy:.2f}"
            + (f"  MAE={result.distance_mae:.1f} mi" if result.distance_mae is not None else "")
        )

        if verbose:
            print(f"       returned : {result.returned_ids}")
            print(f"       expected : {result.expected_ids}")
            if result.errors:
                print(f"       errors   : {result.errors}")

    summary = aggregate(case_results)
    print()
    print(summary)

    if failures:
        print(f"\nFAILED cases ({len(failures)}): {', '.join(failures)}")
        return 1

    print("\nAll cases passed.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Evaluate GeoMatcher accuracy")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-case details")
    args = parser.parse_args()

    sys.exit(run_evaluation(verbose=args.verbose))
