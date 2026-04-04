"""
Evaluation metrics for GeoMatcher.

All metrics are computed per test case and then averaged across the full
evaluation suite. Each function accepts plain Python lists so it can be
used independently of any test framework.

Metrics
-------
precision_at_radius
    Of the trials GeoMatcher returned, what fraction truly belong within
    the radius?  (measures false positives)

recall_at_radius
    Of the trials that truly belong within the radius, what fraction did
    GeoMatcher return?  (measures false negatives)

f1_at_radius
    Harmonic mean of precision and recall.

ranking_accuracy
    Fraction of consecutive result pairs that are correctly ordered
    (i.e., earlier result is closer).  1.0 = perfect ranking.

distance_error_mae
    Mean absolute error between GeoMatcher's reported distance and the
    ground-truth distance for each matched result.  Only applicable when
    expected distances are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Per-case result container
# ---------------------------------------------------------------------------


@dataclass
class CaseMetrics:
    """Metric values for a single test case."""

    case_id: str
    precision: float
    recall: float
    f1: float
    ranking_accuracy: float
    distance_mae: Optional[float]  # None when no expected distances are given
    returned_ids: list[str] = field(default_factory=list)
    expected_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Aggregate metrics across all test cases."""

    num_cases: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_ranking_accuracy: float
    mean_distance_mae: Optional[float]
    case_results: list[CaseMetrics] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"=== GeoMatcher Evaluation Summary ({self.num_cases} cases) ===",
            f"  Precision @ radius : {self.mean_precision:.3f}",
            f"  Recall    @ radius : {self.mean_recall:.3f}",
            f"  F1        @ radius : {self.mean_f1:.3f}",
            f"  Ranking accuracy   : {self.mean_ranking_accuracy:.3f}",
        ]
        if self.mean_distance_mae is not None:
            lines.append(f"  Distance MAE (mi)  : {self.mean_distance_mae:.1f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def precision_at_radius(
    returned_ids: list[str],
    expected_within_radius: list[str],
) -> float:
    """
    Fraction of returned trials that are truly within the radius.

    Returns 1.0 when nothing was returned (no false positives).
    """
    if not returned_ids:
        return 1.0
    expected_set = set(expected_within_radius)
    true_positives = sum(1 for nct_id in returned_ids if nct_id in expected_set)
    return true_positives / len(returned_ids)


def recall_at_radius(
    returned_ids: list[str],
    expected_within_radius: list[str],
) -> float:
    """
    Fraction of truly-within-radius trials that were returned.

    Returns 1.0 when there are no expected results (nothing to miss).
    """
    if not expected_within_radius:
        return 1.0
    returned_set = set(returned_ids)
    found = sum(1 for nct_id in expected_within_radius if nct_id in returned_set)
    return found / len(expected_within_radius)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall. Returns 0.0 when both are 0."""
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2 * precision * recall / denom


def ranking_accuracy(
    returned_ids: list[str],
    expected_order: list[str],
) -> float:
    """
    Fraction of consecutive pairs in *returned_ids* that follow the expected
    ordering defined by *expected_order*.

    A pair (a, b) is "correct" if a appears before b in expected_order,
    and a appears before b in returned_ids.

    Returns 1.0 when there are fewer than 2 results (trivially correct).
    """
    if len(returned_ids) < 2:
        return 1.0

    # Build a position map from the expected order
    expected_pos: dict[str, int] = {nct_id: i for i, nct_id in enumerate(expected_order)}

    correct = 0
    total = 0
    for i in range(len(returned_ids) - 1):
        a = returned_ids[i]
        b = returned_ids[i + 1]

        # Only evaluate pairs where both items exist in the expected order
        if a not in expected_pos or b not in expected_pos:
            continue

        total += 1
        if expected_pos[a] < expected_pos[b]:
            correct += 1

    return correct / total if total > 0 else 1.0


def distance_mae(
    returned_ids: list[str],
    returned_distances: list[float],
    expected_distances: dict[str, float],
) -> Optional[float]:
    """
    Mean absolute error between reported and expected distances.

    Args:
        returned_ids:       NCT IDs in the order returned by GeoMatcher.
        returned_distances: Corresponding distances (same order).
        expected_distances: Mapping {nct_id: expected_distance_miles}.

    Returns None when no expected distances are provided.
    """
    if not expected_distances:
        return None

    errors = []
    for nct_id, dist in zip(returned_ids, returned_distances):
        if nct_id in expected_distances:
            errors.append(abs(dist - expected_distances[nct_id]))

    return sum(errors) / len(errors) if errors else None


# ---------------------------------------------------------------------------
# Aggregate across cases
# ---------------------------------------------------------------------------


def aggregate(case_results: list[CaseMetrics]) -> EvalSummary:
    """Compute mean metrics across all evaluated test cases."""
    n = len(case_results)
    if n == 0:
        return EvalSummary(
            num_cases=0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_f1=0.0,
            mean_ranking_accuracy=0.0,
            mean_distance_mae=None,
            case_results=[],
        )

    mean_p = sum(c.precision for c in case_results) / n
    mean_r = sum(c.recall for c in case_results) / n
    mean_f = sum(c.f1 for c in case_results) / n
    mean_ra = sum(c.ranking_accuracy for c in case_results) / n

    mae_values = [c.distance_mae for c in case_results if c.distance_mae is not None]
    mean_mae = sum(mae_values) / len(mae_values) if mae_values else None

    return EvalSummary(
        num_cases=n,
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_f1=mean_f,
        mean_ranking_accuracy=mean_ra,
        mean_distance_mae=mean_mae,
        case_results=case_results,
    )
