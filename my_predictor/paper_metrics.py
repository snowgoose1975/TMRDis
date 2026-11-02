"""Standard-library metrics matching the paper's residue-level protocol."""

from __future__ import annotations

from bisect import bisect_right
import math
from typing import Iterable

Pair = tuple[float, int]


def _counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, object]:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "mcc": (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate(
    pairs: Iterable[Pair],
    threshold: float | None = None,
    *,
    probabilities: bool | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, int | float | None]]]:
    """Evaluate valid (score, binary-label) pairs.

    If threshold is None, scan distinct observed scores, predict positive for
    scores >= threshold, maximize MCC, and retain the lowest threshold among
    tied optima. probabilities=None infers probability behavior from score range.
    """
    ordered = sorted(((float(score), int(label)) for score, label in pairs), key=lambda x: x[0], reverse=True)
    if not ordered:
        raise ValueError("cannot evaluate an empty score/label set")
    if any(not math.isfinite(score) for score, _ in ordered):
        raise ValueError("scores must be finite")
    if any(label not in (0, 1) for _, label in ordered):
        raise ValueError("labels must be binary 0/1")

    total = len(ordered)
    positives = sum(label for _, label in ordered)
    negatives = total - positives
    if not positives or not negatives:
        raise ValueError("metrics require at least one positive and one negative residue")

    rank_points: list[tuple[str, int]] = [(str(k), k) for k in (5, 10, 20, 50, 100)]
    rank_points.extend((f"top_{pct}pct", math.ceil(total * pct / 100)) for pct in (1, 5, 10))
    ranks: dict[str, dict[str, int | float | None]] = {
        name: {"k": k, "precision": None} for name, k in rank_points
    }

    probability_scores = (
        bool(probabilities)
        if probabilities is not None
        else 0.0 <= ordered[-1][0] and ordered[0][0] <= 1.0
    )
    if probability_scores and any(score < 0.0 or score > 1.0 for score, _ in ordered):
        raise ValueError("probability scores must all be within [0, 1]")
    bins = [[0, 0.0, 0] for _ in range(10)]
    bin_edges = [i * 0.1 for i in range(1, 10)]
    brier_sum = 0.0

    tp = seen = 0
    ap = auc_numerator = trapezoid = 0.0
    old_recall, old_precision = 0.0, 1.0
    best_mcc, best_threshold, best_counts = -math.inf, None, None
    selected_counts = None

    i = 0
    while i < total:
        score = ordered[i][0]
        start, above_tp = i, tp
        group_positive = 0
        while i < total and ordered[i][0] == score:
            label = ordered[i][1]
            group_positive += label
            if probability_scores:
                brier_sum += (score - label) ** 2
                calibration_bin = bins[bisect_right(bin_edges, score)]
                calibration_bin[0] += 1
                calibration_bin[1] += score
                calibration_bin[2] += label
            i += 1

        group_size = i - start
        group_negative = group_size - group_positive
        auc_numerator += group_negative * (above_tp + group_positive / 2.0)
        tp += group_positive
        seen = i
        fp, fn = seen - tp, positives - tp
        tn = negatives - fp
        recall, precision = tp / positives, tp / seen
        ap += group_positive / positives * precision
        trapezoid += (recall - old_recall) * (precision + old_precision) / 2.0
        old_recall, old_precision = recall, precision

        for item in ranks.values():
            k = int(item["k"])
            if start < k <= i:
                item["precision"] = (above_tp + (k - start) * group_positive / group_size) / k

        if threshold is None:
            mcc = float(_counts(tp, tn, fp, fn)["mcc"])
            if mcc >= best_mcc:
                best_mcc, best_threshold = mcc, score
                best_counts = (tp, tn, fp, fn)
        elif score >= threshold:
            selected_counts = (tp, tn, fp, fn)

    if threshold is None:
        threshold, selected_counts = best_threshold, best_counts
    elif selected_counts is None:
        selected_counts = (0, negatives, 0, positives)
    assert threshold is not None and selected_counts is not None

    result = _counts(*selected_counts)
    ece = sum(abs(score_sum - label_sum) / total for count, score_sum, label_sum in bins if count)
    result.update(
        average_precision=ap,
        auprc_trapezoid=trapezoid,
        auroc=auc_numerator / (positives * negatives),
        brier_score=brier_sum / total if probability_scores else None,
        expected_calibration_error=ece if probability_scores else None,
        valid_residues=total,
        positive_residues=positives,
        threshold=float(threshold),
        score_min=ordered[-1][0],
        score_max=ordered[0][0],
    )
    return result, ranks
