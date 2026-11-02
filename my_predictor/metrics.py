from __future__ import annotations

import math
from typing import Any

import numpy as np


def _masked_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(f"scores shape {scores.shape} != labels shape {labels.shape}")
    valid = np.isfinite(scores) & np.isin(labels, (0, 1))
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.shape != scores.shape:
            raise ValueError(f"mask shape {mask.shape} != scores shape {scores.shape}")
        valid &= mask
    return scores[valid], labels[valid].astype(np.int64), int(valid.sum())


def average_precision(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    scores, labels, count = _masked_arrays(scores, labels, valid_mask)
    positives = int(labels.sum())
    if count == 0 or positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    cumulative = np.cumsum(labels[order])
    # Evaluate precision only after every example with the same score has
    # entered the prediction set, so ties cannot depend on input ordering.
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), count - 1]
    true_positives = cumulative[ends]
    precision = true_positives / (ends + 1)
    recall_increments = np.diff(np.r_[0, true_positives]) / positives
    return float(np.sum(recall_increments * precision))


def auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    scores, labels, count = _masked_arrays(scores, labels, valid_mask)
    positives = int(labels.sum())
    negatives = count - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start + 1
        while end < count and ordered_scores[end] == ordered_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    u = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


def binary_counts(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    scores, labels, _ = _masked_arrays(scores, labels, valid_mask)
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    return (
        int(np.sum(predictions & positives)),
        int(np.sum(~predictions & negatives)),
        int(np.sum(predictions & negatives)),
        int(np.sum(~predictions & positives)),
    )


def mcc_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def f1_from_counts(tp: int, _tn: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return float(2 * tp / denominator)


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    objective: str = "mcc",
    candidates: int = 101,
) -> float:
    scores, labels, count = _masked_arrays(scores, labels, valid_mask)
    if count == 0:
        raise ValueError("cannot select a threshold without valid residues")
    if objective not in {"mcc", "f1"}:
        raise ValueError("objective must be mcc or f1")
    quantiles = np.linspace(0.0, 1.0, max(3, candidates))
    thresholds = np.unique(np.quantile(scores, quantiles))
    thresholds = np.unique(np.concatenate([thresholds, np.array([0.5])]))
    best_threshold = float(thresholds[0])
    best_value = -float("inf")
    for threshold in thresholds:
        counts = binary_counts(scores, labels, float(threshold))
        value = (
            mcc_from_counts(*counts)
            if objective == "mcc"
            else f1_from_counts(*counts)
        )
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def _probability_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    scores, labels, count = _masked_arrays(scores, labels, valid_mask)
    if count and not np.all((scores >= 0.0) & (scores <= 1.0)):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64), 0
    return scores, labels, count


def brier_score(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    scores, labels, count = _probability_arrays(scores, labels, valid_mask)
    if count == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    scores, labels, count = _probability_arrays(scores, labels, valid_mask)
    if count == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(scores, edges[1:-1], right=False)
    error = 0.0
    for index in range(bins):
        selected = indices == index
        if not np.any(selected):
            continue
        error += float(np.sum(selected) / count) * abs(
            float(np.mean(scores[selected])) - float(np.mean(labels[selected]))
        )
    return float(error)


def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    total_residues: int | None = None,
) -> dict[str, Any]:
    _, _, valid_count = _masked_arrays(scores, labels, valid_mask)
    tp, tn, fp, fn = binary_counts(scores, labels, threshold, valid_mask)
    denominator = valid_count if total_residues is None else total_residues
    return {
        "average_precision": average_precision(scores, labels, valid_mask),
        "auroc": auroc(scores, labels, valid_mask),
        "brier_score": brier_score(scores, labels, valid_mask),
        "expected_calibration_error": expected_calibration_error(
            scores, labels, valid_mask
        ),
        "mcc": mcc_from_counts(tp, tn, fp, fn),
        "f1": f1_from_counts(tp, tn, fp, fn),
        "threshold": float(threshold),
        "valid_residues": valid_count,
        "coverage": float(valid_count / denominator) if denominator else float("nan"),
        "positive_residues": tp + fn,
        "positive_fraction": float((tp + fn) / valid_count) if valid_count else float("nan"),
    }
