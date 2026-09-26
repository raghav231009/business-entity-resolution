"""
metrics.py — Unified evaluation metrics for Amazon ML Entity Resolution
======================================================================
Implements standard metrics and the official challenge metric:
Macro-averaged Entity-Level F₀.₅ with exact singleton handling.

Challenge formula:
    F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

Singleton rules:
    - true == ∅ and pred == ∅  → F₀.₅ = 1.0
    - true == ∅ and pred != ∅ → F₀.₅ = 0.0
    - Macro F₀.₅ = arithmetic mean across all Source 1 entities (including singletons).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def precision_score(tp: int, fp: int) -> float:
    """Compute precision from true positives and false positives."""
    total = tp + fp
    return float(tp / total) if total > 0 else 0.0


def recall_score(tp: int, fn: int) -> float:
    """Compute recall from true positives and false negatives."""
    total = tp + fn
    return float(tp / total) if total > 0 else 0.0


def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F_β score from precision and recall.

    F_β = (1 + β²) * (precision * recall) / (β² * precision + recall)
    For β = 0.5:
    F₀.₅ = (1.25 * precision * recall) / (0.25 * precision + recall)
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    b2 = beta ** 2
    denominator = (b2 * precision) + recall
    if denominator <= 0.0:
        return 0.0
    return float((1.0 + b2) * precision * recall / denominator)


# Alias for backward compatibility
f_beta = f_beta_score


def entity_f05(
    true_matches: Set[str],
    predicted_matches: Set[str],
) -> Dict[str, Any]:
    """
    Compute entity-level precision, recall, and F₀.₅ for a single Source 1 entity.

    Parameters
    ----------
    true_matches : set[str]
        Set of true matching entity IDs for this Source 1 entity.
    predicted_matches : set[str]
        Set of predicted matching entity IDs for this Source 1 entity.

    Returns
    -------
    dict with keys:
        precision, recall, f05, tp, fp, fn, true_count, pred_count
    """
    true_len = len(true_matches)
    pred_len = len(predicted_matches)

    # Case 1: Singleton correctly predicted empty
    if true_len == 0 and pred_len == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f05": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "true_count": 0,
            "pred_count": 0,
        }

    # Case 2: True singleton with false positive predictions
    if true_len == 0 and pred_len > 0:
        return {
            "precision": 0.0,
            "recall": 1.0,  # vacuously: nothing to recall, but precision is 0
            "f05": 0.0,
            "tp": 0,
            "fp": pred_len,
            "fn": 0,
            "true_count": 0,
            "pred_count": pred_len,
        }

    # Case 3: Entity with true matches
    tp = len(true_matches & predicted_matches)
    fp = len(predicted_matches - true_matches)
    fn = len(true_matches - predicted_matches)

    prec = precision_score(tp, fp)
    rec = recall_score(tp, fn)
    score = f_beta_score(prec, rec, beta=0.5)

    return {
        "precision": prec,
        "recall": rec,
        "f05": score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "true_count": true_len,
        "pred_count": pred_len,
    }


def macro_entity_f05(
    predicted: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """
    Compute macro-averaged entity-level F₀.₅ across all Source 1 entities.

    Parameters
    ----------
    predicted : dict[str, set[str]]
        Predicted match sets keyed by Source 1 entity ID.
    ground_truth : dict[str, set[str]]
        True match sets keyed by Source 1 entity ID.

    Returns
    -------
    dict with keys:
        macro_f05, macro_precision, macro_recall,
        n_entities, singleton_accuracy, per_entity
    """
    per_entity: List[Dict[str, Any]] = []
    all_s1 = sorted(set(ground_truth.keys()) | set(predicted.keys()))

    for s1_id in all_s1:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predicted.get(s1_id, set())

        score_dict = entity_f05(true_set, pred_set)
        score_dict["source1_id"] = s1_id
        per_entity.append(score_dict)

    n = len(per_entity)
    macro_f05 = sum(e["f05"] for e in per_entity) / n if n > 0 else 0.0
    macro_prec = sum(e["precision"] for e in per_entity) / n if n > 0 else 0.0
    macro_rec = sum(e["recall"] for e in per_entity) / n if n > 0 else 0.0

    singletons = [e for e in per_entity if e["true_count"] == 0]
    singleton_acc = (
        sum(1 for e in singletons if e["pred_count"] == 0) / len(singletons)
        if singletons else 1.0
    )

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "n_entities": n,
        "singleton_accuracy": singleton_acc,
        "per_entity": per_entity,
    }


# Backward-compatibility alias
entity_level_f05 = macro_entity_f05
