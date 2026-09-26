"""
evaluate.py — Entity-level macro-averaged F₀.₅ evaluation
===========================================================
Implements the **exact** challenge metric:

1. For each Source-1 entity, compute precision, recall, F₀.₅.
2. Average F₀.₅ across all Source-1 entities (macro).

Singletons (no true matches) with empty prediction → F₀.₅ = 1.0.
Singletons with false prediction → F₀.₅ = 0.0.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


# ================================================================== #
#  Core metric
# ================================================================== #

# ================================================================== #
#  Core metric (imported from unified metrics module)
# ================================================================== #

from .metrics import (  # noqa: F401
    precision_score,
    recall_score,
    f_beta_score,
    f_beta,
    entity_f05,
    macro_entity_f05,
    entity_level_f05,
)



# ================================================================== #
#  Pipeline-level evaluation
# ================================================================== #

def evaluate(cfg: Config, ground_truth_map: Optional[Dict[str, Set[str]]] = None) -> Optional[Dict]:
    """
    Evaluate validation predictions saved during training against FULL ground truth.

    Reads ``models/val_predictions.csv`` and ``models/val_ground_truth.json``
    and applies the current threshold to produce macro F₀.₅.
    """
    val_path = cfg.paths.models_dir / "val_predictions.csv"
    if not val_path.exists():
        logger.warning("No validation predictions found at %s — skipping evaluation.", val_path)
        return None

    val_df = pd.read_csv(val_path)
    from .postprocessing import resolve_threshold
    threshold, threshold_source = resolve_threshold(cfg)
    logger.info("Evaluating with threshold %.2f (source: %s)", threshold, threshold_source)

    return evaluate_from_predictions(val_df, threshold, cfg, ground_truth_map=ground_truth_map)


def evaluate_from_predictions(
    val_df: pd.DataFrame,
    threshold: float,
    cfg: Config,
    ground_truth_map: Optional[Dict[str, Set[str]]] = None,
) -> Dict:
    """
    Given validation predictions and threshold, evaluate against the FULL ground truth.

    Parameters
    ----------
    val_df : pd.DataFrame
        Contains ``source1_id``, ``candidate_id``, ``probability``.
    threshold : float
    cfg : Config
    ground_truth_map : dict[str, set[str]], optional
        Full ground truth. If not provided, loads from ``models/val_ground_truth.json``.
    """
    # 1. Resolve full ground truth
    gt: Dict[str, Set[str]] = {}
    if ground_truth_map is not None:
        gt = {s1: set(ids) for s1, ids in ground_truth_map.items()}
    else:
        gt_file = cfg.paths.models_dir / "val_ground_truth.json"
        if gt_file.exists():
            try:
                with open(gt_file, "r", encoding="utf-8") as f:
                    raw_gt = json.load(f)
                gt = {s1: set(ids) for s1, ids in raw_gt.items()}
                logger.info("Loaded full validation ground truth from %s (%d entities)", gt_file, len(gt))
            except Exception as e:
                logger.warning("Failed to load %s: %s", gt_file, e)

    # Fallback only if no full ground truth exists anywhere
    if not gt:
        logger.warning(
            "CRITICAL: Full validation ground truth not found! Deriving ground truth from "
            "candidate pairs will overestimate recall if candidates missed true matches."
        )
        for s1_id, grp in val_df.groupby("source1_id"):
            positives = grp[grp["label"] == 1]
            gt[s1_id] = set(positives["candidate_id"])

    # 2. Build predicted match sets
    accepted = val_df[val_df["probability"] >= threshold]
    predicted: Dict[str, Set[str]] = {}
    for s1_id, grp in accepted.groupby("source1_id"):
        predicted[s1_id] = set(grp["candidate_id"])

    # Ensure all S1 entities in the full ground truth universe have an entry (even if empty)
    for s1_id in gt:
        if s1_id not in predicted:
            predicted[s1_id] = set()

    # 3. Evaluate macro F0.5
    metrics = entity_level_f05(predicted, gt)

    # 4. Pair-level metrics
    total_pred_pairs = sum(len(p) for p in predicted.values())
    total_true_pairs = sum(len(t) for t in gt.values())
    total_tp_pairs = sum(len(predicted.get(s1, set()) & gt.get(s1, set())) for s1 in gt)
    pair_precision = (total_tp_pairs / total_pred_pairs) if total_pred_pairs > 0 else 0.0
    pair_recall = (total_tp_pairs / total_true_pairs) if total_true_pairs > 0 else 0.0
    metrics["pair_precision"] = pair_precision
    metrics["pair_recall"] = pair_recall
    metrics["predicted_matches"] = total_pred_pairs
    metrics["predicted_singletons"] = sum(1 for p in predicted.values() if len(p) == 0)

    logger.info("==================================================")
    logger.info("VALIDATION EVALUATION @ threshold %.2f", threshold)
    logger.info("==================================================")
    logger.info("  Entity-level Macro F₀.₅ : %.4f", metrics["macro_f05"])
    logger.info("  Entity-level Macro Prec : %.4f", metrics["macro_precision"])
    logger.info("  Entity-level Macro Rec  : %.4f", metrics["macro_recall"])
    logger.info("  Pair Precision          : %.4f", pair_precision)
    logger.info("  Pair Recall             : %.4f", pair_recall)
    logger.info("  Validation Entities     : %d", metrics["n_entities"])
    logger.info("  Singleton Accuracy      : %.4f", metrics["singleton_accuracy"])
    logger.info("  Predicted Links         : %d", total_pred_pairs)
    logger.info("  Predicted Singletons    : %d", metrics["predicted_singletons"])
    logger.info("==================================================")

    # Save metrics
    metrics_dir = cfg.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in metrics.items() if k != "per_entity"}
    out["threshold"] = threshold
    metrics_path = metrics_dir / f"eval_t{threshold:.2f}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics
