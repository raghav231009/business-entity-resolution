"""
threshold_tuning.py — Sweep thresholds to maximise macro F₀.₅
===============================================================
Reads the saved validation predictions, evaluates every candidate
threshold, and selects the one with the best macro F₀.₅.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from .config import Config
from .evaluate import evaluate_from_predictions

logger = logging.getLogger(__name__)


def tune_threshold(cfg: Config, ground_truth_map: Optional[Dict[str, Set[str]]] = None) -> float:
    """
    Evaluate each threshold in ``cfg.threshold.search_range``
    on validation predictions against the FULL ground truth and return the best one.

    Persists the chosen threshold to ``models/selected_threshold.json``
    and saves the full sweep to ``artifacts/metrics/threshold_sweep.json``.
    """
    val_path = cfg.paths.models_dir / "val_predictions.csv"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation predictions not found at {val_path}. "
            "Run training first."
        )

    val_df = pd.read_csv(val_path)
    thresholds = cfg.threshold.search_range

    logger.info("Tuning threshold over %d values using full ground truth …", len(thresholds))

    results: List[Dict] = []
    best_f05 = -1.0
    best_threshold = float(cfg.threshold.default)

    for t in thresholds:
        metrics = evaluate_from_predictions(val_df, t, cfg, ground_truth_map=ground_truth_map)
        row = {
            "threshold": float(t),
            "macro_f05": float(metrics["macro_f05"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "n_entities": int(metrics["n_entities"]),
            "singleton_accuracy": float(metrics["singleton_accuracy"]),
            "predicted_matches": int(metrics["predicted_matches"]),
            "predicted_singletons": int(metrics["predicted_singletons"]),
        }
        results.append(row)

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = float(t)

    # Summary table
    logger.info("%-10s  %-10s  %-10s  %-10s  %-10s  %-10s",
                "Threshold", "F0.5", "Precision", "Recall", "Matches", "Singletons")
    for r in results:
        logger.info(
            "%-10.2f  %-10.4f  %-10.4f  %-10.4f  %-10d  %-10d",
            r["threshold"], r["macro_f05"],
            r["macro_precision"], r["macro_recall"],
            r["predicted_matches"], r["predicted_singletons"],
        )

    logger.info("Best threshold: %.2f  →  macro F₀.₅ = %.4f", best_threshold, best_f05)

    # Save selected threshold to models/selected_threshold.json
    models_dir = cfg.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    selected_path = models_dir / "selected_threshold.json"
    selected_data = {
        "threshold": best_threshold,
        "metric": "macro_f0.5",
        "validation_f0_5": best_f05,
    }
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected_data, f, indent=2)
    logger.info("Selected threshold persisted to %s", selected_path)

    # Save full sweep
    metrics_dir = cfg.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = metrics_dir / "threshold_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_threshold,
            "best_f05": best_f05,
            "results": results,
        }, f, indent=2)
    logger.info("Threshold sweep saved to %s", sweep_path)

    return best_threshold
