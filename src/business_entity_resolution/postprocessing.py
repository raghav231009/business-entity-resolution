"""
postprocessing.py — Filter predictions & singleton protection
==============================================================
Applies the selected threshold and optional safety guards
(min name similarity, min address similarity, country match)
to produce the final match sets.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from .config import Config
from .data_loader import SourceData

logger = logging.getLogger(__name__)


def resolve_threshold(cfg: Config, explicit_threshold: Optional[float] = None) -> Tuple[float, str]:
    """
    Resolve prediction threshold following the required fallback priority:
    1. explicitly supplied threshold
    2. persisted tuned threshold (models/selected_threshold.json)
    3. configured selected threshold (config.yaml)
    4. configured default threshold (config.yaml)
    """
    if explicit_threshold is not None:
        return float(explicit_threshold), "explicit parameter"

    sel_path = cfg.paths.models_dir / "selected_threshold.json"
    if sel_path.exists():
        try:
            with open(sel_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = float(data.get("threshold", cfg.threshold.default))
            f05 = data.get("validation_f0_5")
            source_desc = f"models/selected_threshold.json (validation macro F0.5={f05:.4f})" if f05 is not None else "models/selected_threshold.json"
            return t, source_desc
        except Exception as e:
            logger.warning("Could not read %s: %s", sel_path, e)

    if cfg.threshold.selected is not None:
        return float(cfg.threshold.selected), "config.yaml (threshold.selected)"

    return float(cfg.threshold.default), "config.yaml (threshold.default)"


def postprocess(
    scored_df: pd.DataFrame,
    data: SourceData,
    cfg: Config,
    threshold: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """
    Apply threshold and safety guards, returning a dict of match sets.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Must contain ``source1_id``, ``candidate_id``, ``probability``.
    data : SourceData
        Test source DataFrames (used to get all S1 IDs).
    cfg : Config
    threshold : float, optional
        Override threshold if explicitly passed.

    Returns
    -------
    dict[str, set[str]]
        Mapping from Source-1 entity ID to matched entity IDs.
    """
    pp = cfg.postprocessing
    selected_threshold, source_name = resolve_threshold(cfg, explicit_threshold=threshold)

    logger.info("Using prediction threshold: %.2f", selected_threshold)
    logger.info("Threshold source: %s", source_name)

    # Start with probability threshold
    mask = scored_df["probability"] >= selected_threshold

    # Optional: minimum name similarity guard
    if pp.min_name_similarity > 0 and "name_fuzzy_ratio" in scored_df.columns:
        mask = mask & (scored_df["name_fuzzy_ratio"] >= pp.min_name_similarity)

    # Optional: minimum address similarity guard
    if pp.min_address_similarity > 0 and "address_fuzzy_ratio" in scored_df.columns:
        mask = mask & (scored_df["address_fuzzy_ratio"] >= pp.min_address_similarity)

    # Optional: require country match
    if pp.require_country_match and "country_exact" in scored_df.columns:
        mask = mask & (scored_df["country_exact"] == 1)

    accepted = scored_df[mask]

    # Build match sets
    results: Dict[str, Set[str]] = {}
    for s1_id, grp in accepted.groupby("source1_id"):
        results[s1_id] = set(grp["candidate_id"])

    # Ensure every test S1 entity has an entry (even if empty)
    all_s1 = set(data.source1["entity_id"])
    for s1_id in all_s1:
        if s1_id not in results:
            results[s1_id] = set()

    n_matched = sum(1 for v in results.values() if len(v) > 0)
    n_singleton = sum(1 for v in results.values() if len(v) == 0)
    total_links = sum(len(v) for v in results.values())

    logger.info("Postprocessing results:")
    logger.info("  S1 entities with matches : %d", n_matched)
    logger.info("  S1 singletons            : %d", n_singleton)
    logger.info("  Total match links        : %d", total_links)

    return results
