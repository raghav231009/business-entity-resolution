"""
dataset_builder.py — Labelled train/validation dataset creation
================================================================
Splits at the **Source-1 entity level** to prevent leakage,
labels candidate pairs using ground truth, and optionally
down-samples negatives.
"""

from __future__ import annotations

import logging
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .ground_truth import is_positive_pair

logger = logging.getLogger(__name__)


def _is_hard_negative(df: pd.DataFrame) -> pd.Series:
    """Identify challenging negative candidate pairs based on similarity signals."""
    if len(df) == 0:
        return pd.Series(dtype=bool)

    cond = pd.Series(False, index=df.index)
    if "name_fuzzy_ratio" in df.columns:
        cond |= (df["name_fuzzy_ratio"] >= 65.0)
    if "name_token_sort_ratio" in df.columns:
        cond |= (df["name_token_sort_ratio"] >= 65.0)
    if "name_jaccard" in df.columns:
        cond |= (df["name_jaccard"] >= 0.4)
    if "generated_by" in df.columns:
        # Multiple blocking rules triggered candidate
        cond |= df["generated_by"].astype(str).str.contains(",")
    return cond


def build_training_dataset(
    feat_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Create labelled training and validation sets partitioned strictly at the
    Source-1 entity level.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix. Must contain ``source1_id`` and ``candidate_id``.
    gt_map : dict[str, set[str]]
        Full ground-truth mapping.
    cfg : Config

    Returns
    -------
    (train_df, val_df, val_gt_map) : tuple of (DataFrame, DataFrame, dict)
        val_gt_map contains the FULL original ground-truth mapping for all
        validation Source-1 entities.
    """
    rng = np.random.RandomState(cfg.random_seed)

    # ---- Label every candidate pair --------------------------------
    feat_df = feat_df.copy()
    feat_df["label"] = feat_df.apply(
        lambda r: int(is_positive_pair(r["source1_id"], r["candidate_id"], gt_map)),
        axis=1,
    )

    # ---- Hard vs easy negative analysis ----------------------------
    neg_mask = feat_df["label"] == 0
    pos_mask = feat_df["label"] == 1
    hard_neg_mask = neg_mask & _is_hard_negative(feat_df)
    easy_neg_mask = neg_mask & (~hard_neg_mask)

    feat_df["is_hard_negative"] = hard_neg_mask.astype(int)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    n_hard_neg = int(hard_neg_mask.sum())
    n_easy_neg = int(easy_neg_mask.sum())
    ratio_str = f"1:{n_neg / max(n_pos, 1):.1f}" if n_pos > 0 else "0:0"

    logger.info("==================================================")
    logger.info("TRAINING DATASET COMPOSITION")
    logger.info("==================================================")
    logger.info("  Positive pairs        : %d", n_pos)
    logger.info("  Negative pairs (total): %d", n_neg)
    logger.info("  Hard negatives        : %d", n_hard_neg)
    logger.info("  Easy negatives        : %d", n_easy_neg)
    logger.info("  Positive/Negative ratio: %s", ratio_str)
    logger.info("==================================================")

    # ---- Split at Source-1 entity level using FULL ground-truth universe ---
    all_s1_universe = sorted(gt_map.keys()) if gt_map else sorted(feat_df["source1_id"].unique())
    rng.shuffle(all_s1_universe)

    val_size = int(len(all_s1_universe) * cfg.training.validation_split)
    max_val_s1 = getattr(cfg.training, "max_validation_s1_entities", None)
    if max_val_s1 is not None and val_size > max_val_s1:
        logger.info("Dev mode: Bounding validation S1 entities to %d (out of %d)", max_val_s1, val_size)
        val_size = max_val_s1

    val_s1 = set(all_s1_universe[:val_size])
    train_s1 = set(all_s1_universe[val_size:])

    # Complete ground-truth map for validation S1 entities
    val_gt_map: Dict[str, Set[str]] = {s1: set(gt_map.get(s1, set())) for s1 in val_s1}

    train_df = feat_df[feat_df["source1_id"].isin(train_s1)].copy()
    val_df = feat_df[feat_df["source1_id"].isin(val_s1)].copy()

    logger.info("Train / Val split (Source-1 entity level):")
    logger.info("  Train S1 entities     : %d  → %d pairs (%d pos, %d neg)",
                len(train_s1), len(train_df),
                int(train_df["label"].sum()), int((train_df["label"] == 0).sum()))
    logger.info("  Val   S1 entities     : %d  → %d pairs (%d pos, %d neg)",
                len(val_s1), len(val_df),
                int(val_df["label"].sum()), int((val_df["label"] == 0).sum()))

    # ---- Optional negative down-sampling (preserving hard negatives) ---
    max_neg = cfg.training.max_negatives_per_positive
    if max_neg and max_neg > 0:
        train_df = _downsample_negatives(train_df, max_neg, rng)

    if cfg.training.shuffle:
        train_df = train_df.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)

    return train_df, val_df, val_gt_map


def _downsample_negatives(
    df: pd.DataFrame,
    max_neg_per_pos: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    For each Source-1 entity, keep at most ``max_neg_per_pos`` negatives
    per positive example. Positives and hard negatives are prioritised.
    """
    parts = []
    for s1_id, grp in df.groupby("source1_id"):
        pos = grp[grp["label"] == 1]
        neg = grp[grp["label"] == 0]
        max_neg = max(max_neg_per_pos * max(len(pos), 1), 1)

        if len(neg) > max_neg:
            # Prioritize hard negatives
            hard_neg = neg[neg["is_hard_negative"] == 1]
            easy_neg = neg[neg["is_hard_negative"] == 0]

            if len(hard_neg) >= max_neg:
                selected_neg = hard_neg.sample(n=max_neg, random_state=rng)
            else:
                needed = max_neg - len(hard_neg)
                sampled_easy = easy_neg.sample(n=min(needed, len(easy_neg)), random_state=rng)
                selected_neg = pd.concat([hard_neg, sampled_easy])
            parts.append(pd.concat([pos, selected_neg]))
        else:
            parts.append(pd.concat([pos, neg]))

    result = pd.concat(parts, ignore_index=True)
    logger.info("  After negative sampling: %d pairs (%d pos, %d neg)",
                len(result), int(result["label"].sum()),
                int((result["label"] == 0).sum()))
    return result


def prefilter_candidates_for_training(
    candidates: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    max_neg_per_s1: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Subsample candidate pairs before feature building.
    Retains ALL positive pairs (100% recall), and prioritises multi-rule
    hard negatives up to ``max_neg_per_s1`` per Source-1 entity.
    """
    if len(candidates) == 0 or max_neg_per_s1 <= 0:
        return candidates

    s1_col = candidates["source1_id"].values
    cand_col = candidates["candidate_id"].values
    is_pos = np.array([
        c in gt_map.get(s, set())
        for s, c in zip(s1_col, cand_col)
    ], dtype=bool)

    candidates = candidates.assign(_is_pos=is_pos)
    if "generated_by" in candidates.columns:
        rule_counts = candidates["generated_by"].astype(str).str.count(",").values + 1
    else:
        rule_counts = np.ones(len(candidates), dtype=int)
    candidates = candidates.assign(_rule_count=rule_counts)

    parts = []
    for s1_id, grp in candidates.groupby("source1_id"):
        pos = grp[grp["_is_pos"]]
        neg = grp[~grp["_is_pos"]]

        if len(neg) > max_neg_per_s1:
            neg_sorted = neg.sort_values(by="_rule_count", ascending=False)
            chosen_neg = neg_sorted.iloc[:max_neg_per_s1]
            parts.append(pos)
            parts.append(chosen_neg)
        else:
            parts.append(grp)

    filtered = pd.concat(parts, ignore_index=True).drop(columns=["_is_pos", "_rule_count"])
    logger.info("Prefiltered candidates for training: %d pairs (was %d pairs, kept %d positives)",
                len(filtered), len(candidates), int(is_pos.sum()))
    return filtered
