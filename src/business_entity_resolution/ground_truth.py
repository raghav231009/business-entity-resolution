"""
ground_truth.py — Parse and query ground-truth labels
======================================================
Converts the raw ground-truth DataFrame into a clean mapping::

    { "S1-00001": {"S2-00047", "S3-00812"},
      "S1-00002": {"S3-00004"},
      "S1-00003": set(),  }

Handles:
* empty strings
* NaN values
* extra whitespace in IDs
"""

from __future__ import annotations

import logging
from typing import Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Parse the ground-truth DataFrame into a dict of sets.

    Parameters
    ----------
    gt_df : pd.DataFrame
        Must contain columns ``source1_entity_id`` and ``matched_entity_ids``.

    Returns
    -------
    dict[str, set[str]]
        Mapping from Source-1 entity ID to the set of matched IDs.
    """
    gt_map: Dict[str, Set[str]] = {}

    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw = str(row.get("matched_entity_ids", "")).strip()

        if raw == "" or raw.lower() == "nan":
            gt_map[s1_id] = set()
        else:
            ids = {m.strip() for m in raw.split(",") if m.strip()}
            gt_map[s1_id] = ids

    # Summary
    n_total = len(gt_map)
    n_singletons = sum(1 for v in gt_map.values() if len(v) == 0)
    n_with_match = n_total - n_singletons
    total_matches = sum(len(v) for v in gt_map.values())

    logger.info("Ground-truth parsed:")
    logger.info("  Total S1 entities    : %d", n_total)
    logger.info("  With ≥1 match        : %d", n_with_match)
    logger.info("  Singletons (0 match) : %d", n_singletons)
    logger.info("  Total match links    : %d", total_matches)

    return gt_map


def is_positive_pair(s1_id: str, candidate_id: str, gt_map: Dict[str, Set[str]]) -> bool:
    """Return True if *candidate_id* is a true match for *s1_id*."""
    return candidate_id in gt_map.get(s1_id, set())
