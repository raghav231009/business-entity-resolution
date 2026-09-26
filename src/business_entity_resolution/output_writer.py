"""
output_writer.py — Write matching_results.tsv & candidate_pairs.tsv
====================================================================
Produces the two required output files in exact challenge format.
"""

from __future__ import annotations

import logging
from typing import Dict, Set

import pandas as pd

from .config import Config
from .data_loader import SourceData

logger = logging.getLogger(__name__)


def write_outputs(
    results: Dict[str, Set[str]],
    candidates: pd.DataFrame,
    data: SourceData,
    cfg: Config,
) -> None:
    """
    Write both output TSV files.

    Parameters
    ----------
    results : dict[str, set[str]]
        Final match sets from ``postprocess()``.
    candidates : pd.DataFrame
        Full candidate DataFrame (before scoring).
    data : SourceData
        Test source DataFrames.
    cfg : Config
    """
    output_dir = cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_s1 = sorted(data.source1["entity_id"].unique())

    # ---- 1. Precompute candidate sets -----------------------------
    cand_sets: Dict[str, Set[str]] = {}
    for s1_id, grp in candidates.groupby("source1_id"):
        cand_sets[s1_id] = set(grp["candidate_id"])

    # ---- 2. matching_results.tsv ---------------------------------
    match_rows = []
    for s1_id in all_s1:
        matched = results.get(s1_id, set())
        cands = cand_sets.get(s1_id, set())
        # Strictly enforce constraint: predicted match MUST exist in candidate_pairs
        valid_matched = sorted(matched & cands)
        if len(valid_matched) < len(matched):
            logger.warning(
                "Filtered out %d matches for %s that were not in candidate set.",
                len(matched) - len(valid_matched), s1_id
            )
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(valid_matched) if valid_matched else "",
        })

    match_df = pd.DataFrame(match_rows)
    match_path = output_dir / "matching_results.tsv"
    match_df.to_csv(match_path, sep="\t", index=False)
    logger.info("Written %s  (%d rows)", match_path, len(match_df))

    # ---- 3. candidate_pairs.tsv ----------------------------------
    cand_rows = []
    for s1_id in all_s1:
        cids = sorted(cand_sets.get(s1_id, set()))
        cand_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cids) if cids else "",
        })

    cand_df = pd.DataFrame(cand_rows)
    cand_path = output_dir / "candidate_pairs.tsv"
    cand_df.to_csv(cand_path, sep="\t", index=False)
    logger.info("Written %s  (%d rows)", cand_path, len(cand_df))

    # ---- Consistency check: matched ⊆ candidates ------------------
    violations = 0
    for s1_id in all_s1:
        matched = results.get(s1_id, set())
        cands = cand_sets.get(s1_id, set())
        diff = matched - cands
        if diff:
            logger.error(
                "CONSISTENCY VIOLATION: %s has matched IDs not in candidates: %s",
                s1_id, diff,
            )
            violations += 1

    if violations > 0:
        logger.error("%d S1 entities have matched IDs outside their candidate set!", violations)
    else:
        logger.info("Consistency check passed: all matched IDs ⊆ candidate IDs.")


def append_chunk_outputs(
    results: Dict[str, Set[str]],
    cand_sets: Dict[str, Set[str]],
    s1_ids: List[str],
    cfg: Config,
    mode: str = "a",
) -> None:
    """
    Write or append rows for a chunk of S1 entities to matching_results.tsv and candidate_pairs.tsv.
    Guarantees strict 2-column format, tab separation, and predicted_matches ⊆ candidate_pairs.
    """
    output_dir = cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    match_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"

    write_header = (mode == "w" or not match_path.exists())

    with open(match_path, mode, encoding="utf-8") as fm, open(cand_path, mode, encoding="utf-8") as fc:
        if write_header:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in s1_ids:
            matched = results.get(s1_id, set())
            cands = cand_sets.get(s1_id, set())
            valid_matched = sorted(matched & cands)
            cids_sorted = sorted(cands)

            fm.write(f"{s1_id}\t{','.join(valid_matched)}\n")
            fc.write(f"{s1_id}\t{','.join(cids_sorted)}\n")

