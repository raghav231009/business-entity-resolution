"""
validation.py — Output file consistency checks
================================================
Validates the two output files against the challenge rules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import Config
from .data_loader import load_source

logger = logging.getLogger(__name__)


def validate_outputs(cfg: Config) -> bool:
    """
    Run all validation checks on the output files.

    Returns True if all checks pass.
    """
    output_dir = cfg.paths.output_dir
    match_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"

    ok = True

    # ---- Load output files ----------------------------------------
    if not match_path.exists():
        logger.error("matching_results.tsv not found at %s", match_path)
        return False
    if not cand_path.exists():
        logger.error("candidate_pairs.tsv not found at %s", cand_path)
        return False

    match_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
    cand_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)

    # ---- Load test S1, S2, S3 IDs --------------------------------
    s1_df = load_source(cfg.paths.test_source1, "test_source1")
    s2_df = load_source(cfg.paths.test_source2, "test_source2")
    s3_df = load_source(cfg.paths.test_source3, "test_source3")

    s1_ids = set(s1_df["entity_id"])
    valid_cand_ids = set(s2_df["entity_id"]) | set(s3_df["entity_id"])

    # ---- Check 0: TSV formatting check -----------------------------
    with open(match_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" not in first_line:
            logger.error("matching_results.tsv is not properly tab-separated!")
            ok = False

    with open(cand_path, "r", encoding="utf-8") as f:
        first_line_cand = f.readline()
        if "\t" not in first_line_cand:
            logger.error("candidate_pairs.tsv is not properly tab-separated!")
            ok = False

    # ---- Check 1: correct column names ----------------------------
    if list(match_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        logger.error("matching_results.tsv has wrong columns: %s", list(match_df.columns))
        ok = False

    if list(cand_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        logger.error("candidate_pairs.tsv has wrong columns: %s", list(cand_df.columns))
        ok = False

    # ---- Check 2: every S1 ID appears exactly once ----------------
    match_s1 = set(match_df["source1_entity_id"])
    missing_match = s1_ids - match_s1
    extra_match = match_s1 - s1_ids
    if missing_match:
        logger.error("matching_results.tsv missing S1 IDs (%d missing): %s", len(missing_match), list(missing_match)[:10])
        ok = False
    if extra_match:
        logger.error("matching_results.tsv has extra S1 IDs (%d extra): %s", len(extra_match), list(extra_match)[:10])
        ok = False

    dupes = match_df[match_df.duplicated(subset=["source1_entity_id"])]
    if len(dupes) > 0:
        logger.error("matching_results.tsv has %d duplicate S1 rows: %s",
                      len(dupes), dupes["source1_entity_id"].tolist()[:10])
        ok = False

    # ---- Check 3: same for candidate_pairs.tsv --------------------
    cand_s1 = set(cand_df["source1_entity_id"])
    missing_cand = s1_ids - cand_s1
    extra_cand = cand_s1 - s1_ids
    if missing_cand:
        logger.error("candidate_pairs.tsv missing S1 IDs (%d missing): %s", len(missing_cand), list(missing_cand)[:10])
        ok = False
    if extra_cand:
        logger.error("candidate_pairs.tsv has extra S1 IDs (%d extra): %s", len(extra_cand), list(extra_cand)[:10])
        ok = False

    dupes_c = cand_df[cand_df.duplicated(subset=["source1_entity_id"])]
    if len(dupes_c) > 0:
        logger.error("candidate_pairs.tsv has %d duplicate S1 rows: %s",
                      len(dupes_c), dupes_c["source1_entity_id"].tolist()[:10])
        ok = False

    # ---- Check 4: matched IDs exist in S2/S3 & no dupes/self-matches --
    for _, row in match_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        if not raw:
            continue
        ids_list = [m.strip() for m in raw.split(",") if m.strip()]
        matched_ids = set(ids_list)

        # No self-matches
        if s1_id in matched_ids:
            logger.error("Self-match detected in matching_results for %s", s1_id)
            ok = False

        # All IDs valid in test pool
        invalid = matched_ids - valid_cand_ids
        if invalid:
            logger.error("Invalid matched IDs for %s: %s", s1_id, invalid)
            ok = False

        # No duplicate IDs within row
        if len(ids_list) != len(matched_ids):
            logger.error("Duplicate matched IDs within row for %s: %s", s1_id, ids_list)
            ok = False

    # ---- Check 5: candidate IDs exist in S2/S3 & no dupes/self-matches -
    cand_map: Dict[str, Set[str]] = {}
    for _, row in cand_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("candidate_entity_ids", "")
        if not raw:
            cand_map[s1_id] = set()
            continue
        cids_list = [c.strip() for c in raw.split(",") if c.strip()]
        cids_set = set(cids_list)
        cand_map[s1_id] = cids_set

        # No self-matches in candidate pairs
        if s1_id in cids_set:
            logger.error("Self-match detected in candidate_pairs for %s", s1_id)
            ok = False

        # All candidate IDs valid in test pool
        invalid_cands = cids_set - valid_cand_ids
        if invalid_cands:
            logger.error("Invalid candidate IDs for %s: %s", s1_id, invalid_cands)
            ok = False

        # No duplicate IDs within row
        if len(cids_list) != len(cids_set):
            logger.error("Duplicate candidate IDs within row for %s: %s", s1_id, cids_list)
            ok = False

    # ---- Check 6: matched ⊆ candidates (STRICT) ------------------
    subset_violations = 0
    for _, row in match_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        if not raw:
            continue
        matched_ids = {m.strip() for m in raw.split(",") if m.strip()}
        cands = cand_map.get(s1_id, set())
        outside = matched_ids - cands
        if outside:
            logger.error("CRITICAL: Matched IDs not in candidates for %s: %s", s1_id, outside)
            subset_violations += 1
            ok = False

    if subset_violations > 0:
        logger.error(
            "VIOLATION: %d Source-1 entities have predicted matches outside candidate_pairs.tsv!",
            subset_violations,
        )

    # ---- Summary ---------------------------------------------------
    if ok:
        logger.info("[OK] All output validation checks passed successfully.")
    else:
        logger.error("[FAIL] Validation failed - see errors above.")

    return ok
