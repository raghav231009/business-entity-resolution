#!/usr/bin/env python3
"""
utils/validate_submission.py — Official Challenge Submission Validator
======================================================================
Validates the format and constraints of matching_results.tsv and candidate_pairs.tsv
against test_source1.tsv, test_source2.tsv, and test_source3.tsv.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir data/test
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official Submission Validator for Amazon ML Entity Resolution Challenge"
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        help="Path to matching_results.tsv",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate_pairs.tsv",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv",
    )
    args = parser.parse_args()

    match_path = Path(args.matching)
    cand_path = Path(args.candidate)
    test_dir = Path(args.test_dir)

    print("=" * 60)
    print("OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 60)

    # 1. Check file existence
    if not match_path.exists():
        print(f"[FAIL] Matching results file not found: {match_path}")
        sys.exit(1)
    if not cand_path.exists():
        print(f"[FAIL] Candidate pairs file not found: {cand_path}")
        sys.exit(1)

    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    for f in (s1_file, s2_file, s3_file):
        if not f.exists():
            print(f"[FAIL] Required test file not found: {f}")
            sys.exit(1)

    print(f"Matching results : {match_path} ({match_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Candidate pairs  : {cand_path} ({cand_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Test directory   : {test_dir}")
    print("-" * 60)

    # 2. Load ground-truth test IDs
    print("Loading test entity pools...")
    s1_ids = set()
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        eid_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > eid_idx:
                s1_ids.add(parts[eid_idx])
    n_expected = len(s1_ids)
    print(f"  [OK] Test Source-1 entities : {n_expected:,}")

    valid_pool_ids = set()
    for s_file, name in ((s2_file, "Source-2"), (s3_file, "Source-3")):
        count = 0
        with open(s_file, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            eid_idx = header.index("entity_id") if "entity_id" in header else 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) > eid_idx:
                    valid_pool_ids.add(parts[eid_idx])
                    count += 1
        print(f"  [OK] Test {name} entities    : {count:,}")
    print(f"  [OK] Total target pool IDs   : {len(valid_pool_ids):,}")
    print("-" * 60)

    errors = []

    # 3. Validate candidate_pairs.tsv
    print("Validating candidate_pairs.tsv...")
    cand_s1_seen = set()
    cand_map = {}
    invalid_cands = 0
    cand_rows = 0

    with open(cand_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" not in first_line:
            errors.append("candidate_pairs.tsv is not tab-separated")
        parts = first_line.rstrip("\r\n").split("\t")
        if parts != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"candidate_pairs.tsv invalid header: {parts}")

        for line_num, line in enumerate(f, start=2):
            cand_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in cand_s1_seen:
                errors.append(f"Duplicate S1 ID in candidate_pairs.tsv at line {line_num}: {s1_id}")
            cand_s1_seen.add(s1_id)

            raw = parts[1] if len(parts) > 1 else ""
            if not raw:
                cand_map[s1_id] = set()
                continue

            c_list = [c.strip() for c in raw.split(",") if c.strip()]
            c_set = set(c_list)
            if len(c_list) != len(c_set):
                errors.append(f"Duplicate candidate IDs within row for {s1_id}")
            if s1_id in c_set:
                errors.append(f"Self-match in candidate_pairs.tsv for {s1_id}")

            invalid = c_set - valid_pool_ids
            if invalid:
                invalid_cands += len(invalid)

            cand_map[s1_id] = c_set

    if cand_rows != n_expected:
        errors.append(f"candidate_pairs.tsv row count {cand_rows:,} != expected {n_expected:,}")
    missing_cand_s1 = s1_ids - cand_s1_seen
    if missing_cand_s1:
        errors.append(f"candidate_pairs.tsv missing {len(missing_cand_s1)} test S1 entities")
    extra_cand_s1 = cand_s1_seen - s1_ids
    if extra_cand_s1:
        errors.append(f"candidate_pairs.tsv has {len(extra_cand_s1)} unknown S1 entities")
    if invalid_cands > 0:
        errors.append(f"candidate_pairs.tsv contains {invalid_cands} candidate IDs outside test pool")

    print(f"  [OK] candidate_pairs.tsv rows : {cand_rows:,}")
    print(f"  [OK] candidate_pairs.tsv unique S1 coverage: {len(cand_s1_seen):,}/{n_expected:,}")

    # 4. Validate matching_results.tsv
    print("\nValidating matching_results.tsv...")
    match_s1_seen = set()
    subset_violations = 0
    invalid_matches = 0
    self_matches = 0
    dup_matches = 0
    total_pred_links = 0
    singletons = 0
    match_rows = 0

    with open(match_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" not in first_line:
            errors.append("matching_results.tsv is not tab-separated")
        parts = first_line.rstrip("\r\n").split("\t")
        if parts != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"matching_results.tsv invalid header: {parts}")

        for line_num, line in enumerate(f, start=2):
            match_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in match_s1_seen:
                errors.append(f"Duplicate S1 ID in matching_results.tsv at line {line_num}: {s1_id}")
            match_s1_seen.add(s1_id)

            raw = parts[1] if len(parts) > 1 else ""
            if not raw:
                singletons += 1
                continue

            m_list = [m.strip() for m in raw.split(",") if m.strip()]
            m_set = set(m_list)
            total_pred_links += len(m_set)

            if len(m_list) != len(m_set):
                dup_matches += 1
            if s1_id in m_set:
                self_matches += 1

            invalid = m_set - valid_pool_ids
            if invalid:
                invalid_matches += len(invalid)

            # Strict subset constraint: matched_entity_ids ⊆ candidate_entity_ids
            c_set = cand_map.get(s1_id, set())
            outside = m_set - c_set
            if outside:
                subset_violations += 1

    if match_rows != n_expected:
        errors.append(f"matching_results.tsv row count {match_rows:,} != expected {n_expected:,}")
    missing_match_s1 = s1_ids - match_s1_seen
    if missing_match_s1:
        errors.append(f"matching_results.tsv missing {len(missing_match_s1)} test S1 entities")
    extra_match_s1 = match_s1_seen - s1_ids
    if extra_match_s1:
        errors.append(f"matching_results.tsv has {len(extra_match_s1)} unknown S1 entities")
    if subset_violations > 0:
        errors.append(f"CRITICAL CONSTRAINT: {subset_violations} S1 entities have matches not in candidates")
    if invalid_matches > 0:
        errors.append(f"matching_results.tsv contains {invalid_matches} matched IDs outside test pool")
    if self_matches > 0:
        errors.append(f"matching_results.tsv contains {self_matches} self-matches")
    if dup_matches > 0:
        errors.append(f"matching_results.tsv contains {dup_matches} rows with duplicate matched IDs")

    print(f"  [OK] matching_results.tsv rows : {match_rows:,}")
    print(f"  [OK] matching_results.tsv unique S1 coverage: {len(match_s1_seen):,}/{n_expected:,}")
    print(f"  [OK] Predicted links           : {total_pred_links:,}")
    print(f"  [OK] Predicted singletons      : {singletons:,} ({singletons / max(n_expected, 1) * 100:.2f}%)")
    print(f"  [OK] Strict subset check       : matches <= candidates for 100% of entities")

    print("=" * 60)
    if errors:
        print(f"[FAIL] Validation failed with {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.")
        sys.exit(1)
    else:
        print("[SUCCESS] ALL SUBMISSION VALIDATION CHECKS PASSED.")
        print("Both matching_results.tsv and candidate_pairs.tsv are 100% compliant.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
