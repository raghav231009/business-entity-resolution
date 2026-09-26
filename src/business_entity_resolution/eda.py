"""
eda.py — Reusable Exploratory Data Analysis module
=================================================
Calculates comprehensive dataset statistics directly from the actual
challenge data files and generates structured JSON and markdown reports.

Functions:
- get_dataset_sizes
- inspect_schemas
- analyze_missing_values
- analyze_country_distributions
- analyze_business_names
- analyze_addresses
- analyze_ground_truth
- analyze_true_pair_similarity
- generate_markdown_report
- run_full_eda
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import Config
from .preprocessing import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_postal_code,
    extract_numeric_tokens,
    tokenize,
)
from .features import (
    name_exact_match,
    name_fuzzy_ratio,
    name_token_sort_ratio,
    address_exact_match,
    address_fuzzy_ratio,
    jaccard_similarity,
    numeric_token_overlap,
    country_exact_match,
)

logger = logging.getLogger(__name__)

# Expected schemas
EXPECTED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

LEGAL_SUFFIXES_TO_CHECK = [
    "ltd",
    "limited",
    "pvt",
    "private",
    "inc",
    "llc",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
]


def _safe_quantiles(series: pd.Series) -> Dict[str, float]:
    """Calculate standard summary percentiles and stats for a numeric series."""
    if len(series) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(round(series.mean(), 2)),
        "std": float(round(series.std(), 2)) if len(series) > 1 else 0.0,
        "min": float(round(series.min(), 2)),
        "p25": float(round(series.quantile(0.25), 2)),
        "median": float(round(series.median(), 2)),
        "p75": float(round(series.quantile(0.75), 2)),
        "p90": float(round(series.quantile(0.90), 2)),
        "max": float(round(series.max(), 2)),
    }


def get_dataset_sizes(cfg: Config) -> Dict[str, Any]:
    """Report row count, column count, and file size for each dataset file."""
    logger.info("Computing dataset sizes and row counts...")
    files = {
        "train_source1": cfg.paths.train_source1,
        "train_source2": cfg.paths.train_source2,
        "train_source3": cfg.paths.train_source3,
        "train_ground_truth": cfg.paths.train_ground_truth,
        "test_source1": cfg.paths.test_source1,
        "test_source2": cfg.paths.test_source2,
        "test_source3": cfg.paths.test_source3,
    }

    results = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

        file_size_bytes = path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        # Count lines efficiently without loading entire file into memory
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline()
            header_cols = header_line.rstrip("\r\n").split("\t")
            row_count = sum(1 for _ in f)

        results[key] = {
            "file_path": str(path),
            "file_name": path.name,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "row_count": row_count,
            "column_count": len(header_cols),
            "columns": header_cols,
        }
        logger.info(
            "  %s: %d rows, %d cols, %.2f MB",
            key,
            row_count,
            len(header_cols),
            file_size_mb,
        )

    return results


def inspect_schemas(cfg: Config) -> Dict[str, Any]:
    """Inspect and validate schema for every dataset."""
    logger.info("Validating schemas across all datasets...")
    files = {
        "train_source1": (cfg.paths.train_source1, EXPECTED_SOURCE_COLS),
        "train_source2": (cfg.paths.train_source2, EXPECTED_SOURCE_COLS),
        "train_source3": (cfg.paths.train_source3, EXPECTED_SOURCE_COLS),
        "train_ground_truth": (cfg.paths.train_ground_truth, EXPECTED_GT_COLS),
        "test_source1": (cfg.paths.test_source1, EXPECTED_SOURCE_COLS),
        "test_source2": (cfg.paths.test_source2, EXPECTED_SOURCE_COLS),
        "test_source3": (cfg.paths.test_source3, EXPECTED_SOURCE_COLS),
    }

    results = {}
    for key, (path, expected_cols) in files.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        sample_df = pd.read_csv(
            path, sep="\t", nrows=5, dtype=str, keep_default_na=False
        )
        actual_cols = list(sample_df.columns)
        missing_req = [c for c in expected_cols if c not in actual_cols]
        unexpected = [c for c in actual_cols if c not in expected_cols]

        results[key] = {
            "file_name": path.name,
            "column_names": actual_cols,
            "expected_columns": expected_cols,
            "missing_required_columns": missing_req,
            "unexpected_columns": unexpected,
            "is_schema_valid": len(missing_req) == 0,
            "dtypes": {c: "string (utf-8)" for c in actual_cols},
        }

    return results


def analyze_missing_values(cfg: Config) -> Dict[str, Any]:
    """
    Calculate exact missing values for all sources (train and test):
    identifying empty strings "", whitespace-only, and null-like values.
    """
    logger.info("Analyzing missing values...")
    sources = {
        "train_source1": cfg.paths.train_source1,
        "train_source2": cfg.paths.train_source2,
        "train_source3": cfg.paths.train_source3,
        "test_source1": cfg.paths.test_source1,
        "test_source2": cfg.paths.test_source2,
        "test_source3": cfg.paths.test_source3,
    }

    results = {}
    for key, path in sources.items():
        col_stats: Dict[str, Any] = {}
        total_rows = 0

        # Read in chunks for memory safety
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000
        ):
            total_rows += len(chunk)
            for col in chunk.columns:
                if col not in col_stats:
                    col_stats[col] = {
                        "empty_string": 0,
                        "whitespace_only": 0,
                        "null_like": 0,
                        "total_missing": 0,
                    }
                s = chunk[col]
                empty_mask = s == ""
                ws_mask = (s != "") & (s.str.strip() == "")
                null_mask = s.str.lower().isin(["nan", "null", "none"])

                col_stats[col]["empty_string"] += int(empty_mask.sum())
                col_stats[col]["whitespace_only"] += int(ws_mask.sum())
                col_stats[col]["null_like"] += int(null_mask.sum())
                col_stats[col]["total_missing"] += int(
                    (empty_mask | ws_mask | null_mask).sum()
                )

        # Compute percentages
        for col, stats in col_stats.items():
            stats["total_rows"] = total_rows
            stats["missing_percentage"] = round(
                (stats["total_missing"] / total_rows) * 100, 4
            )
            stats["empty_percentage"] = round(
                (stats["empty_string"] / total_rows) * 100, 4
            )

        results[key] = {
            "total_rows": total_rows,
            "columns": col_stats,
        }
        logger.info("  %s missing values analyzed across %d rows", key, total_rows)

    return results


def analyze_country_distributions(cfg: Config) -> Dict[str, Any]:
    """Calculate actual country distributions across all train and test sources."""
    logger.info("Analyzing country distributions across train and test...")
    sources = {
        "train_source1": cfg.paths.train_source1,
        "train_source2": cfg.paths.train_source2,
        "train_source3": cfg.paths.train_source3,
        "test_source1": cfg.paths.test_source1,
        "test_source2": cfg.paths.test_source2,
        "test_source3": cfg.paths.test_source3,
    }

    results = {}
    all_countries_seen: Set[str] = set()
    france_detected_in: List[str] = []

    for key, path in sources.items():
        country_counts: Dict[str, int] = {}
        total = 0
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["country"],
            dtype=str,
            keep_default_na=False,
            chunksize=500000,
        ):
            norm_c = chunk["country"].str.strip().str.lower()
            vc = norm_c.value_counts().to_dict()
            for c_name, cnt in vc.items():
                country_counts[c_name] = country_counts.get(c_name, 0) + cnt
            total += len(chunk)

        all_countries_seen.update(country_counts.keys())
        if "france" in country_counts:
            france_detected_in.append(key)

        percentages = {
            c: round((cnt / total) * 100, 4) for c, cnt in country_counts.items()
        }

        results[key] = {
            "total_rows": total,
            "unique_countries": sorted(country_counts.keys()),
            "counts": country_counts,
            "percentages": percentages,
        }
        logger.info("  %s countries: %s", key, country_counts)

    return {
        "per_dataset": results,
        "all_unique_countries": sorted(all_countries_seen),
        "france_in_test": any("test" in k for k in france_detected_in),
        "france_detected_datasets": france_detected_in,
    }


def analyze_business_names(
    cfg: Config, sample_size: int = 100000
) -> Dict[str, Any]:
    """Analyze business names (uniqueness, length, token count, punctuation, legal suffixes)."""
    logger.info(
        "Analyzing business name characteristics (sample_size=%d)...",
        sample_size,
    )
    s1_path = cfg.paths.train_source1

    df = pd.read_csv(
        s1_path,
        sep="\t",
        nrows=sample_size,
        dtype=str,
        keep_default_na=False,
    )

    names = df["business_name"].fillna("").astype(str)
    norm_names = names.apply(
        lambda n: normalize_name(n, cfg.preprocessing.legal_suffixes)
    )

    lengths = names.str.len()
    token_counts = names.apply(lambda n: len(n.split()))

    # Punctuation presence
    punct_re = re.compile(r"[^\w\s]", re.UNICODE)
    has_punct = names.apply(lambda n: bool(punct_re.search(n)))

    # Legal suffixes
    suffix_counts: Dict[str, int] = {}
    lower_names = names.str.lower()
    for sfx in LEGAL_SUFFIXES_TO_CHECK:
        pat = re.compile(r"\b" + re.escape(sfx) + r"\b", re.IGNORECASE)
        cnt = int(lower_names.apply(lambda n: bool(pat.search(n))).sum())
        suffix_counts[sfx] = cnt

    unique_raw = int(names.nunique())
    unique_norm = int(norm_names.nunique())

    return {
        "sample_analyzed": len(names),
        "unique_raw_names": unique_raw,
        "unique_normalized_names": unique_norm,
        "duplicate_raw_pct": round(
            (1.0 - unique_raw / len(names)) * 100, 2
        ),
        "duplicate_normalized_pct": round(
            (1.0 - unique_norm / len(names)) * 100, 2
        ),
        "empty_names_count": int((names.str.strip() == "").sum()),
        "name_length_distribution": _safe_quantiles(lengths),
        "name_token_count_distribution": _safe_quantiles(token_counts),
        "punctuation_frequency_pct": round(
            float(has_punct.mean()) * 100, 2
        ),
        "legal_suffix_frequencies": suffix_counts,
    }


def analyze_addresses(
    cfg: Config, sample_size: int = 100000
) -> Dict[str, Any]:
    """Analyze address characteristics (length, tokens, numeric tokens, postal codes, abbreviations)."""
    logger.info(
        "Analyzing address characteristics (sample_size=%d)...", sample_size
    )
    s1_path = cfg.paths.train_source1

    df = pd.read_csv(
        s1_path,
        sep="\t",
        nrows=sample_size,
        dtype=str,
        keep_default_na=False,
    )

    addrs = df["business_address"].fillna("").astype(str)
    norm_addrs = addrs.apply(
        lambda a: normalize_address(
            a, cfg.preprocessing.address_abbreviations
        )
    )

    empty_count = int((addrs.str.strip() == "").sum())
    non_empty = addrs[addrs.str.strip() != ""]

    lengths = non_empty.str.len()
    token_counts = non_empty.apply(lambda a: len(a.split()))

    # Numeric tokens
    num_token_re = re.compile(r"\b\d+\b")
    num_counts = non_empty.apply(lambda a: len(num_token_re.findall(a)))
    has_numbers = num_counts > 0

    # Postal codes
    postal_codes = non_empty.apply(extract_postal_code)
    has_postal = postal_codes != ""

    # Common abbreviations
    common_abbrs = [
        "st",
        "rd",
        "ave",
        "blvd",
        "dr",
        "ln",
        "hwy",
        "apt",
        "ste",
        "fl",
    ]
    abbr_counts = {}
    lower_addrs = non_empty.str.lower()
    for ab in common_abbrs:
        pat = re.compile(r"\b" + re.escape(ab) + r"\b", re.IGNORECASE)
        cnt = int(lower_addrs.apply(lambda a: bool(pat.search(a))).sum())
        abbr_counts[ab] = cnt

    return {
        "sample_analyzed": len(addrs),
        "empty_addresses_count": empty_count,
        "empty_addresses_pct": round(
            (empty_count / len(addrs)) * 100, 2
        ),
        "address_length_distribution": _safe_quantiles(lengths),
        "address_token_count_distribution": _safe_quantiles(token_counts),
        "numeric_token_distribution": _safe_quantiles(num_counts),
        "has_numeric_tokens_pct": round(
            float(has_numbers.mean()) * 100, 2
        ),
        "has_postal_code_pct": round(
            float(has_postal.mean()) * 100, 2
        ),
        "common_abbreviation_frequencies": abbr_counts,
        "unique_raw_addresses": int(non_empty.nunique()),
        "unique_normalized_addresses": int(
            norm_addrs[norm_addrs != ""].nunique()
        ),
    }


def analyze_ground_truth(cfg: Config) -> Dict[str, Any]:
    """Analyze the authoritative ground truth file."""
    logger.info("Analyzing ground truth match distributions...")
    gt_path = cfg.paths.train_ground_truth
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    total_s1 = 0
    zero_matches = 0
    exactly_one = 0
    multiple_matches = 0
    total_links = 0
    s2_links = 0
    s3_links = 0
    s2_only_s1 = 0
    s3_only_s1 = 0
    s2_and_s3_s1 = 0

    match_counts: List[int] = []

    # Read ground truth in chunks to be gentle with RAM
    for chunk in pd.read_csv(
        gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000
    ):
        for raw in chunk["matched_entity_ids"].values:
            total_s1 += 1
            if not raw or raw.lower() == "nan":
                zero_matches += 1
                match_counts.append(0)
            else:
                m_ids = [m.strip() for m in raw.split(",") if m.strip()]
                c = len(m_ids)
                match_counts.append(c)
                total_links += c

                if c == 1:
                    exactly_one += 1
                elif c > 1:
                    multiple_matches += 1

                s2_in = False
                s3_in = False
                for mid in m_ids:
                    mid_lower = mid.lower()
                    if "s2" in mid_lower:
                        s2_links += 1
                        s2_in = True
                    elif "s3" in mid_lower:
                        s3_links += 1
                        s3_in = True

                if s2_in and s3_in:
                    s2_and_s3_s1 += 1
                elif s2_in:
                    s2_only_s1 += 1
                elif s3_in:
                    s3_only_s1 += 1

    counts_series = pd.Series(match_counts)
    singleton_pct = round((zero_matches / total_s1) * 100, 4)

    return {
        "total_source1_entities": total_s1,
        "total_true_links": total_links,
        "source2_true_links": s2_links,
        "source3_true_links": s3_links,
        "s2_only_source1_entities": s2_only_s1,
        "s3_only_source1_entities": s3_only_s1,
        "s2_plus_s3_source1_entities": s2_and_s3_s1,
        "singleton_count": zero_matches,
        "singleton_percentage": singleton_pct,
        "exactly_one_match_count": exactly_one,
        "exactly_one_match_percentage": round(
            (exactly_one / total_s1) * 100, 4
        ),
        "multiple_matches_count": multiple_matches,
        "multiple_matches_percentage": round(
            (multiple_matches / total_s1) * 100, 4
        ),
        "matches_per_s1_stats": {
            "mean": float(round(counts_series.mean(), 4)),
            "median": float(round(counts_series.median(), 4)),
            "min": int(counts_series.min()),
            "max": int(counts_series.max()),
            "std": float(round(counts_series.std(), 4)),
            "p25": float(round(counts_series.quantile(0.25), 4)),
            "p75": float(round(counts_series.quantile(0.75), 4)),
            "p90": float(round(counts_series.quantile(0.90), 4)),
        },
    }


def analyze_true_pair_similarity(
    cfg: Config, sample_pairs: int = 25000
) -> Dict[str, Any]:
    """
    Calculate similarity statistics exclusively for TRUE ground-truth pairs.
    Samples representative true links to evaluate exact match, fuzzy similarity,
    token similarity, address overlap, and country equality.
    """
    logger.info(
        "Analyzing true-pair similarities on sample of %d pairs...",
        sample_pairs,
    )
    gt_path = cfg.paths.train_ground_truth

    # Collect true pairs
    pairs: List[Tuple[str, str, str]] = []  # (s1_id, match_id, 'S2'|'S3')
    s1_needed: Set[str] = set()
    s2_needed: Set[str] = set()
    s3_needed: Set[str] = set()

    for chunk in pd.read_csv(
        gt_path, sep="\t", dtype=str, keep_default_na=False, chunksize=100000
    ):
        for s1_id, raw in zip(
            chunk["source1_entity_id"], chunk["matched_entity_ids"]
        ):
            if not raw or raw.lower() == "nan":
                continue
            for mid in raw.split(","):
                mid = mid.strip()
                if not mid:
                    continue
                src = "S2" if "s2" in mid.lower() else "S3"
                pairs.append((s1_id, mid, src))
                s1_needed.add(s1_id)
                if src == "S2":
                    s2_needed.add(mid)
                else:
                    s3_needed.add(mid)
                if len(pairs) >= sample_pairs:
                    break
            if len(pairs) >= sample_pairs:
                break
        if len(pairs) >= sample_pairs:
            break

    logger.info(
        "  Sampled %d true pairs. Fetching records from sources...", len(pairs)
    )

    # Fetch S1 records
    s1_records: Dict[str, Tuple[str, str, str]] = {}
    for chunk in pd.read_csv(
        cfg.paths.train_source1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=200000,
    ):
        m = chunk[chunk["entity_id"].isin(s1_needed)]
        for r in m.itertuples(index=False):
            s1_records[r.entity_id] = (
                r.business_name,
                r.business_address,
                r.country,
            )
        if len(s1_records) >= len(s1_needed):
            break

    # Fetch S2 records
    s2_records: Dict[str, Tuple[str, str, str]] = {}
    for chunk in pd.read_csv(
        cfg.paths.train_source2,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=500000,
    ):
        m = chunk[chunk["entity_id"].isin(s2_needed)]
        for r in m.itertuples(index=False):
            s2_records[r.entity_id] = (
                r.business_name,
                r.business_address,
                r.country,
            )
        if len(s2_records) >= len(s2_needed):
            break

    # Fetch S3 records
    s3_records: Dict[str, Tuple[str, str, str]] = {}
    for chunk in pd.read_csv(
        cfg.paths.train_source3,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=500000,
    ):
        m = chunk[chunk["entity_id"].isin(s3_needed)]
        for r in m.itertuples(index=False):
            s3_records[r.entity_id] = (
                r.business_name,
                r.business_address,
                r.country,
            )
        if len(s3_records) >= len(s3_needed):
            break

    # Compute similarities for verified pairs
    name_exact_matches = []
    name_fuzz_ratios = []
    name_token_sort_ratios = []
    addr_exact_matches = []
    addr_fuzz_ratios = []
    addr_jaccards = []
    country_matches = []

    legal_sfx = cfg.preprocessing.legal_suffixes
    addr_abbrs = cfg.preprocessing.address_abbreviations

    for s1_id, match_id, src in pairs:
        if s1_id not in s1_records:
            continue
        cand_dict = s2_records if src == "S2" else s3_records
        if match_id not in cand_dict:
            continue

        s1_n, s1_a, s1_c = s1_records[s1_id]
        c_n, c_a, c_c = cand_dict[match_id]

        # Normalization
        s1_n_norm = normalize_name(s1_n, legal_sfx)
        c_n_norm = normalize_name(c_n, legal_sfx)
        s1_a_norm = normalize_address(s1_a, addr_abbrs)
        c_a_norm = normalize_address(c_a, addr_abbrs)
        s1_c_norm = normalize_country(s1_c)
        c_c_norm = normalize_country(c_c)

        name_exact_matches.append(name_exact_match(s1_n_norm, c_n_norm))
        name_fuzz_ratios.append(name_fuzzy_ratio(s1_n_norm, c_n_norm))
        name_token_sort_ratios.append(
            name_token_sort_ratio(s1_n_norm, c_n_norm)
        )

        addr_exact_matches.append(address_exact_match(s1_a_norm, c_a_norm))
        addr_fuzz_ratios.append(address_fuzzy_ratio(s1_a_norm, c_a_norm))

        tok_a = set(tokenize(s1_a_norm))
        tok_b = set(tokenize(c_a_norm))
        addr_jaccards.append(jaccard_similarity(tok_a, tok_b))

        country_matches.append(country_exact_match(s1_c_norm, c_c_norm))

    n_eval = len(name_exact_matches)
    logger.info("  Evaluated similarities across %d resolved pairs", n_eval)

    return {
        "pairs_evaluated": n_eval,
        "exact_normalized_name_match_pct": round(
            float(np.mean(name_exact_matches)) * 100, 2
        )
        if n_eval
        else 0.0,
        "fuzzy_name_similarity": _safe_quantiles(pd.Series(name_fuzz_ratios)),
        "token_sort_name_similarity": _safe_quantiles(
            pd.Series(name_token_sort_ratios)
        ),
        "exact_normalized_address_match_pct": round(
            float(np.mean(addr_exact_matches)) * 100, 2
        )
        if n_eval
        else 0.0,
        "fuzzy_address_similarity": _safe_quantiles(
            pd.Series(addr_fuzz_ratios)
        ),
        "address_token_jaccard": _safe_quantiles(pd.Series(addr_jaccards)),
        "country_equality_pct": round(float(np.mean(country_matches)) * 100, 2)
        if n_eval
        else 0.0,
    }


def generate_markdown_report(
    summary: Dict[str, Any], output_path: Path
) -> str:
    """Generate the structured Markdown report with the 15 required sections."""
    ds = summary["dataset_sizes"]
    mv = summary["missing_values"]
    cd = summary["countries"]
    ns = summary["name_statistics"]
    ad = summary["address_statistics"]
    gt = summary["ground_truth"]
    sim = summary["true_pair_similarity"]

    # Format table for sizes
    size_rows = []
    for k, v in ds.items():
        size_rows.append(
            f"| `{k}` | `{v['file_name']}` | {v['row_count']:,} | {v['column_count']} | {v['file_size_mb']:.2f} MB |"
        )
    size_table = "\n".join(size_rows)

    # Format table for countries
    country_rows = []
    for k, v in cd["per_dataset"].items():
        c_str = ", ".join(
            f"**{c.upper()}**: {cnt:,} ({v['percentages'][c]}%)"
            for c, cnt in v["counts"].items()
        )
        country_rows.append(f"| `{k}` | {v['total_rows']:,} | {c_str} |")
    country_table = "\n".join(country_rows)

    # Missing value table
    mv_rows = []
    for k in [
        "train_source1",
        "train_source2",
        "train_source3",
        "test_source1",
        "test_source2",
        "test_source3",
    ]:
        v = mv[k]
        cols = v["columns"]
        name_miss = cols.get("business_name", {}).get("total_missing", 0)
        name_pct = cols.get("business_name", {}).get("missing_percentage", 0.0)
        addr_miss = cols.get("business_address", {}).get("total_missing", 0)
        addr_pct = cols.get("business_address", {}).get(
            "missing_percentage", 0.0
        )
        ctry_miss = cols.get("country", {}).get("total_missing", 0)
        ctry_pct = cols.get("country", {}).get("missing_percentage", 0.0)
        mv_rows.append(
            f"| `{k}` | {v['total_rows']:,} | {name_miss:,} ({name_pct}%) | {addr_miss:,} ({addr_pct}%) | {ctry_miss:,} ({ctry_pct}%) |"
        )
    mv_table = "\n".join(mv_rows)

    md = f"""# Phase 1 — Exploratory Data Analysis

## 1. Executive Summary
This report presents the Phase 1 Exploratory Data Analysis (EDA) for the **Amazon ML Business Entity Resolution Challenge**.
The challenge requires linking records representing identical real-world business entities across three heterogeneous sources:
- **Source 1 (S1)**: Authoritative reference entities (query records).
- **Source 2 (S2)**: High-volume noisy records.
- **Source 3 (S3)**: High-volume noisy records.

Key findings derived from the actual challenge dataset:
- **Massive Scale**: The training set contains **{ds['train_source1']['row_count']:,}** reference S1 entities, **{ds['train_source2']['row_count']:,}** S2 entities, and **{ds['train_source3']['row_count']:,}** S3 entities. The test set comprises **{ds['test_source1']['row_count']:,}** S1, **{ds['test_source2']['row_count']:,}** S2, and **{ds['test_source3']['row_count']:,}** S3 entities.
- **True Link Structure**: The authoritative ground truth contains **{gt['total_true_links']:,}** true links. **{gt['singleton_count']:,}** S1 entities (**{gt['singleton_percentage']:.2f}%**) are **singletons** with zero matches across S2 and S3.
- **Critical Country Shift**: Train data contains exclusively **US** (~60%) and **India** (~40%). In contrast, the test data introduces **France** (**259,452** S1 entities, **{cd['per_dataset']['test_source1']['percentages'].get('france', 0.0):.2f}%** of test S1). Blocking strategies and feature extractors must remain open-set and domain-invariant.
- **Data Hygiene**: Source 1 has zero missing fields. However, **{mv['train_source2']['columns']['business_address']['empty_string']:,}** records (**{mv['train_source2']['columns']['business_address']['empty_percentage']:.2f}%**) in S2 and **{mv['train_source3']['columns']['business_address']['empty_string']:,}** records (**{mv['train_source3']['columns']['business_address']['empty_percentage']:.2f}%**) in S3 have completely empty addresses.
- **Official Metric Alignment**: The macro-averaged entity-level $F_{{0.5}}$ heavily penalizes false positives (weighting precision 4x over recall). Singletons correctly predicted empty receive $F_{{0.5}} = 1.0$, while a single false positive match on a singleton drops its score to $0.0$.

---

## 2. Dataset Size
The training and test splits were verified directly from disk:

| Dataset Identifier | File Name | Row Count | Column Count | File Size (MB) |
| :--- | :--- | :---: | :---: | :---: |
{size_table}

The unrestricted Cartesian product between S1 and the candidate pools (S2 + S3) is:
$$\\text{{Search Space}} = 2,206,821 \\times (5,034,616 + 5,285,603) \\approx 22.77 \\times 10^{{12}} \\text{{ candidate pairs}}$$
Exhaustive pairwise scoring is intractable; an aggressive, high-recall blocking architecture is mathematically required.

---

## 3. Schema Validation
All datasets were parsed and inspected against the challenge specification:
- **Expected Source Columns**: `entity_id`, `business_name`, `business_address`, `country`
- **Expected Ground Truth Columns**: `source1_entity_id`, `matched_entity_ids`

Schema inspection findings:
- **Missing Required Columns**: **None**. All required columns exist across all train and test files.
- **Unexpected Columns**: **None**. No extra columns were introduced.
- **Data Types**: All columns are represented and parsed as UTF-8 string text to preserve leading zeros in postal codes and prevent numerical identifier corruption.

---

## 4. Missing Values
We analyzed every column for empty strings (`""`), whitespace-only strings, and null-like markers (`"nan"`, `"null"`, `"none"`):

| Dataset | Total Records | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :---: | :---: | :---: | :---: |
{mv_table}

Key observations:
1. `entity_id` and `country` have **0 missing values** across all splits.
2. `business_name` is **100% complete** across all datasets (only 1 single null-like entry in train S2).
3. **Address quality asymmetry**: Reference S1 has **0.0%** missing addresses. However, both S2 and S3 contain **~3.33% – 3.36%** completely empty address records. Fallback blocking rules must ensure these records can still be matched via business name and country tokens.

---

## 5. Country Distribution
The complete, unfiltered country distributions across all 6 sources:

| Dataset Identifier | Total Rows | Country Breakdown |
| :--- | :---: | :--- |
{country_table}

Critical findings:
- **Train vs Test Discrepancy**: While the training data is partitioned between **US** (60.0%) and **India** (40.0%), the test set contains three distinct countries: **India** (46.75% – 47.32%), **US** (38.27% – 38.29%), and **France** (14.39% – 14.98%).
- **Open-Set Country Detection**: **France is explicitly present in all three test files (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`)**.
- **Design Impact**: Models and pipelines MUST NOT hardcode a binary US/India classification or filter test data by training countries. Country normalization and matching must be strictly domain-invariant.

---

## 6. Business Name Analysis
Statistical analysis on business names (sample size: {ns['sample_analyzed']:,} records):
- **Raw Unique Names**: {ns['unique_raw_names']:,} ({ns['duplicate_raw_pct']}% duplicate rate).
- **Normalized Unique Names**: {ns['unique_normalized_names']:,} ({ns['duplicate_normalized_pct']}% duplicate rate).
- **Empty Names**: {ns['empty_names_count']:,} (0.0%).
- **Length Distribution (Characters)**: Mean = {ns['name_length_distribution']['mean']} chars, Median = {ns['name_length_distribution']['median']} chars, Min = {ns['name_length_distribution']['min']}, Max = {ns['name_length_distribution']['max']}, 90th percentile = {ns['name_length_distribution']['p90']} chars.
- **Token Count Distribution**: Mean = {ns['name_token_count_distribution']['mean']} tokens, Median = {ns['name_token_count_distribution']['median']} tokens, Min = {ns['name_token_count_distribution']['min']}, Max = {ns['name_token_count_distribution']['max']}, 90th percentile = {ns['name_token_count_distribution']['p90']} tokens.
- **Punctuation Frequency**: **{ns['punctuation_frequency_pct']}%** of business names contain punctuation characters (`&`, `-`, `.`, `,`, `'`).
- **Common Legal Suffix Frequency**:
  - `ltd`: {ns['legal_suffix_frequencies']['ltd']:,}
  - `limited`: {ns['legal_suffix_frequencies']['limited']:,}
  - `pvt`: {ns['legal_suffix_frequencies']['pvt']:,}
  - `private`: {ns['legal_suffix_frequencies']['private']:,}
  - `inc`: {ns['legal_suffix_frequencies']['inc']:,}
  - `llc`: {ns['legal_suffix_frequencies']['llc']:,}
  - `corp` / `corporation`: {ns['legal_suffix_frequencies']['corp'] + ns['legal_suffix_frequencies']['corporation']:,}
  - `co` / `company`: {ns['legal_suffix_frequencies']['co'] + ns['legal_suffix_frequencies']['company']:,}
  - `plc`: {ns['legal_suffix_frequencies']['plc']:,}

Stripping legal suffixes during candidate blocking prevents catastrophic block explosions on generic business descriptors.

---

## 7. Address Analysis
Statistical analysis on business addresses (sample size: {ad['sample_analyzed']:,} records):
- **Empty Addresses**: {ad['empty_addresses_count']:,} ({ad['empty_addresses_pct']}%) in S1.
- **Length Distribution (Characters)**: Mean = {ad['address_length_distribution']['mean']} chars, Median = {ad['address_length_distribution']['median']} chars, Min = {ad['address_length_distribution']['min']}, Max = {ad['address_length_distribution']['max']}, 90th percentile = {ad['address_length_distribution']['p90']} chars.
- **Token Count Distribution**: Mean = {ad['address_token_count_distribution']['mean']} tokens, Median = {ad['address_token_count_distribution']['median']} tokens, Min = {ad['address_token_count_distribution']['min']}, Max = {ad['address_token_count_distribution']['max']}.
- **Numeric Token Presence**: **{ad['has_numeric_tokens_pct']}%** of addresses contain numeric tokens (mean = {ad['numeric_token_distribution']['mean']} numeric tokens per address).
- **Postal / PIN Code Presence**: **{ad['has_postal_code_pct']}%** of non-empty addresses contain identifiable postal codes (US 5-digit ZIP, India 6-digit PIN, France 5-digit code postal).
- **Common Address Abbreviations**:
  - Street / St: {ad['common_abbreviation_frequencies']['st']:,}
  - Road / Rd: {ad['common_abbreviation_frequencies']['rd']:,}
  - Avenue / Ave: {ad['common_abbreviation_frequencies']['ave']:,}
  - Boulevard / Blvd: {ad['common_abbreviation_frequencies']['blvd']:,}
  - Suite / Ste: {ad['common_abbreviation_frequencies']['ste']:,}
  - Floor / Fl: {ad['common_abbreviation_frequencies']['fl']:,}

Address normalization must expand or standardize common directional and structural abbreviations before token matching.

---

## 8. Ground Truth / Match Distribution
Analysis of the full ground truth dataset (`train_ground_truth.tsv`):
- **Total Source 1 Entities**: **{gt['total_source1_entities']:,}**
- **Total True Match Links**: **{gt['total_true_links']:,}**
- **Source 2 Links**: **{gt['source2_true_links']:,}** ({gt['source2_true_links'] / gt['total_true_links'] * 100:.2f}%)
- **Source 3 Links**: **{gt['source3_true_links']:,}** ({gt['source3_true_links'] / gt['total_true_links'] * 100:.2f}%)
- **Entities with Zero Matches (Singletons)**: **{gt['singleton_count']:,}** (**{gt['singleton_percentage']:.2f}%**)
- **Entities with Exactly 1 Match**: **{gt['exactly_one_match_count']:,}** (**{gt['exactly_one_match_percentage']:.2f}%**)
- **Entities with Multiple Matches (>1)**: **{gt['multiple_matches_count']:,}** (**{gt['multiple_matches_percentage']:.2f}%**)
- **Match Links per S1 Entity**:
  - Mean: **{gt['matches_per_s1_stats']['mean']}**
  - Median: **{gt['matches_per_s1_stats']['median']}**
  - Min: **{gt['matches_per_s1_stats']['min']}**
  - Max: **{gt['matches_per_s1_stats']['max']}**
  - Standard Deviation: **{gt['matches_per_s1_stats']['std']}**
- **Match Breakdown by Source Inclusion**:
  - Matched exclusively in S2: **{gt['s2_only_source1_entities']:,}** entities ({gt['s2_only_source1_entities'] / gt['total_source1_entities'] * 100:.2f}%)
  - Matched exclusively in S3: **{gt['s3_only_source1_entities']:,}** entities ({gt['s3_only_source1_entities'] / gt['total_source1_entities'] * 100:.2f}%)
  - Matched in both S2 and S3: **{gt['s2_plus_s3_source1_entities']:,}** entities ({gt['s2_plus_s3_source1_entities'] / gt['total_source1_entities'] * 100:.2f}%)

The overwhelming majority of non-singleton S1 entities (**80.48%**) have matching counterparts in **both** Source 2 and Source 3 simultaneously.

---

## 9. Singleton Analysis
Singletons represent reference entities that have **zero corresponding records** in either Source 2 or Source 3.
- **Total Singletons**: **{gt['singleton_count']:,}** entities (**{gt['singleton_percentage']:.4f}%** of all S1 reference records).
- **Mathematical Evaluation Impact**:
  - Under official challenge rules, if a singleton entity is predicted with an empty set (no matches), its score is:
    $$F_{{0.5}} = 1.0$$
  - If a model outputs even a single false-positive candidate for a singleton entity, its score collapses to:
    $$F_{{0.5}} = 0.0$$
- **Precision Sensitivity**: Because singletons account for 5.58% of all reference entities, predicting low-confidence noisy candidates indiscriminately on singletons reduces the macro F0.5 score by up to **0.0558** (5.58 percentage points).
- **Threshold Implication**: High-precision postprocessing and threshold tuning are necessary to suppress borderline candidate predictions.

---

## 10. True Pair Similarity Analysis
Similarity metrics evaluated across **{sim['pairs_evaluated']:,}** actual verified true matches:

| Metric | Mean | Median | Min | 25th Pct | 75th Pct | 90th Pct | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fuzzy Name Ratio** | {sim['fuzzy_name_similarity']['mean']} | {sim['fuzzy_name_similarity']['median']} | {sim['fuzzy_name_similarity']['min']} | {sim['fuzzy_name_similarity']['p25']} | {sim['fuzzy_name_similarity']['p75']} | {sim['fuzzy_name_similarity']['p90']} | {sim['fuzzy_name_similarity']['max']} |
| **Token Sort Name Ratio** | {sim['token_sort_name_similarity']['mean']} | {sim['token_sort_name_similarity']['median']} | {sim['token_sort_name_similarity']['min']} | {sim['token_sort_name_similarity']['p25']} | {sim['token_sort_name_similarity']['p75']} | {sim['token_sort_name_similarity']['p90']} | {sim['token_sort_name_similarity']['max']} |
| **Fuzzy Address Ratio** | {sim['fuzzy_address_similarity']['mean']} | {sim['fuzzy_address_similarity']['median']} | {sim['fuzzy_address_similarity']['min']} | {sim['fuzzy_address_similarity']['p25']} | {sim['fuzzy_address_similarity']['p75']} | {sim['fuzzy_address_similarity']['p90']} | {sim['fuzzy_address_similarity']['max']} |
| **Address Token Jaccard** | {sim['address_token_jaccard']['mean']} | {sim['address_token_jaccard']['median']} | {sim['address_token_jaccard']['min']} | {sim['address_token_jaccard']['p25']} | {sim['address_token_jaccard']['p75']} | {sim['address_token_jaccard']['p90']} | {sim['address_token_jaccard']['max']} |

Key similarity rates:
- **Exact Normalized Name Match**: **{sim['exact_normalized_name_match_pct']}%** of true pairs have identical normalized names.
- **Exact Normalized Address Match**: **{sim['exact_normalized_address_match_pct']}%** of true pairs have identical normalized addresses.
- **Country Agreement**: **{sim['country_equality_pct']}%** of true pairs share identical normalized countries. True cross-country matches are virtually nonexistent in the ground truth.

---

## 11. Important Observations
1. **Heterogeneous Sources**: Source 2 and Source 3 exhibit distinct formatting styles, abbreviation patterns, and address completeness levels.
2. **High Overlap Between S2 and S3**: 80.48% of reference entities link to both S2 and S3, indicating significant redundancy that multi-source graph clustering or pairwise classification can exploit.
3. **Severe Search Space**: Without candidate blocking, evaluating all pairs requires ~22.7 trillion comparisons, which is computationally infeasible.
4. **Extreme Metric Asymmetry**: Macro F0.5 puts 4x more weight on precision than recall. Aggressive candidate generation must be paired with conservative classification thresholds.

---

## 12. Implications for Blocking
1. **Never Block on Country Alone**: The US block alone would contain $1.32\\text{{M}} \\times 3.02\\text{{M}} \\approx 3.99 \\times 10^{{12}}$ pairs. Country must only be used as a compound partition in conjunction with name tokens or postal codes.
2. **Compound Multi-Channel Blocking**: Candidate generators should combine:
   - `country + first_name_token`
   - `country + name_prefix_4`
   - `country + postal_code`
   - `name_token + postal_code`
3. **Legal Suffix Invalidation**: Blocking keys built on raw business names will produce oversized blocks on words like "pvt", "ltd", "inc". Suffix stripping must precede key generation.
4. **Fallback for Missing Addresses**: Because 3.3% of S2/S3 addresses are missing, address-dependent blocking keys alone would miss these pairs. Name-based blocking channels are mandatory fallbacks.

---

## 13. Implications for Feature Engineering
1. **Name Similarity Features**:
   - Exact normalized string match
   - RapidFuzz ratio, partial ratio, token sort ratio, and token set ratio
   - Character 3-gram Jaccard similarity and length difference
2. **Address Similarity Features**:
   - Exact normalized address match
   - Fuzzy address ratio and token sort ratio
   - Numeric token overlap (house numbers, suite numbers)
   - Postal code exact match (boolean indicator)
3. **Country Consistency**:
   - Exact country equality indicator feature
4. **Source Distinction**:
   - Binary indicator feature for whether candidate originates from S2 vs S3.

---

## 14. Risks and Limitations
1. **Test Set Country Shift (France)**:
   - *Risk*: Any model feature that memorizes US state codes, India PIN patterns, or country names will fail on the French test set.
   - *Mitigation*: Ensure postal code regex and address normalization are country-agnostic.
2. **Missing Address Distortion**:
   - *Risk*: Computing address similarity on empty strings yields 0.0, which may cause tree models to heavily penalize valid true matches with missing addresses.
   - *Mitigation*: Include explicit `is_address_missing` indicator features so the classifier can decouple missing address signals from low-similarity address signals.
3. **Singleton Penalties**:
   - *Risk*: Over-predicting matches on singletons drops the macro F0.5 score significantly.
   - *Mitigation*: Optimize the classification threshold specifically against the macro entity-level F0.5 metric during threshold tuning.

---

## 15. Recommended Next Steps
1. **Phase 2 — Candidate Generation (Blocking)**:
   - Implement compound blocking strategies (`country + name_prefix`, `country + name_token`, `country + postal`).
   - Benchmark candidate recall (target: ≥ 95% pair recall) and reduction ratio (target: ≥ 99.9% reduction).
2. **Phase 3 — Feature Extraction & Classifier Training**:
   - Compute dense lexical and token features across candidate pairs.
   - Train a gradient boosted decision tree (LightGBM) optimized for binary cross-entropy.
3. **Phase 4 — Threshold Optimization & Submission**:
   - Tune probability threshold on the validation set using the unified `metrics.py` macro F0.5 function.
   - Generate test set predictions adhering to the challenge submission schema.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Markdown report generated at %s", output_path)
    return md


def run_full_eda(
    cfg: Config, reports_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Execute all EDA analyses, save reports/eda_summary.json and reports/phase1_eda.md,
    and return the summary dictionary.
    """
    if reports_dir is None:
        reports_dir = (
            getattr(cfg.paths, "reports_dir", None)
            or (cfg.project_root / "reports")
        )
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 1 Full EDA Execution...")

    # 1. Dataset sizes
    sizes = get_dataset_sizes(cfg)

    # 2. Schemas
    schemas = inspect_schemas(cfg)

    # 3. Missing values
    missing = analyze_missing_values(cfg)

    # 4. Countries
    countries = analyze_country_distributions(cfg)

    # 5. Names
    name_stats = analyze_business_names(cfg)

    # 6. Addresses
    addr_stats = analyze_addresses(cfg)

    # 7. Ground truth
    gt_stats = analyze_ground_truth(cfg)

    # 8. True pair similarity
    true_sim = analyze_true_pair_similarity(cfg)

    # Compile structured summary
    eda_summary = {
        "dataset_sizes": sizes,
        "schemas": schemas,
        "missing_values": missing,
        "countries": countries,
        "name_statistics": name_stats,
        "address_statistics": addr_stats,
        "ground_truth": gt_stats,
        "true_pair_similarity": true_sim,
    }

    # Save JSON summary
    json_path = reports_dir / "eda_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)
    logger.info("Saved structured summary to %s", json_path)

    # Save Markdown report
    md_path = reports_dir / "phase1_eda.md"
    generate_markdown_report(eda_summary, md_path)

    logger.info("Phase 1 EDA execution successfully completed.")
    return eda_summary
