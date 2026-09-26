"""
candidate_generation.py — Multi-strategy candidate pair generation
===================================================================
Combines blocking indices from ``blocking.py`` to produce a union
of candidate pairs ``(source1_id, candidate_id)``.

Supports informative compound blocking, rule provenance tracking,
similarity-based pre-ranking (no arbitrary ID truncation), and
candidate recall measurement against ground truth.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .blocking import BlockIndex, build_block_indices, ADDRESS_STOPWORDS
from .config import Config
from .preprocessing import tokenize, char_ngrams

logger = logging.getLogger(__name__)


# ================================================================== #
#  Fast pool record representation for pre-ranking
# ================================================================== #

class PoolRecordSummary:
    """Pre-extracted lightweight features for fast similarity pre-ranking."""
    __slots__ = ("name", "postal_code", "country", "source")

    def __init__(self, name: str, address: str = "", postal: str = "", country: str = "", source: str = "S2"):
        self.name = name
        self.postal_code = postal
        self.country = country
        self.source = source


# ================================================================== #
#  Candidate lookup for a single S1 entity
# ================================================================== #

def _candidates_for_entity(
    s1_row: pd.Series,
    indices: Dict[str, BlockIndex],
    cfg: "Config",
    pool_records: Dict[str, PoolRecordSummary],
    true_matches_for_s1: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[str]], int, int, int]:
    """
    Return candidate IDs with rule provenance for one Source-1 entity.

    Returns
    -------
    (cand_rules, n_before_cap, n_after_cap, n_true_lost_to_cap)
    where cand_rules maps candidate_id -> list of rule names.
    """
    bcfg = cfg.blocking
    cand_rules: Dict[str, List[str]] = defaultdict(list)

    name = s1_row.get("business_name_normalized", "")
    address = s1_row.get("business_address_normalized", "")
    country = s1_row.get("country_normalized", "")
    postal = s1_row.get("postal_code", "")

    name_tokens = [tok for tok in tokenize(name) if len(tok) >= 2]
    addr_tokens = [tok for tok in tokenize(address) if len(tok) >= 2 and tok not in ADDRESS_STOPWORDS]
    pfx = name[: bcfg.name_prefix_length] if len(name) >= bcfg.name_prefix_length else ""

    # 1. Country block (broad - disabled by default)
    if bcfg.country_block and "country" in indices and country:
        for cid in indices["country"].get(f"country:{country}", set()):
            cand_rules[cid].append("country")

    # 2. Name token block
    if bcfg.name_token_block and "name_token" in indices:
        for tok in name_tokens:
            for cid in indices["name_token"].get(f"ntok:{tok}", set()):
                cand_rules[cid].append("name_token")

    # 3. Name prefix block
    if bcfg.name_prefix_block and "name_prefix" in indices and pfx:
        for cid in indices["name_prefix"].get(f"npfx:{pfx}", set()):
            cand_rules[cid].append("name_prefix")

    # 4. Address token block
    if bcfg.address_token_block and "address_token" in indices:
        for tok in addr_tokens:
            for cid in indices["address_token"].get(f"atok:{tok}", set()):
                cand_rules[cid].append("address_token")

    # 5. Postal code block
    if bcfg.postal_code_block and "postal_code" in indices and postal:
        for cid in indices["postal_code"].get(f"post:{postal}", set()):
            cand_rules[cid].append("postal_code")

    # 6. Character n-gram block
    if bcfg.char_ngram_block and "char_ngram" in indices:
        ngrams = char_ngrams(name, bcfg.char_ngram_n)
        ngram_hits: Dict[str, int] = defaultdict(int)
        for ng in ngrams:
            for cid in indices["char_ngram"].get(f"cng:{ng}", set()):
                ngram_hits[cid] += 1
        for cid, count in ngram_hits.items():
            if count >= bcfg.char_ngram_min_shared:
                cand_rules[cid].append("char_ngram")

    # 7. Compound: country + name token
    if bcfg.country_name_token_block and "country_name_token" in indices and country:
        for tok in name_tokens:
            for cid in indices["country_name_token"].get(f"c_ntok:{country}::{tok}", set()):
                cand_rules[cid].append("country_name_token")

    # 8. Compound: country + name prefix
    if bcfg.country_name_prefix_block and "country_name_prefix" in indices and country and pfx:
        for cid in indices["country_name_prefix"].get(f"c_npfx:{country}::{pfx}", set()):
            cand_rules[cid].append("country_name_prefix")

    # 9. Compound: country + postal code
    if bcfg.country_postal_block and "country_postal" in indices and country and postal:
        for cid in indices["country_postal"].get(f"c_post:{country}::{postal}", set()):
            cand_rules[cid].append("country_postal")

    # 10. Compound: country + address token
    if bcfg.country_address_token_block and "country_address_token" in indices and country:
        for tok in addr_tokens:
            for cid in indices["country_address_token"].get(f"c_atok:{country}::{tok}", set()):
                cand_rules[cid].append("country_address_token")

    # 11. Compound: name token + postal code
    if bcfg.name_token_postal_block and "name_token_postal" in indices and postal:
        for tok in name_tokens:
            for cid in indices["name_token_postal"].get(f"ntok_post:{tok}::{postal}", set()):
                cand_rules[cid].append("name_token_postal")

    # 12. Compound: name prefix + postal code
    if bcfg.name_prefix_postal_block and "name_prefix_postal" in indices and postal and pfx:
        for cid in indices["name_prefix_postal"].get(f"npfx_post:{pfx}::{postal}", set()):
            cand_rules[cid].append("name_prefix_postal")

    # 13. Compound: name token + address token
    if bcfg.name_token_address_token_block and "name_token_address_token" in indices:
        ntoks = [t for t in name_tokens if len(t) >= 3][:3]
        atoks = [t for t in addr_tokens if len(t) >= 3][:3]
        for nt in ntoks:
            for at in atoks:
                for cid in indices["name_token_address_token"].get(f"ntok_atok:{nt}::{at}", set()):
                    cand_rules[cid].append("name_token_address_token")

    n_before_cap = len(cand_rules)
    n_after_cap = n_before_cap
    n_true_lost = 0

    # Capping via similarity-based pre-ranking (never arbitrary ID ordering)
    if n_before_cap > bcfg.max_candidates_per_s1:
        s1_name_set = set(name_tokens)
        s1_addr_set = set(addr_tokens)

        # Deterministic similarity pre-ranking
        def _score(cid: str) -> float:
            rec = pool_records.get(cid)
            if not rec:
                return 0.0
            rec_toks = set(rec.name.split())
            inter_name = len(s1_name_set & rec_toks)
            union_name = len(s1_name_set | rec_toks)
            name_jaccard = inter_name / union_name if union_name > 0 else 0.0
            postal_match = 1.0 if (postal and postal == rec.postal_code) else 0.0
            country_match = 1.0 if (country and country == rec.country) else 0.0
            rule_count = len(cand_rules[cid])
            return 4.0 * name_jaccard + 2.0 * postal_match + 1.0 * country_match + 0.5 * rule_count

        # Sort by (-score, cid) for deterministic ranking
        ranked_cids = sorted(cand_rules.keys(), key=lambda cid: (-_score(cid), cid))
        selected_cids = set(ranked_cids[: bcfg.max_candidates_per_s1])

        if true_matches_for_s1:
            lost = (true_matches_for_s1 & set(cand_rules.keys())) - selected_cids
            n_true_lost = len(lost)
            if n_true_lost > 0:
                logger.warning(
                    "Candidate cap dropped %d true match(es) for %s! True lost: %s",
                    n_true_lost, s1_row.get("entity_id"), lost
                )

        cand_rules = {cid: cand_rules[cid] for cid in selected_cids}
        n_after_cap = len(cand_rules)

    return cand_rules, n_before_cap, n_after_cap, n_true_lost


# ================================================================== #
#  Full candidate generation
# ================================================================== #

class CandidateIndex:
    """Pre-built blocking indices and pool record metadata for fast streaming queries."""

    def __init__(self, pool: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.indices = build_block_indices(pool, cfg.blocking)
        self.pool_records: Dict[str, PoolRecordSummary] = {}

        names = pool["business_name_normalized"] if "business_name_normalized" in pool.columns else pool.get("business_name", "")
        postals = pool["postal_code"] if "postal_code" in pool.columns else [""] * len(pool)
        countries = pool["country_normalized"] if "country_normalized" in pool.columns else pool.get("country", "")
        sources = pool["_source"] if "_source" in pool.columns else ["S2"] * len(pool)

        for eid, name, postal, country, src in zip(
            pool["entity_id"], names, postals, countries, sources
        ):
            self.pool_records[str(eid)] = PoolRecordSummary(
                str(name), postal=str(postal), country=str(country), source=str(src)
            )

    def query(
        self,
        s1: pd.DataFrame,
        ground_truth_map: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[pd.DataFrame, int, int, int]:
        """
        Query candidates for an S1 DataFrame against the pre-built index.
        Returns (cand_df, total_before_cap, total_after_cap, total_lost_to_cap)
        """
        rows = []
        total_before_cap = 0
        total_after_cap = 0
        total_lost_to_cap = 0

        s1_rows = s1.to_dict(orient="records")
        for s1_row in s1_rows:
            s1_id = s1_row["entity_id"]
            true_for_s1 = ground_truth_map.get(s1_id) if ground_truth_map else None
            cands_with_rules, n_before, n_after, n_lost = _candidates_for_entity(
                s1_row, self.indices, self.cfg, self.pool_records, true_matches_for_s1=true_for_s1,
            )
            total_before_cap += n_before
            total_after_cap += n_after
            total_lost_to_cap += n_lost

            for cid, rules in cands_with_rules.items():
                if cid != s1_id:
                    rec = self.pool_records.get(cid)
                    if rec is not None:
                        unique_rules = ",".join(sorted(set(rules)))
                        rows.append((s1_id, cid, rec.source, unique_rules))

        cand_df = pd.DataFrame(rows, columns=["source1_id", "candidate_id", "source", "generated_by"])
        cand_df.drop_duplicates(subset=["source1_id", "candidate_id"], inplace=True)
        return cand_df, total_before_cap, total_after_cap, total_lost_to_cap


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    cfg: Config,
    ground_truth_map: Optional[Dict[str, Set[str]]] = None,
) -> pd.DataFrame:
    """
    Generate candidate pairs for every Source-1 entity.
    """
    logger.info("Building blocking indices for candidate pool (S2 + S3) …")

    if "_source" not in s2.columns:
        s2 = s2.assign(_source="S2")
    if "_source" not in s3.columns:
        s3 = s3.assign(_source="S3")
    pool = pd.concat([s2, s3], ignore_index=True)

    c_idx = CandidateIndex(pool, cfg)
    cand_df, total_before_cap, total_after_cap, total_lost_to_cap = c_idx.query(s1, ground_truth_map)

    # ---- Statistics ------------------------------------------------
    n_s1 = len(s1)
    n_pool = len(pool)
    n_pairs = len(cand_df)
    total_possible = n_s1 * n_pool
    reduction = 1.0 - (n_pairs / total_possible) if total_possible > 0 else 0.0

    per_s1 = cand_df.groupby("source1_id")["candidate_id"].count()
    avg_per_s1 = per_s1.mean() if len(per_s1) > 0 else 0
    max_per_s1 = per_s1.max() if len(per_s1) > 0 else 0

    logger.info("Candidate generation statistics:")
    logger.info("  S1 entities             : %d", n_s1)
    logger.info("  Pool size (S2+S3)       : %d", n_pool)
    logger.info("  Total possible pairs    : %d", total_possible)
    logger.info("  Generated pairs         : %d", n_pairs)
    logger.info("  Reduction ratio         : %.4f", reduction)
    logger.info("  Avg candidates / S1     : %.1f", avg_per_s1)
    logger.info("  Max candidates / S1     : %d", max_per_s1)
    logger.info("  Candidates before cap   : %d", total_before_cap)
    logger.info("  Candidates after cap    : %d", total_after_cap)
    if ground_truth_map is not None:
        logger.info("  True matches lost to cap: %d", total_lost_to_cap)
        _report_candidate_recall(cand_df, ground_truth_map, cfg, total_before_cap, total_after_cap, total_lost_to_cap)

    return cand_df


# ================================================================== #
#  Candidate recall reporting
# ================================================================== #

def _report_candidate_recall(
    cand_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    cfg: Config,
    total_before_cap: int = 0,
    total_after_cap: int = 0,
    total_lost_to_cap: int = 0,
) -> Dict[str, float]:
    """
    Compute, log, and persist candidate recall and coverage metrics.
    """
    total_true = 0
    found = 0
    s1_all_covered = 0
    s1_any_covered = 0
    s1_with_truth = 0

    cand_sets: Dict[str, Set[str]] = (
        cand_df.groupby("source1_id")["candidate_id"]
        .apply(set)
        .to_dict()
    )

    for s1_id, true_ids in gt_map.items():
        if len(true_ids) == 0:
            continue
        s1_with_truth += 1
        total_true += len(true_ids)
        cands = cand_sets.get(s1_id, set())
        overlap = true_ids & cands
        overlap_count = len(overlap)
        found += overlap_count

        if overlap_count == len(true_ids):
            s1_all_covered += 1
        if overlap_count > 0:
            s1_any_covered += 1

    zero_cand_entities = sum(1 for s1_id in gt_map if len(cand_sets.get(s1_id, set())) == 0)
    missed = total_true - found

    if total_true == 0:
        recall = None
        pct_all = None
        pct_any = None
    else:
        recall = found / total_true
        pct_all = (s1_all_covered / s1_with_truth * 100.0) if s1_with_truth > 0 else 0.0
        pct_any = (s1_any_covered / s1_with_truth * 100.0) if s1_with_truth > 0 else 0.0

    logger.info("==================================================")
    logger.info("BLOCKING RECALL EVALUATION")
    logger.info("==================================================")
    if recall is not None:
        logger.info("  Candidate recall (link level) : %.4f  (%d / %d true links covered)", recall, found, total_true)
        logger.info("  Entities with ALL matches covered: %d (%.2f%%)", s1_all_covered, pct_all)
        logger.info("  Entities with ANY matches covered: %d (%.2f%%)", s1_any_covered, pct_any)
    else:
        logger.info("  Candidate recall (link level) : NOT_AVAILABLE (zero known true links in evaluated population)")
        logger.info("  Candidate recall not measured because the evaluation population contained zero known true links.")
    logger.info("  Total true links missed       : %d", missed)
    logger.info("  S1 entities with >=1 true match: %d", s1_with_truth)
    logger.info("  S1 entities with 0 candidates : %d", zero_cand_entities)
    logger.info("  Candidates before cap         : %d", total_before_cap)
    logger.info("  Candidates after cap          : %d", total_after_cap)
    logger.info("  True matches lost due to cap  : %d", total_lost_to_cap)
    logger.info("==================================================")

    metrics = {
        "candidate_recall": recall,
        "candidate_recall_status": "MEASURED" if total_true > 0 else "NOT_AVAILABLE",
        "total_true_links": total_true,
        "true_links_covered": found,
        "true_links_missed": missed,
        "s1_entities_with_truth": s1_with_truth,
        "entities_all_covered_count": s1_all_covered,
        "entities_all_covered_pct": pct_all,
        "entities_any_covered_count": s1_any_covered,
        "entities_any_covered_pct": pct_any,
        "entities_zero_candidates": zero_cand_entities,
        "candidates_before_cap": total_before_cap,
        "candidates_after_cap": total_after_cap,
        "true_matches_lost_to_cap": total_lost_to_cap,
        "message": (
            f"Candidate recall: {recall:.4f}"
            if recall is not None
            else "Candidate recall not measured because the evaluation population contained zero known true links."
        ),
    }

    # Save to metrics dir
    try:
        metrics_dir = cfg.paths.metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        recall_path = metrics_dir / "candidate_recall.json"
        with open(recall_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Candidate recall saved to %s", recall_path)
    except Exception as e:
        logger.warning("Could not save candidate_recall.json: %s", e)

    return metrics
