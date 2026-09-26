"""
feature_builder.py — Vectorised feature matrix construction
=============================================================
Takes a candidate-pairs DataFrame and the source DataFrames,
and produces a dense feature matrix ready for ML.

Supports TF-IDF cosine similarity via scikit-learn.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from .config import Config, FeaturesConfig
from .data_loader import SourceData
from .features import (
    name_exact_match,
    name_fuzzy_ratio,
    name_partial_ratio,
    name_token_sort_ratio,
    name_token_set_ratio,
    jaccard_similarity,
    common_token_count,
    token_count_diff,
    length_diff,
    char_similarity,
    address_exact_match,
    address_fuzzy_ratio,
    address_token_sort_ratio,
    numeric_token_overlap,
    postal_code_match,
    house_number_match,
    country_exact_match,
    source_is_s2,
    source_is_s3,
)
from .preprocessing import tokenize

logger = logging.getLogger(__name__)


# ================================================================== #
#  TF-IDF helper
# ================================================================== #

# ================================================================== #
#  TF-IDF Bundle for leak-free training & inference
# ================================================================== #

class TfidfBundle:
    """
    Holds trained TF-IDF vectorizers for name and address.
    Can be fitted on training corpus, saved to disk, and used for transforming
    validation or test corpora without data leakage.
    """

    def __init__(self, fcfg: FeaturesConfig):
        self.fcfg = fcfg
        self.feature_version = "2.0.0"
        self.is_fitted = False
        self.fit_stats: Dict[str, Any] = {}
        self.name_word_vec: Optional[TfidfVectorizer] = None
        self.name_char_vec: Optional[TfidfVectorizer] = None
        self.addr_word_vec: Optional[TfidfVectorizer] = None
        self.addr_char_vec: Optional[TfidfVectorizer] = None

    def fit(self, name_corpus: pd.Series, addr_corpus: pd.Series, corpus_stats: Optional[Dict[str, Any]] = None) -> None:
        """Fit vectorizers on training corpus only."""
        if self.fcfg.name_tfidf_cosine:
            self.name_word_vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.fcfg.tfidf.name_word_ngram_range,
                max_features=self.fcfg.tfidf.max_features,
                sublinear_tf=True,
            ).fit(name_corpus.fillna(""))

            self.name_char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.fcfg.tfidf.name_char_ngram_range,
                max_features=self.fcfg.tfidf.max_features,
                sublinear_tf=True,
            ).fit(name_corpus.fillna(""))
            logger.info("Fitted TF-IDF name vectorizers (word & char_wb).")

        if self.fcfg.address_tfidf_cosine:
            self.addr_word_vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=self.fcfg.tfidf.address_word_ngram_range,
                max_features=self.fcfg.tfidf.max_features,
                sublinear_tf=True,
            ).fit(addr_corpus.fillna(""))

            self.addr_char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.fcfg.tfidf.address_char_ngram_range,
                max_features=self.fcfg.tfidf.max_features,
                sublinear_tf=True,
            ).fit(addr_corpus.fillna(""))
            logger.info("Fitted TF-IDF address vectorizers (word & char_wb).")

        self.is_fitted = True
        self.fit_stats = corpus_stats or {}

    def compute_cosine(
        self,
        s1_texts: pd.Series,
        pool_texts: pd.Series,
        s1_ids: pd.Series,
        pool_ids: pd.Series,
        cand_s1_ids: List[str],
        cand_pool_ids: List[str],
        vec_type: str = "name_word",
    ) -> np.ndarray:
        """Compute pairwise cosine similarity for candidate pairs using sparse dot products."""
        vec: Optional[TfidfVectorizer] = None
        if vec_type == "name_word":
            vec = self.name_word_vec
        elif vec_type == "name_char":
            vec = self.name_char_vec
        elif vec_type == "addr_word":
            vec = self.addr_word_vec
        elif vec_type == "addr_char":
            vec = self.addr_char_vec

        if vec is None or len(cand_s1_ids) == 0:
            return np.zeros(len(cand_s1_ids), dtype=float)

        # Unique texts transform for referenced candidates
        cand_s1_set = set(cand_s1_ids)
        cand_pool_set = set(cand_pool_ids)

        s1_needed = s1_texts[s1_texts.index.isin(cand_s1_set)]
        s1_unique = s1_needed[~s1_needed.index.duplicated(keep="first")]

        pool_needed = pool_texts[pool_texts.index.isin(cand_pool_set)]
        pool_unique = pool_needed[~pool_needed.index.duplicated(keep="first")]

        mat_s1 = vec.transform(s1_unique.fillna(""))
        mat_pool = vec.transform(pool_unique.fillna(""))

        s1_idx_map = {eid: i for i, eid in enumerate(s1_unique.index)}
        pool_idx_map = {eid: i for i, eid in enumerate(pool_unique.index)}

        row_a = [s1_idx_map.get(sid, -1) for sid in cand_s1_ids]
        row_b = [pool_idx_map.get(cid, -1) for cid in cand_pool_ids]

        valid_mask = [ra >= 0 and rb >= 0 for ra, rb in zip(row_a, row_b)]
        out = np.zeros(len(cand_s1_ids), dtype=float)

        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        if valid_idx:
            batch_size = 50000
            for b_start in range(0, len(valid_idx), batch_size):
                b_idx = valid_idx[b_start : b_start + batch_size]
                sub_a = [row_a[i] for i in b_idx]
                sub_b = [row_b[i] for i in b_idx]
                # Since TF-IDF rows are unit normalized (L2=1), dot product is cosine similarity
                sims = np.asarray(mat_s1[sub_a].multiply(mat_pool[sub_b]).sum(axis=1)).ravel()
                out[b_idx] = sims

        return np.clip(out, 0.0, 1.0)


# ================================================================== #
#  Standalone TF-IDF bundle helpers
# ================================================================== #

def fit_tfidf_bundle(
    s1_train: pd.DataFrame,
    pool: pd.DataFrame,
    cfg: Config,
) -> TfidfBundle:
    """
    Fit TF-IDF bundle strictly on training Source-1 entities and candidate pool.
    Validation entities are NEVER passed to or seen by this function.
    """
    fcfg = cfg.features
    bundle = TfidfBundle(fcfg)

    s1_name = s1_train["business_name_normalized"] if "business_name_normalized" in s1_train.columns else s1_train.get("business_name", "")
    s1_addr = s1_train["business_address_normalized"] if "business_address_normalized" in s1_train.columns else s1_train.get("business_address", "")
    pool_name = pool["business_name_normalized"] if "business_name_normalized" in pool.columns else pool.get("business_name", "")
    pool_addr = pool["business_address_normalized"] if "business_address_normalized" in pool.columns else pool.get("business_address", "")

    name_train = pd.concat([s1_name, pool_name]).fillna("")
    addr_train = pd.concat([s1_addr, pool_addr]).fillna("")

    max_corpus = 500000
    if len(name_train) > max_corpus:
        sample_idx = name_train.sample(n=max_corpus, random_state=cfg.random_seed).index
        name_train = name_train.loc[sample_idx]
        addr_train = addr_train.loc[sample_idx]

    stats = {
        "s1_train_records": len(s1_train),
        "pool_records": len(pool),
        "fitted_corpus_records": len(name_train),
    }
    bundle.fit(name_train, addr_train, corpus_stats=stats)
    return bundle


def save_tfidf_bundle(bundle: TfidfBundle, path: Path) -> None:
    """Persist fitted TF-IDF bundle to disk."""
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved fitted TF-IDF bundle to %s", path)


def load_tfidf_bundle(path: Path) -> TfidfBundle:
    """Load fitted TF-IDF bundle from disk."""
    import joblib
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF bundle not found at {path}")
    bundle = joblib.load(path)
    return bundle


# ================================================================== #
#  Build the feature matrix
# ================================================================== #

def build_feature_matrix(
    candidates: pd.DataFrame,
    data: SourceData,
    cfg: Config,
    fit_tfidf: bool = False,
    tfidf_bundle: Optional[TfidfBundle] = None,
    pool: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute all enabled features for every candidate pair.

    Optimized with pre-cached token lookups and leak-free TF-IDF bundle.
    """
    fcfg = cfg.features
    n = len(candidates)
    logger.info("Computing features for %d candidate pairs …", n)

    if n == 0:
        feature_cols = get_feature_columns(cfg)
        empty_df = pd.DataFrame(columns=["source1_id", "candidate_id", "source"] + feature_cols)
        return empty_df

    # ---- Fast dictionary lookups for records -----------------------
    if pool is None:
        pool = pd.concat([data.source2, data.source3], ignore_index=True)

    cand_s1_list = list(candidates["source1_id"])
    cand_pool_list = list(candidates["candidate_id"])
    cand_src_list = list(candidates["source"])

    cand_s1_set = set(cand_s1_list)
    cand_pool_set = set(cand_pool_list)

    # Subset records to only those referenced in candidate pairs
    s1_sub = data.source1[data.source1["entity_id"].isin(cand_s1_set)]
    pool_sub = pool[pool["entity_id"].isin(cand_pool_set)]

    # Pre-build entity ID to attribute mappings
    s1_name_map = dict(zip(s1_sub["entity_id"], s1_sub["business_name_normalized"].fillna("")))
    s1_addr_map = dict(zip(s1_sub["entity_id"], s1_sub["business_address_normalized"].fillna("")))
    s1_country_map = dict(zip(s1_sub["entity_id"], s1_sub["country_normalized"].fillna("")))
    s1_postal_map = dict(zip(s1_sub["entity_id"], s1_sub["postal_code"].fillna("")))
    s1_house_map = dict(zip(s1_sub["entity_id"], s1_sub["house_number"].fillna("")))
    s1_numtok_map = {
        eid: str(val).split(",") if val else []
        for eid, val in zip(s1_sub["entity_id"], s1_sub["address_numeric_tokens"].fillna(""))
    }

    pool_name_map = dict(zip(pool_sub["entity_id"], pool_sub["business_name_normalized"].fillna("")))
    pool_addr_map = dict(zip(pool_sub["entity_id"], pool_sub["business_address_normalized"].fillna("")))
    pool_country_map = dict(zip(pool_sub["entity_id"], pool_sub["country_normalized"].fillna("")))
    pool_postal_map = dict(zip(pool_sub["entity_id"], pool_sub["postal_code"].fillna("")))
    pool_house_map = dict(zip(pool_sub["entity_id"], pool_sub["house_number"].fillna("")))
    pool_numtok_map = {
        eid: str(val).split(",") if val else []
        for eid, val in zip(pool_sub["entity_id"], pool_sub["address_numeric_tokens"].fillna(""))
    }

    # Pre-tokenize needed names and addresses
    s1_name_toks = {eid: set(tokenize(name)) for eid, name in s1_name_map.items()}
    s1_addr_toks = {eid: set(tokenize(addr)) for eid, addr in s1_addr_map.items()}
    pool_name_toks = {eid: set(tokenize(name)) for eid, name in pool_name_map.items()}
    pool_addr_toks = {eid: set(tokenize(addr)) for eid, addr in pool_addr_map.items()}

    # ---- TF-IDF handling (leak-free) ------------------------------
    import joblib
    tfidf_file = cfg.paths.models_dir / "tfidf_bundle.joblib"

    if tfidf_bundle is None:
        if fit_tfidf:
            tfidf_bundle = TfidfBundle(fcfg)
            name_train = pd.concat([data.source1["business_name_normalized"], pool["business_name_normalized"]])
            addr_train = pd.concat([data.source1["business_address_normalized"], pool["business_address_normalized"]])
            if len(name_train) > 500000:
                sample_idx = name_train.sample(n=500000, random_state=cfg.random_seed).index
                name_train = name_train.loc[sample_idx]
                addr_train = addr_train.loc[sample_idx]
            tfidf_bundle.fit(name_train, addr_train)
            cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(tfidf_bundle, tfidf_file)
            logger.info("Saved fitted TF-IDF bundle to %s", tfidf_file)
        elif tfidf_file.exists():
            try:
                tfidf_bundle = joblib.load(tfidf_file)
                logger.info("Loaded persisted TF-IDF bundle from %s", tfidf_file)
            except Exception as e:
                logger.warning("Could not load TF-IDF bundle from %s (%s). Re-fitting.", tfidf_file, e)
                tfidf_bundle = TfidfBundle(fcfg)
                name_train = pd.concat([data.source1["business_name_normalized"], pool["business_name_normalized"]])
                addr_train = pd.concat([data.source1["business_address_normalized"], pool["business_address_normalized"]])
                tfidf_bundle.fit(name_train, addr_train)
        else:
            tfidf_bundle = TfidfBundle(fcfg)
            name_train = pd.concat([data.source1["business_name_normalized"], pool["business_name_normalized"]])
            addr_train = pd.concat([data.source1["business_address_normalized"], pool["business_address_normalized"]])
            tfidf_bundle.fit(name_train, addr_train)

    # Compute TF-IDF similarities in batch if enabled
    s1_name_series = s1_sub.set_index("entity_id")["business_name_normalized"]
    pool_name_series = pool_sub.set_index("entity_id")["business_name_normalized"]
    s1_addr_series = s1_sub.set_index("entity_id")["business_address_normalized"]
    pool_addr_series = pool_sub.set_index("entity_id")["business_address_normalized"]

    sim_name_word = np.zeros(n, dtype=float)
    sim_name_char = np.zeros(n, dtype=float)
    sim_addr_word = np.zeros(n, dtype=float)
    sim_addr_char = np.zeros(n, dtype=float)

    if fcfg.name_tfidf_cosine and tfidf_bundle:
        sim_name_word = tfidf_bundle.compute_cosine(
            s1_name_series, pool_name_series, data.source1["entity_id"], pool["entity_id"],
            cand_s1_list, cand_pool_list, "name_word"
        )
        sim_name_char = tfidf_bundle.compute_cosine(
            s1_name_series, pool_name_series, data.source1["entity_id"], pool["entity_id"],
            cand_s1_list, cand_pool_list, "name_char"
        )

    if fcfg.address_tfidf_cosine and tfidf_bundle:
        sim_addr_word = tfidf_bundle.compute_cosine(
            s1_addr_series, pool_addr_series, data.source1["entity_id"], pool["entity_id"],
            cand_s1_list, cand_pool_list, "addr_word"
        )
        sim_addr_char = tfidf_bundle.compute_cosine(
            s1_addr_series, pool_addr_series, data.source1["entity_id"], pool["entity_id"],
            cand_s1_list, cand_pool_list, "addr_char"
        )

    # ---- Fast pre-cached feature loop -----------------------------
    feat_dict: Dict[str, list] = {col: [] for col in get_feature_columns(cfg)}

    for i in range(n):
        s1_id = cand_s1_list[i]
        c_id = cand_pool_list[i]
        src = cand_src_list[i]

        name_s1 = s1_name_map.get(s1_id, "")
        name_c = pool_name_map.get(c_id, "")
        addr_s1 = s1_addr_map.get(s1_id, "")
        addr_c = pool_addr_map.get(c_id, "")
        country_s1 = s1_country_map.get(s1_id, "")
        country_c = pool_country_map.get(c_id, "")
        postal_s1 = s1_postal_map.get(s1_id, "")
        postal_c = pool_postal_map.get(c_id, "")
        house_s1 = s1_house_map.get(s1_id, "")
        house_c = pool_house_map.get(c_id, "")
        numtok_s1 = s1_numtok_map.get(s1_id, [])
        numtok_c = pool_numtok_map.get(c_id, [])

        tok_s1 = s1_name_toks.get(s1_id, set())
        tok_c = pool_name_toks.get(c_id, set())
        atok_s1 = s1_addr_toks.get(s1_id, set())
        atok_c = pool_addr_toks.get(c_id, set())

        # Name features
        if fcfg.name_exact:
            feat_dict["name_exact"].append(name_exact_match(name_s1, name_c))
        if fcfg.name_fuzzy_ratio:
            feat_dict["name_fuzzy_ratio"].append(name_fuzzy_ratio(name_s1, name_c))
        if fcfg.name_partial_ratio:
            feat_dict["name_partial_ratio"].append(name_partial_ratio(name_s1, name_c))
        if fcfg.name_token_sort_ratio:
            feat_dict["name_token_sort_ratio"].append(name_token_sort_ratio(name_s1, name_c))
        if fcfg.name_token_set_ratio:
            feat_dict["name_token_set_ratio"].append(name_token_set_ratio(name_s1, name_c))
        if fcfg.name_jaccard:
            feat_dict["name_jaccard"].append(jaccard_similarity(tok_s1, tok_c))
        if fcfg.name_char_sim:
            feat_dict["name_char_sim"].append(char_similarity(name_s1, name_c))
        if fcfg.name_tfidf_cosine:
            feat_dict["name_tfidf_word"].append(float(sim_name_word[i]))
            feat_dict["name_tfidf_char"].append(float(sim_name_char[i]))
        if fcfg.name_length_diff:
            feat_dict["name_length_diff"].append(length_diff(name_s1, name_c))
        if fcfg.name_token_count_diff:
            feat_dict["name_token_count_diff"].append(abs(len(tok_s1) - len(tok_c)))
        if fcfg.name_common_tokens:
            feat_dict["name_common_tokens"].append(common_token_count(tok_s1, tok_c))

        # Address features
        if fcfg.address_exact:
            feat_dict["address_exact"].append(address_exact_match(addr_s1, addr_c))
        if fcfg.address_fuzzy:
            feat_dict["address_fuzzy_ratio"].append(address_fuzzy_ratio(addr_s1, addr_c))
        if fcfg.address_token_sim:
            feat_dict["address_token_sort_ratio"].append(address_token_sort_ratio(addr_s1, addr_c))
        if fcfg.address_jaccard:
            feat_dict["address_jaccard"].append(jaccard_similarity(atok_s1, atok_c))
        if fcfg.address_char_sim:
            feat_dict["address_char_sim"].append(char_similarity(addr_s1, addr_c))
        if fcfg.address_tfidf_cosine:
            feat_dict["address_tfidf_word"].append(float(sim_addr_word[i]))
            feat_dict["address_tfidf_char"].append(float(sim_addr_char[i]))
        if fcfg.address_length_diff:
            feat_dict["address_length_diff"].append(length_diff(addr_s1, addr_c))
        if fcfg.address_common_tokens:
            feat_dict["address_common_tokens"].append(common_token_count(atok_s1, atok_c))
        if fcfg.address_numeric_overlap:
            feat_dict["address_numeric_overlap"].append(numeric_token_overlap(numtok_s1, numtok_c))
        if fcfg.postal_code_match:
            feat_dict["postal_code_match"].append(postal_code_match(postal_s1, postal_c))
        if fcfg.house_number_match:
            feat_dict["house_number_match"].append(house_number_match(house_s1, house_c))

        # Country
        if fcfg.country_exact:
            feat_dict["country_exact"].append(country_exact_match(country_s1, country_c))

        # Source indicator
        if fcfg.source_indicator:
            feat_dict["source_is_s2"].append(source_is_s2(src))
            feat_dict["source_is_s3"].append(source_is_s3(src))

    feat_df = pd.DataFrame(feat_dict)
    result = pd.concat([candidates.reset_index(drop=True), feat_df], axis=1)

    logger.info("Feature matrix built: %d rows × %d feature columns", len(result), len(feat_df.columns))
    return result


def get_feature_columns(cfg: Config) -> list[str]:
    """Return the list of feature column names the model should use."""
    # Build a dummy and inspect — or hard-code the known names
    fcfg = cfg.features
    cols = []

    if fcfg.name_exact:            cols.append("name_exact")
    if fcfg.name_fuzzy_ratio:      cols.append("name_fuzzy_ratio")
    if fcfg.name_partial_ratio:    cols.append("name_partial_ratio")
    if fcfg.name_token_sort_ratio: cols.append("name_token_sort_ratio")
    if fcfg.name_token_set_ratio:  cols.append("name_token_set_ratio")
    if fcfg.name_jaccard:          cols.append("name_jaccard")
    if fcfg.name_char_sim:         cols.append("name_char_sim")
    if fcfg.name_tfidf_cosine:
        cols.append("name_tfidf_word")
        cols.append("name_tfidf_char")
    if fcfg.name_length_diff:      cols.append("name_length_diff")
    if fcfg.name_token_count_diff: cols.append("name_token_count_diff")
    if fcfg.name_common_tokens:    cols.append("name_common_tokens")

    if fcfg.address_exact:         cols.append("address_exact")
    if fcfg.address_fuzzy:         cols.append("address_fuzzy_ratio")
    if fcfg.address_token_sim:     cols.append("address_token_sort_ratio")
    if fcfg.address_jaccard:       cols.append("address_jaccard")
    if fcfg.address_char_sim:      cols.append("address_char_sim")
    if fcfg.address_tfidf_cosine:
        cols.append("address_tfidf_word")
        cols.append("address_tfidf_char")
    if fcfg.address_length_diff:   cols.append("address_length_diff")
    if fcfg.address_common_tokens: cols.append("address_common_tokens")
    if fcfg.address_numeric_overlap: cols.append("address_numeric_overlap")
    if fcfg.postal_code_match:     cols.append("postal_code_match")
    if fcfg.house_number_match:    cols.append("house_number_match")

    if fcfg.country_exact:         cols.append("country_exact")
    if fcfg.source_indicator:
        cols.append("source_is_s2")
        cols.append("source_is_s3")

    return cols
