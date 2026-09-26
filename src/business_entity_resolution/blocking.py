"""
blocking.py — Multiple blocking strategies
============================================
Each strategy returns a ``dict[str, set[str]]`` mapping a
*blocking key* to the set of entity-IDs that share that key.

The candidate generator (``candidate_generation.py``) uses these
inverted indices to build candidate pairs efficiently.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Set

import pandas as pd

from .config import BlockingConfig
from .preprocessing import tokenize, char_ngrams

logger = logging.getLogger(__name__)

# Type alias: blocking_key → {entity_id, …}
BlockIndex = Dict[str, Set[str]]


# ================================================================== #
#  Individual blocking strategies
# ================================================================== #

def country_block(df: pd.DataFrame) -> BlockIndex:
    """Group entities by normalised country."""
    idx: BlockIndex = defaultdict(set)
    for eid, country in zip(df["entity_id"], df["country_normalized"]):
        if country:
            idx[f"country:{country}"].add(eid)
    return dict(idx)


def name_token_block(df: pd.DataFrame) -> BlockIndex:
    """
    Inverted index on individual name tokens.

    Every entity contributes its name tokens as blocking keys.
    Two entities sharing *any* token will be candidate pairs.
    """
    idx: BlockIndex = defaultdict(set)
    for eid, name in zip(df["entity_id"], df["business_name_normalized"]):
        for tok in tokenize(name):
            if len(tok) >= 2:  # skip single-char tokens
                idx[f"ntok:{tok}"].add(eid)
    return dict(idx)


def name_prefix_block(df: pd.DataFrame, prefix_len: int = 4) -> BlockIndex:
    """Block on the first *prefix_len* characters of the normalised name."""
    idx: BlockIndex = defaultdict(set)
    for eid, name in zip(df["entity_id"], df["business_name_normalized"]):
        if len(name) >= prefix_len:
            idx[f"npfx:{name[:prefix_len]}"].add(eid)
    return dict(idx)


def address_token_block(df: pd.DataFrame) -> BlockIndex:
    """Inverted index on address tokens (excluding very common words)."""
    # Stopwords that are too frequent to be useful for blocking
ADDRESS_STOPWORDS = {
    "road", "street", "avenue", "floor", "building", "suite",
    "near", "opposite", "lane", "drive", "highway", "no",
    "block", "plot", "sector", "phase", "unit", "tower",
}


# ================================================================== #
#  Individual blocking strategies
# ================================================================== #

def country_block(df: pd.DataFrame) -> BlockIndex:
    """Group entities by normalised country (broad; should be used with caution)."""
    idx: BlockIndex = defaultdict(set)
    for eid, country in zip(df["entity_id"], df["country_normalized"]):
        if country:
            idx[f"country:{country}"].add(eid)
    return dict(idx)


def name_token_block(df: pd.DataFrame) -> BlockIndex:
    """
    Inverted index on individual name tokens.
    Every entity contributes its name tokens as blocking keys.
    """
    idx: BlockIndex = defaultdict(set)
    for eid, name in zip(df["entity_id"], df["business_name_normalized"]):
        for tok in tokenize(name):
            if len(tok) >= 2:  # skip single-char tokens
                idx[f"ntok:{tok}"].add(eid)
    return dict(idx)


def name_prefix_block(df: pd.DataFrame, prefix_len: int = 4) -> BlockIndex:
    """Block on the first *prefix_len* characters of the normalised name."""
    idx: BlockIndex = defaultdict(set)
    for eid, name in zip(df["entity_id"], df["business_name_normalized"]):
        if len(name) >= prefix_len:
            idx[f"npfx:{name[:prefix_len]}"].add(eid)
    return dict(idx)


def address_token_block(df: pd.DataFrame) -> BlockIndex:
    """Inverted index on address tokens (excluding very common words)."""
    idx: BlockIndex = defaultdict(set)
    for eid, addr in zip(df["entity_id"], df["business_address_normalized"]):
        for tok in tokenize(addr):
            if len(tok) >= 2 and tok not in ADDRESS_STOPWORDS:
                idx[f"atok:{tok}"].add(eid)
    return dict(idx)


def postal_code_block(df: pd.DataFrame) -> BlockIndex:
    """Block on extracted postal / ZIP / PIN code."""
    idx: BlockIndex = defaultdict(set)
    for eid, pc in zip(df["entity_id"], df["postal_code"]):
        if pc:
            idx[f"post:{pc}"].add(eid)
    return dict(idx)


def char_ngram_block(
    df: pd.DataFrame,
    n: int = 3,
) -> BlockIndex:
    """Block on character n-grams of the normalised name."""
    idx: BlockIndex = defaultdict(set)
    for eid, name in zip(df["entity_id"], df["business_name_normalized"]):
        for ng in char_ngrams(name, n):
            idx[f"cng:{ng}"].add(eid)
    return dict(idx)


# ================================================================== #
#  Compound multi-channel blocking strategies
# ================================================================== #

def country_name_token_block(df: pd.DataFrame) -> BlockIndex:
    """Compound block on (country + name token). Highly informative."""
    idx: BlockIndex = defaultdict(set)
    for eid, country, name in zip(df["entity_id"], df["country_normalized"], df["business_name_normalized"]):
        if country and name:
            for tok in tokenize(name):
                if len(tok) >= 2:
                    idx[f"c_ntok:{country}::{tok}"].add(eid)
    return dict(idx)


def country_name_prefix_block(df: pd.DataFrame, prefix_len: int = 4) -> BlockIndex:
    """Compound block on (country + name prefix)."""
    idx: BlockIndex = defaultdict(set)
    for eid, country, name in zip(df["entity_id"], df["country_normalized"], df["business_name_normalized"]):
        if country and len(name) >= prefix_len:
            idx[f"c_npfx:{country}::{name[:prefix_len]}"].add(eid)
    return dict(idx)


def country_postal_block(df: pd.DataFrame) -> BlockIndex:
    """Compound block on (country + postal code)."""
    idx: BlockIndex = defaultdict(set)
    for eid, country, pc in zip(df["entity_id"], df["country_normalized"], df["postal_code"]):
        if country and pc:
            idx[f"c_post:{country}::{pc}"].add(eid)
    return dict(idx)


def country_address_token_block(df: pd.DataFrame) -> BlockIndex:
    """Compound block on (country + address token)."""
    idx: BlockIndex = defaultdict(set)
    for eid, country, addr in zip(df["entity_id"], df["country_normalized"], df["business_address_normalized"]):
        if country and addr:
            for tok in tokenize(addr):
                if len(tok) >= 2 and tok not in ADDRESS_STOPWORDS:
                    idx[f"c_atok:{country}::{tok}"].add(eid)
    return dict(idx)


def name_token_postal_block(df: pd.DataFrame) -> BlockIndex:
    """Compound block on (name token + postal code)."""
    idx: BlockIndex = defaultdict(set)
    for eid, name, pc in zip(df["entity_id"], df["business_name_normalized"], df["postal_code"]):
        if pc and name:
            for tok in tokenize(name):
                if len(tok) >= 2:
                    idx[f"ntok_post:{tok}::{pc}"].add(eid)
    return dict(idx)


def name_prefix_postal_block(df: pd.DataFrame, prefix_len: int = 4) -> BlockIndex:
    """Compound block on (name prefix + postal code)."""
    idx: BlockIndex = defaultdict(set)
    for eid, name, pc in zip(df["entity_id"], df["business_name_normalized"], df["postal_code"]):
        if pc and len(name) >= prefix_len:
            idx[f"npfx_post:{name[:prefix_len]}::{pc}"].add(eid)
    return dict(idx)


def name_token_address_token_block(df: pd.DataFrame) -> BlockIndex:
    """Compound block on (name token + address token)."""
    idx: BlockIndex = defaultdict(set)
    for eid, name, addr in zip(df["entity_id"], df["business_name_normalized"], df["business_address_normalized"]):
        if name and addr:
            ntoks = [t for t in tokenize(name) if len(t) >= 3][:3]
            atoks = [t for t in tokenize(addr) if len(t) >= 3 and t not in ADDRESS_STOPWORDS][:3]
            for nt in ntoks:
                for at in atoks:
                    idx[f"ntok_atok:{nt}::{at}"].add(eid)
    return dict(idx)


# ================================================================== #
#  Build combined block index for a pool (S2+S3)
# ================================================================== #

def _prune_block_index(idx: BlockIndex, max_size: int, name: str) -> BlockIndex:
    if max_size <= 0:
        return idx
    pruned = {k: v for k, v in idx.items() if len(v) <= max_size}
    dropped = len(idx) - len(pruned)
    if dropped > 0:
        logger.info("    [%s] pruned %d large blocks (size > %d)", name, dropped, max_size)
    return pruned


def build_block_indices(
    pool: pd.DataFrame,
    bcfg: BlockingConfig,
) -> Dict[str, BlockIndex]:
    """
    Build all configured blocking indices for the candidate *pool*.

    Returns
    -------
    dict[strategy_name, BlockIndex]
    """
    indices: Dict[str, BlockIndex] = {}
    max_bs = getattr(bcfg, "max_block_size", 500)

    if bcfg.country_block:
        indices["country"] = _prune_block_index(country_block(pool), max_bs, "country")
        logger.info("  country_block                 → %d keys", len(indices["country"]))

    if bcfg.name_token_block:
        indices["name_token"] = _prune_block_index(name_token_block(pool), max_bs, "name_token")
        logger.info("  name_token_block              → %d keys", len(indices["name_token"]))

    if bcfg.name_prefix_block:
        indices["name_prefix"] = _prune_block_index(name_prefix_block(pool, bcfg.name_prefix_length), max_bs, "name_prefix")
        logger.info("  name_prefix_block             → %d keys", len(indices["name_prefix"]))

    if bcfg.address_token_block:
        indices["address_token"] = _prune_block_index(address_token_block(pool), max_bs, "address_token")
        logger.info("  address_token_block           → %d keys", len(indices["address_token"]))

    if bcfg.postal_code_block:
        indices["postal_code"] = _prune_block_index(postal_code_block(pool), max_bs, "postal_code")
        logger.info("  postal_code_block             → %d keys", len(indices["postal_code"]))

    if bcfg.char_ngram_block:
        indices["char_ngram"] = _prune_block_index(char_ngram_block(pool, bcfg.char_ngram_n), max_bs, "char_ngram")
        logger.info("  char_ngram_block              → %d keys", len(indices["char_ngram"]))

    if bcfg.country_name_token_block:
        indices["country_name_token"] = _prune_block_index(country_name_token_block(pool), max_bs, "country_name_token")
        logger.info("  country_name_token_block      → %d keys", len(indices["country_name_token"]))

    if bcfg.country_name_prefix_block:
        indices["country_name_prefix"] = _prune_block_index(country_name_prefix_block(pool, bcfg.name_prefix_length), max_bs, "country_name_prefix")
        logger.info("  country_name_prefix_block     → %d keys", len(indices["country_name_prefix"]))

    if bcfg.country_postal_block:
        indices["country_postal"] = _prune_block_index(country_postal_block(pool), max_bs, "country_postal")
        logger.info("  country_postal_block          → %d keys", len(indices["country_postal"]))

    if bcfg.country_address_token_block:
        indices["country_address_token"] = _prune_block_index(country_address_token_block(pool), max_bs, "country_address_token")
        logger.info("  country_address_token_block   → %d keys", len(indices["country_address_token"]))

    if bcfg.name_token_postal_block:
        indices["name_token_postal"] = _prune_block_index(name_token_postal_block(pool), max_bs, "name_token_postal")
        logger.info("  name_token_postal_block       → %d keys", len(indices["name_token_postal"]))

    if bcfg.name_prefix_postal_block:
        indices["name_prefix_postal"] = _prune_block_index(name_prefix_postal_block(pool, bcfg.name_prefix_length), max_bs, "name_prefix_postal")
        logger.info("  name_prefix_postal_block      → %d keys", len(indices["name_prefix_postal"]))

    if bcfg.name_token_address_token_block:
        indices["name_token_address_token"] = _prune_block_index(name_token_address_token_block(pool), max_bs, "name_token_address_token")
        logger.info("  name_token_address_token_block → %d keys", len(indices["name_token_address_token"]))

    return indices
