"""
features.py — Individual feature computation functions
=======================================================
Each function takes two strings (or pre-computed values) and
returns a numeric feature value.  They are intentionally *pure*
functions with no side effects, making them easy to test and swap.
"""

from __future__ import annotations

from typing import List, Set

from rapidfuzz import fuzz

from .preprocessing import tokenize


# ================================================================== #
#  Name similarity features
# ================================================================== #

def name_exact_match(a: str, b: str) -> int:
    """1 if normalised names are identical, else 0."""
    return int(a == b and a != "")


def name_fuzzy_ratio(a: str, b: str) -> float:
    """RapidFuzz simple ratio (0–100)."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b)


def name_partial_ratio(a: str, b: str) -> float:
    """RapidFuzz partial ratio (0–100)."""
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b)


def name_token_sort_ratio(a: str, b: str) -> float:
    """RapidFuzz token sort ratio (0–100)."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b)


def name_token_set_ratio(a: str, b: str) -> float:
    """RapidFuzz token set ratio (0–100)."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b)


def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Jaccard index between two token sets."""
    if not tokens_a and not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union > 0 else 0.0


def common_token_count(tokens_a: Set[str], tokens_b: Set[str]) -> int:
    """Number of shared tokens."""
    return len(tokens_a & tokens_b)


def token_count_diff(a: str, b: str) -> int:
    """Absolute difference in token count."""
    return abs(len(tokenize(a)) - len(tokenize(b)))


def length_diff(a: str, b: str) -> int:
    """Absolute character-length difference."""
    return abs(len(a) - len(b))


def char_similarity(a: str, b: str) -> float:
    """
    Character-level Jaccard similarity.
    Useful for catching typos and transliterations.
    """
    if not a and not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


# ================================================================== #
#  Address similarity features
# ================================================================== #

def address_exact_match(a: str, b: str) -> int:
    """1 if normalised addresses are identical, else 0."""
    return int(a == b and a != "")


def address_fuzzy_ratio(a: str, b: str) -> float:
    """RapidFuzz ratio on addresses."""
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b)


def address_token_sort_ratio(a: str, b: str) -> float:
    """RapidFuzz token sort ratio on addresses."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b)


def numeric_token_overlap(nums_a: List[str], nums_b: List[str]) -> float:
    """
    Fraction of numeric tokens in *a* that also appear in *b*.
    Captures house numbers, postal codes, floor numbers, etc.
    """
    if not nums_a:
        return 0.0
    set_b = set(nums_b)
    return sum(1 for n in nums_a if n in set_b) / len(nums_a)


def postal_code_match(pc_a: str, pc_b: str) -> int:
    """1 if both postal codes are non-empty and equal."""
    return int(bool(pc_a) and pc_a == pc_b)


def house_number_match(hn_a: str, hn_b: str) -> int:
    """1 if both house numbers are non-empty and equal."""
    return int(bool(hn_a) and hn_a == hn_b)


# ================================================================== #
#  Country feature
# ================================================================== #

def country_exact_match(a: str, b: str) -> int:
    """1 if normalised country strings are equal."""
    return int(a == b and a != "")


# ================================================================== #
#  Source indicator
# ================================================================== #

def source_is_s2(source: str) -> int:
    """1 if the candidate comes from Source 2."""
    return int(source == "S2")


def source_is_s3(source: str) -> int:
    """1 if the candidate comes from Source 3."""
    return int(source == "S3")
