"""
preprocessing.py — Text normalization & address info extraction
================================================================
Creates ``*_normalized`` columns alongside originals so raw data
is always available for debugging.

All normalization rules are configurable via ``config.yaml``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import Config, PreprocessingConfig
from .data_loader import SourceData

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Regex patterns compiled once
# ------------------------------------------------------------------ #
_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_RE_ADDR_PUNCT = re.compile(r"[^\w\s\-]")
_RE_POSTAL_CODE = re.compile(
    r"\b("
    r"\d{5}(?:-\d{4})?"   # US ZIP / ZIP+4
    r"|"
    r"\d{6}"               # India PIN
    r"|"
    r"\d{5}"               # France code postal
    r"|"
    r"[A-Z]{1,2}\d[\dA-Z]?\s*\d[A-Z]{2}"  # UK postcode (kept generic)
    r")\b",
    re.IGNORECASE,
)
_RE_HOUSE_NUMBER = re.compile(r"^(\d+[A-Za-z]?)\b")
_RE_NUMERIC_TOKENS = re.compile(r"\b\d+\b")

# Cache compiled regex for legal suffixes
_COMPILED_SUFFIX_REGEX: Dict[tuple, re.Pattern] = {}

def _get_suffix_pattern(legal_suffixes: List[str]) -> Optional[re.Pattern]:
    key = tuple(legal_suffixes)
    if key not in _COMPILED_SUFFIX_REGEX:
        if not legal_suffixes:
            return None
        sorted_suffixes = sorted(legal_suffixes, key=len, reverse=True)
        pat = r"[,.\s]*\b(?:" + "|".join(re.escape(s) for s in sorted_suffixes) + r")\.?\s*$"
        _COMPILED_SUFFIX_REGEX[key] = re.compile(pat, re.IGNORECASE)
    return _COMPILED_SUFFIX_REGEX[key]


# ================================================================== #
#  Core normalization functions
# ================================================================== #

def normalize_name(
    name: str,
    legal_suffixes: List[str],
) -> str:
    """
    Normalize a business name.

    Steps
    -----
    1. Unicode NFKD normalization (if non-ASCII).
    2. Lowercase & & -> and.
    3. Remove legal suffixes (single-pass compiled regex).
    4. Strip punctuation.
    5. Collapse whitespace.
    """
    if not name or pd.isna(name):
        return ""

    text = str(name)
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = text.replace("&", " and ")

    pattern = _get_suffix_pattern(legal_suffixes)
    if pattern:
        text = pattern.sub("", text)

    text = _RE_PUNCTUATION.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def normalize_address(
    address: str,
    address_abbreviations: dict[str, str],
) -> str:
    """
    Normalize a business address.
    """
    if not address or pd.isna(address):
        return ""

    text = str(address)
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = _RE_ADDR_PUNCT.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    if address_abbreviations:
        tokens = text.split()
        text = " ".join(address_abbreviations.get(t, t) for t in tokens)

    return text


def normalize_country(country: str) -> str:
    """
    Normalize a country string. Open-set; no whitelist.
    """
    if not country or pd.isna(country):
        return ""
    text = str(country)
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = _RE_WHITESPACE.sub(" ", text)
    return text


# ================================================================== #
#  Address information extraction
# ================================================================== #

def extract_postal_code(address: str) -> str:
    """Extract postal / ZIP / PIN code from an address string."""
    if not address:
        return ""
    m = _RE_POSTAL_CODE.search(address)
    return m.group(1).strip() if m else ""


def extract_house_number(address: str) -> str:
    """Extract leading house/building number."""
    if not address:
        return ""
    m = _RE_HOUSE_NUMBER.match(address.strip())
    return m.group(1) if m else ""


def extract_numeric_tokens(address: str) -> List[str]:
    """Return all numeric tokens in an address."""
    if not address:
        return []
    return _RE_NUMERIC_TOKENS.findall(address)


# ================================================================== #
#  Tokenization utilities
# ================================================================== #

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer on already-normalized text."""
    if not text:
        return []
    return text.split()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Return the set of character n-grams for *text*."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


# ================================================================== #
#  DataFrame-level preprocessing
# ================================================================== #

def preprocess_df(
    df: pd.DataFrame,
    cfg_prep: Any,
    label: str = "",
) -> pd.DataFrame:
    """
    Add ``*_normalized`` columns and extracted address info to *df*.

    The original columns are never modified.
    """
    if hasattr(cfg_prep, "preprocessing"):
        cfg_prep = cfg_prep.preprocessing

    logger.info("Preprocessing %s (%d rows)", label, len(df))

    suffixes = sorted(cfg_prep.legal_suffixes, key=len, reverse=True)
    abbrevs = cfg_prep.address_abbreviations

    df = df.copy()

    # Normalized columns
    df["business_name_normalized"] = df["business_name"].apply(
        lambda x: normalize_name(x, suffixes)
    )
    df["business_address_normalized"] = df["business_address"].apply(
        lambda x: normalize_address(x, abbrevs)
    )
    df["country_normalized"] = df["country"].apply(normalize_country)

    # Extracted address information
    df["postal_code"] = df["business_address"].apply(
        lambda x: extract_postal_code(str(x) if not pd.isna(x) else "")
    )
    df["house_number"] = df["business_address_normalized"].apply(extract_house_number)
    df["address_numeric_tokens"] = df["business_address_normalized"].apply(
        lambda x: ",".join(extract_numeric_tokens(x))
    )

    return df


def preprocess_sources(data: SourceData, cfg: Config) -> SourceData:
    """
    Apply preprocessing to all three source DataFrames in *data*.

    Returns a new ``SourceData`` with the preprocessed DataFrames.
    """
    return SourceData(
        source1=preprocess_df(data.source1, cfg.preprocessing, "Source1"),
        source2=preprocess_df(data.source2, cfg.preprocessing, "Source2"),
        source3=preprocess_df(data.source3, cfg.preprocessing, "Source3"),
        ground_truth=data.ground_truth,
    )
