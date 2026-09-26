"""
tests/test_core.py — Unit tests for core pipeline components
==============================================================
Uses synthetic mini-data so no challenge dataset is required.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from business_entity_resolution.preprocessing import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_postal_code,
    extract_house_number,
    extract_numeric_tokens,
    tokenize,
    char_ngrams,
)
from business_entity_resolution.ground_truth import parse_ground_truth
from business_entity_resolution.evaluate import entity_level_f05, f_beta

import pandas as pd


# ================================================================== #
#  Normalization tests
# ================================================================== #

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "limited",
    "llc", "inc.", "inc", "corp.", "corp", "ltd.", "ltd",
]


class TestNormalizeName:
    def test_lowercase(self):
        assert normalize_name("ACME Corp", LEGAL_SUFFIXES) == "acme"

    def test_ampersand(self):
        assert normalize_name("Ben & Jerry", []) == "ben and jerry"

    def test_legal_suffix(self):
        result = normalize_name("Tata Motors Ltd.", LEGAL_SUFFIXES)
        assert "ltd" not in result
        assert "tata" in result and "motors" in result

    def test_empty(self):
        assert normalize_name("", []) == ""

    def test_unicode(self):
        result = normalize_name("Café Résumé", [])
        assert "cafe" in result


class TestNormalizeAddress:
    def test_abbreviation(self):
        abbrevs = {"st": "street", "rd": "road"}
        result = normalize_address("123 Main St", abbrevs)
        assert "street" in result

    def test_empty(self):
        assert normalize_address("", {}) == ""


class TestNormalizeCountry:
    def test_lowercase(self):
        assert normalize_country("United States") == "united states"

    def test_open_set(self):
        # Must not filter unknown countries
        assert normalize_country("France") == "france"
        assert normalize_country("Brazil") == "brazil"

    def test_empty(self):
        assert normalize_country("") == ""


class TestAddressExtraction:
    def test_us_zip(self):
        assert extract_postal_code("123 Main St, Springfield, IL 62704") == "62704"

    def test_india_pin(self):
        assert extract_postal_code("MG Road, Bangalore 560001") == "560001"

    def test_no_code(self):
        assert extract_postal_code("Some random address") == ""

    def test_house_number(self):
        assert extract_house_number("42B Elm Street") == "42B"

    def test_numeric_tokens(self):
        nums = extract_numeric_tokens("Floor 3, Unit 204, 500 5th Ave")
        assert "3" in nums and "204" in nums and "500" in nums


class TestTokenization:
    def test_basic(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []

    def test_char_ngrams(self):
        ng = char_ngrams("abcd", 3)
        assert ng == {"abc", "bcd"}


# ================================================================== #
#  Ground truth parsing
# ================================================================== #

class TestGroundTruth:
    def test_parse(self):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
            "matched_entity_ids": ["S2-010,S3-020", "S3-005", ""],
        })
        gt = parse_ground_truth(df)
        assert gt["S1-001"] == {"S2-010", "S3-020"}
        assert gt["S1-002"] == {"S3-005"}
        assert gt["S1-003"] == set()

    def test_nan_handling(self):
        df = pd.DataFrame({
            "source1_entity_id": ["S1-001"],
            "matched_entity_ids": ["nan"],
        })
        gt = parse_ground_truth(df)
        assert gt["S1-001"] == set()


# ================================================================== #
#  F0.5 evaluation
# ================================================================== #

class TestF05:
    def test_perfect(self):
        pred = {"S1-001": {"S2-010"}}
        gt = {"S1-001": {"S2-010"}}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 1.0

    def test_singleton_correct(self):
        pred = {"S1-001": set()}
        gt = {"S1-001": set()}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 1.0

    def test_singleton_false_positive(self):
        pred = {"S1-001": {"S2-010"}}
        gt = {"S1-001": set()}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 0.0

    def test_complete_miss(self):
        pred = {"S1-001": set()}
        gt = {"S1-001": {"S2-010"}}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 0.0

    def test_partial(self):
        pred = {"S1-001": {"S2-010", "S2-020"}}
        gt = {"S1-001": {"S2-010"}}
        m = entity_level_f05(pred, gt)
        # precision = 1/2, recall = 1/1
        expected = f_beta(0.5, 1.0, 0.5)
        assert abs(m["macro_f05"] - expected) < 1e-6

    def test_macro_average(self):
        pred = {"S1-001": {"S2-010"}, "S1-002": set()}
        gt = {"S1-001": {"S2-010"}, "S1-002": set()}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 1.0

    def test_f_beta_zero_inputs(self):
        assert f_beta(0, 0, 0.5) == 0.0
