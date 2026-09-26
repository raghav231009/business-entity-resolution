"""
test_metrics.py — Tests for metrics.py
======================================
Tests verify the exact challenge F0.5 metric, singleton edge cases,
and macro-level entity aggregation according to specifications.
"""

import pytest

from business_entity_resolution.metrics import (
    precision_score,
    recall_score,
    f_beta_score,
    entity_f05,
    macro_entity_f05,
    f_beta,
    entity_level_f05,
)


class TestBasicMetrics:
    def test_precision(self):
        assert precision_score(2, 0) == 1.0
        assert precision_score(1, 1) == 0.5
        assert precision_score(0, 5) == 0.0
        assert precision_score(0, 0) == 0.0

    def test_recall(self):
        assert recall_score(2, 0) == 1.0
        assert recall_score(1, 1) == 0.5
        assert recall_score(0, 5) == 0.0
        assert recall_score(0, 0) == 0.0

    def test_f_beta_score(self):
        # When p=1.0 and r=1.0, F0.5 = 1.0
        assert pytest.approx(f_beta_score(1.0, 1.0, beta=0.5)) == 1.0
        # When p=0 or r=0, F0.5 = 0.0
        assert f_beta_score(0.0, 1.0, beta=0.5) == 0.0
        assert f_beta_score(1.0, 0.0, beta=0.5) == 0.0
        # Precision=1.0, Recall=0.5
        # (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 ≈ 0.8333
        assert pytest.approx(f_beta_score(1.0, 0.5, beta=0.5), rel=1e-4) == 5 / 6
        # Alias test
        assert f_beta(1.0, 1.0) == f_beta_score(1.0, 1.0)


class TestEntityF05:
    def test_perfect_match(self):
        """truth: {A}, prediction: {A} -> F0.5 = 1.0"""
        res = entity_f05({"A"}, {"A"})
        assert res["f05"] == 1.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["tp"] == 1
        assert res["fp"] == 0
        assert res["fn"] == 0

    def test_completely_wrong(self):
        """truth: {A}, prediction: {B} -> F0.5 = 0.0"""
        res = entity_f05({"A"}, {"B"})
        assert res["f05"] == 0.0
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["tp"] == 0
        assert res["fp"] == 1
        assert res["fn"] == 1

    def test_partial_recall(self):
        """truth: {A, B}, prediction: {A} -> Expected F0.5 < 1.0"""
        res = entity_f05({"A", "B"}, {"A"})
        assert res["precision"] == 1.0
        assert res["recall"] == 0.5
        assert res["f05"] < 1.0
        assert pytest.approx(res["f05"], rel=1e-4) == 5 / 6

    def test_singleton_correctly_predicted(self):
        """truth: {}, prediction: {} -> F0.5 = 1.0"""
        res = entity_f05(set(), set())
        assert res["f05"] == 1.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["true_count"] == 0
        assert res["pred_count"] == 0

    def test_singleton_false_positive(self):
        """truth: {}, prediction: {A} -> F0.5 = 0.0"""
        res = entity_f05(set(), {"A"})
        assert res["f05"] == 0.0
        assert res["precision"] == 0.0
        assert res["true_count"] == 0
        assert res["pred_count"] == 1


class TestMacroEntityF05:
    def test_macro_evaluation(self):
        """Include multiple Source 1 entities and verify mean is calculated at entity level."""
        gt = {
            "S1_1": {"S2_1"},       # Perfect match -> F0.5 = 1.0
            "S1_2": {"S2_2"},       # Completely wrong -> F0.5 = 0.0
            "S1_3": set(),          # Singleton correct -> F0.5 = 1.0
            "S1_4": set(),          # Singleton false positive -> F0.5 = 0.0
        }
        pred = {
            "S1_1": {"S2_1"},
            "S1_2": {"S2_999"},
            "S1_3": set(),
            "S1_4": {"S2_3"},
        }
        res = macro_entity_f05(pred, gt)
        assert res["n_entities"] == 4
        # Macro F0.5 = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.50
        assert pytest.approx(res["macro_f05"]) == 0.50
        # Singleton accuracy = 1 correct out of 2 singletons = 0.50
        assert pytest.approx(res["singleton_accuracy"]) == 0.50

    def test_alias_equivalence(self):
        gt = {"E1": {"M1"}}
        pred = {"E1": {"M1"}}
        res1 = macro_entity_f05(pred, gt)
        res2 = entity_level_f05(pred, gt)
        assert res1["macro_f05"] == res2["macro_f05"]
