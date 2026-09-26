"""
tests/test_fixes.py — Tests for critical bug fixes and challenge requirements
=============================================================================
Covers:
1. Ground-truth evaluation: missed match in blocking counts against recall.
2. Threshold persistence and fallback hierarchy.
3. Candidate pre-ranking: no arbitrary alphabetical ID truncation.
4. Candidate recall measurement and metrics.
5. Strict output validation: invalid IDs, duplicate IDs, extra/missing S1, self-matches, subset check.
6. Open-set country handling (France).
7. Singleton handling in evaluation.
8. Leak-free TF-IDF bundle fitting and inference.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from business_entity_resolution.config import Config, BlockingConfig, FeaturesConfig
from business_entity_resolution.data_loader import SourceData
from business_entity_resolution.evaluate import entity_level_f05, evaluate_from_predictions
from business_entity_resolution.threshold_tuning import tune_threshold
from business_entity_resolution.postprocessing import resolve_threshold, postprocess
from business_entity_resolution.candidate_generation import generate_candidates, _report_candidate_recall
from business_entity_resolution.feature_builder import TfidfBundle, build_feature_matrix
from business_entity_resolution.preprocessing import normalize_country
from business_entity_resolution.validation import validate_outputs


# ================================================================== #
#  1. Ground-truth evaluation against full ground truth
# ================================================================== #

class TestGroundTruthFullEvaluation:
    def test_missed_match_by_blocking_penalized(self, tmp_path):
        """
        True match {B, C} where blocking only found B.
        Predicting B should give recall = 1/2 = 0.5, NOT 1.0.
        """
        # Full ground truth
        full_gt = {"S1-001": {"S2-B", "S3-C"}}

        # Candidate set only contained S2-B; matcher predicted S2-B
        val_df = pd.DataFrame([{
            "source1_id": "S1-001",
            "candidate_id": "S2-B",
            "label": 1,
            "probability": 0.90,
        }])

        cfg = Config()
        cfg.paths.metrics_dir = tmp_path / "metrics"
        metrics = evaluate_from_predictions(val_df, threshold=0.50, cfg=cfg, ground_truth_map=full_gt)

        # Macro recall must be 0.5, and F0.5 must be < 1.0
        assert metrics["macro_recall"] == pytest.approx(0.5, abs=1e-4)
        assert metrics["macro_precision"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["macro_f05"] < 1.0
        # Specifically: (1 + 0.25) * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 1.25 * 0.5 / 0.75 = 0.8333
        assert metrics["macro_f05"] == pytest.approx(0.8333, abs=1e-3)

    def test_entity_with_zero_candidates_in_validation(self, tmp_path):
        """
        An S1 entity in validation ground truth with a true match,
        but blocking generated 0 candidate pairs. Recall should be 0.0.
        """
        full_gt = {
            "S1-001": {"S2-A"},
            "S1-002": {"S2-B"},  # 0 candidates generated
        }

        val_df = pd.DataFrame([{
            "source1_id": "S1-001",
            "candidate_id": "S2-A",
            "label": 1,
            "probability": 0.90,
        }])

        cfg = Config()
        cfg.paths.metrics_dir = tmp_path / "metrics"
        metrics = evaluate_from_predictions(val_df, threshold=0.50, cfg=cfg, ground_truth_map=full_gt)

        # S1-001 has F0.5 = 1.0, S1-002 has F0.5 = 0.0
        assert metrics["macro_f05"] == pytest.approx(0.5, abs=1e-4)
        assert metrics["n_entities"] == 2


# ================================================================== #
#  2. Threshold persistence and resolution
# ================================================================== #

class TestThresholdPersistence:
    def test_persistence_and_resolution(self, tmp_path):
        cfg = Config()
        cfg.paths.models_dir = tmp_path / "models"
        cfg.paths.models_dir.mkdir(parents=True)
        cfg.paths.metrics_dir = tmp_path / "metrics"
        cfg.threshold.default = 0.50

        # Save synthetic val predictions and full ground truth
        val_df = pd.DataFrame([
            {"source1_id": "S1-001", "candidate_id": "S2-001", "label": 1, "probability": 0.82},
            {"source1_id": "S1-002", "candidate_id": "S2-002", "label": 0, "probability": 0.65},
        ])
        val_df.to_csv(cfg.paths.models_dir / "val_predictions.csv", index=False)

        gt_map = {
            "S1-001": {"S2-001"},
            "S1-002": set(),
        }
        with open(cfg.paths.models_dir / "val_ground_truth.json", "w") as f:
            json.dump({k: list(v) for k, v in gt_map.items()}, f)

        # Tune threshold
        best_t = tune_threshold(cfg, ground_truth_map=gt_map)
        assert best_t >= 0.70  # threshold > 0.65 filters out false positive for S1-002

        # Check file was persisted
        sel_file = cfg.paths.models_dir / "selected_threshold.json"
        assert sel_file.exists()
        with open(sel_file) as f:
            saved_data = json.load(f)
        assert saved_data["threshold"] == best_t
        assert saved_data["metric"] == "macro_f0.5"

        # Check resolve_threshold picks it up automatically
        resolved_t, source = resolve_threshold(cfg)
        assert resolved_t == best_t
        assert "selected_threshold.json" in source

        # Check fallback hierarchy
        # 1. Explicit override
        assert resolve_threshold(cfg, explicit_threshold=0.99)[0] == 0.99
        # 2. Persisted threshold (already checked)
        # 3. If file removed, fallback to config
        sel_file.unlink()
        cfg.threshold.selected = 0.77
        assert resolve_threshold(cfg)[0] == 0.77
        # 4. If selected is None, fallback to default
        cfg.threshold.selected = None
        assert resolve_threshold(cfg)[0] == 0.50


# ================================================================== #
#  3. Candidate pre-ranking (no arbitrary ID truncation)
# ================================================================== #

class TestCandidateRankingNoAlphabeticalBias:
    def test_high_similarity_candidate_preserved_over_alphabetical_order(self, tmp_path):
        """
        True match has ID 'S2-99999' (sorts last alphabetically).
        Noise records have IDs 'S2-00001' to 'S2-00300'.
        Candidate cap is 5.
        Similarity pre-ranking MUST preserve 'S2-99999'.
        """
        cfg = Config()
        cfg.paths.metrics_dir = tmp_path / "metrics"
        cfg.blocking.max_candidates_per_s1 = 5
        cfg.blocking.similarity_prerank = True
        # Enable token block
        cfg.blocking.name_token_block = True
        cfg.blocking.country_block = False

        s1 = pd.DataFrame([{
            "entity_id": "S1-001",
            "business_name": "Antigravity Robotics Solutions",
            "business_name_normalized": "antigravity robotics solutions",
            "business_address": "100 Innovation Way",
            "business_address_normalized": "100 innovation way",
            "country": "India",
            "country_normalized": "india",
            "postal_code": "560001",
            "house_number": "100",
            "address_numeric_tokens": "100",
        }])

        # S2-99999 is highly similar
        target_cand = {
            "entity_id": "S2-99999",
            "business_name": "Antigravity Robotics Solutions",
            "business_name_normalized": "antigravity robotics solutions",
            "business_address": "100 Innovation Way",
            "business_address_normalized": "100 innovation way",
            "country": "India",
            "country_normalized": "india",
            "postal_code": "560001",
            "house_number": "100",
            "address_numeric_tokens": "100",
        }

        # 20 noise records that share only one generic token ('solutions')
        noise_cands = []
        for i in range(1, 21):
            noise_cands.append({
                "entity_id": f"S2-0000{i:02d}",
                "business_name": f"Random Generic Solutions {i}",
                "business_name_normalized": f"random generic solutions {i}",
                "business_address": f"{i} Unrelated Street",
                "business_address_normalized": f"{i} unrelated street",
                "country": "India",
                "country_normalized": "india",
                "postal_code": "110001",
                "house_number": str(i),
                "address_numeric_tokens": str(i),
            })

        s2 = pd.DataFrame(noise_cands + [target_cand])
        s3 = pd.DataFrame(columns=s2.columns)

        gt = {"S1-001": {"S2-99999"}}

        cands = generate_candidates(s1, s2, s3, cfg, ground_truth_map=gt)

        # Total candidates must be capped at 5
        assert len(cands) == 5

        # S2-99999 MUST be among the 5 candidates despite sorting alphabetically last!
        cand_ids = set(cands["candidate_id"])
        assert "S2-99999" in cand_ids


# ================================================================== #
#  4. Candidate recall measurement
# ================================================================== #

class TestCandidateRecallMeasurement:
    def test_candidate_recall_reporting(self, tmp_path):
        cfg = Config()
        cfg.paths.metrics_dir = tmp_path / "metrics"

        cand_df = pd.DataFrame([
            {"source1_id": "S1-001", "candidate_id": "S2-100"},
            {"source1_id": "S1-002", "candidate_id": "S2-200"},
        ])
        gt_map = {
            "S1-001": {"S2-100", "S3-150"},  # 1 found, 1 missed
            "S1-002": {"S2-200"},            # 1 found, 0 missed
            "S1-003": set(),                 # singleton
        }

        metrics = _report_candidate_recall(cand_df, gt_map, cfg)
        assert metrics["total_true_links"] == 3
        assert metrics["true_links_covered"] == 2
        assert metrics["true_links_missed"] == 1
        assert metrics["candidate_recall"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["entities_all_covered_count"] == 1  # S1-002
        assert metrics["entities_any_covered_count"] == 2  # S1-001, S1-002

        # Verify saved file
        saved_file = cfg.paths.metrics_dir / "candidate_recall.json"
        assert saved_file.exists()


# ================================================================== #
#  5. Strict output validation
# ================================================================== #

class TestOutputValidation:
    @pytest.fixture
    def setup_data(self, tmp_path):
        cfg = Config()
        cfg.paths.output_dir = tmp_path / "output"
        cfg.paths.output_dir.mkdir(parents=True)

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        cfg.paths.test_source1 = data_dir / "test_s1.tsv"
        cfg.paths.test_source2 = data_dir / "test_s2.tsv"
        cfg.paths.test_source3 = data_dir / "test_s3.tsv"

        # S1: S1-001, S1-002
        s1 = pd.DataFrame({
            "entity_id": ["S1-001", "S1-002"],
            "business_name": ["A", "B"],
            "business_address": ["Addr A", "Addr B"],
            "country": ["India", "US"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-010"],
            "business_name": ["A"],
            "business_address": ["Addr A"],
            "country": ["India"],
        })
        s3 = pd.DataFrame({
            "entity_id": ["S3-020"],
            "business_name": ["B"],
            "business_address": ["Addr B"],
            "country": ["US"],
        })
        s1.to_csv(cfg.paths.test_source1, sep="\t", index=False)
        s2.to_csv(cfg.paths.test_source2, sep="\t", index=False)
        s3.to_csv(cfg.paths.test_source3, sep="\t", index=False)

        return cfg

    def test_valid_outputs_pass(self, setup_data):
        cfg = setup_data
        match_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["S2-010", ""],
        })
        cand_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_ids": ["S2-010", "S3-020"],
        })
        match_df.to_csv(cfg.paths.output_dir / "matching_results.tsv", sep="\t", index=False)
        cand_df.to_csv(cfg.paths.output_dir / "candidate_pairs.tsv", sep="\t", index=False)

        assert validate_outputs(cfg) is True

    def test_match_not_in_candidate_fails(self, setup_data):
        cfg = setup_data
        match_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["S2-010", ""],
        })
        # Candidate pairs missing S2-010
        cand_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_ids": ["", "S3-020"],
        })
        match_df.to_csv(cfg.paths.output_dir / "matching_results.tsv", sep="\t", index=False)
        cand_df.to_csv(cfg.paths.output_dir / "candidate_pairs.tsv", sep="\t", index=False)

        assert validate_outputs(cfg) is False

    def test_missing_s1_fails(self, setup_data):
        cfg = setup_data
        # S1-002 missing
        match_df = pd.DataFrame({
            "source1_entity_id": ["S1-001"],
            "matched_entity_ids": ["S2-010"],
        })
        cand_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_ids": ["S2-010", "S3-020"],
        })
        match_df.to_csv(cfg.paths.output_dir / "matching_results.tsv", sep="\t", index=False)
        cand_df.to_csv(cfg.paths.output_dir / "candidate_pairs.tsv", sep="\t", index=False)

        assert validate_outputs(cfg) is False

    def test_duplicate_matched_ids_in_row_fails(self, setup_data):
        cfg = setup_data
        match_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["S2-010,S2-010", ""],  # duplicate S2-010
        })
        cand_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_ids": ["S2-010", "S3-020"],
        })
        match_df.to_csv(cfg.paths.output_dir / "matching_results.tsv", sep="\t", index=False)
        cand_df.to_csv(cfg.paths.output_dir / "candidate_pairs.tsv", sep="\t", index=False)

        assert validate_outputs(cfg) is False

    def test_invalid_candidate_id_fails(self, setup_data):
        cfg = setup_data
        match_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "matched_entity_ids": ["", ""],
        })
        cand_df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002"],
            "candidate_entity_ids": ["S2-INVALID", ""],  # S2-INVALID not in test pool
        })
        match_df.to_csv(cfg.paths.output_dir / "matching_results.tsv", sep="\t", index=False)
        cand_df.to_csv(cfg.paths.output_dir / "candidate_pairs.tsv", sep="\t", index=False)

        assert validate_outputs(cfg) is False


# ================================================================== #
#  6. Open-set country (France)
# ================================================================== #

class TestOpenSetCountry:
    def test_france_normalized_correctly(self):
        assert normalize_country("France") == "france"
        assert normalize_country("FRANCE") == "france"
        assert normalize_country("  France  ") == "france"


# ================================================================== #
#  7. Singleton evaluation
# ================================================================== #

class TestSingletonScoring:
    def test_true_singleton_predicted_empty(self):
        pred = {"S1-10": set()}
        gt = {"S1-10": set()}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 1.0

    def test_true_singleton_false_match(self):
        pred = {"S1-10": {"S2-99"}}
        gt = {"S1-10": set()}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 0.0

    def test_true_match_predicted_empty(self):
        pred = {"S1-10": set()}
        gt = {"S1-10": {"S2-99"}}
        m = entity_level_f05(pred, gt)
        assert m["macro_f05"] == 0.0


# ================================================================== #
#  8. Leak-free TF-IDF bundle
# ================================================================== #

class TestLeakFreeTfidf:
    def test_tfidf_fit_train_transform_test(self):
        fcfg = FeaturesConfig()
        bundle = TfidfBundle(fcfg)

        train_name = pd.Series(["apple computers", "google technology", "microsoft software"])
        train_addr = pd.Series(["1 infinite loop", "1600 amphitheatre", "1 microsoft way"])

        bundle.fit(train_name, train_addr)
        assert bundle.name_word_vec is not None
        assert bundle.addr_word_vec is not None

        # Test transformation without fitting
        s1_texts = pd.Series(["apple inc"], index=["S1-A"])
        pool_texts = pd.Series(["apple corp", "banana market"], index=["S2-B1", "S2-B2"])

        sims = bundle.compute_cosine(
            s1_texts, pool_texts,
            pd.Series(["S1-A"]), pd.Series(["S2-B1", "S2-B2"]),
            ["S1-A", "S1-A"], ["S2-B1", "S2-B2"],
            "name_word"
        )
        assert len(sims) == 2
        # apple inc and apple corp share 'apple', should have positive cosine
        assert sims[0] > 0.0
        # apple inc and banana market share nothing
        assert sims[1] == 0.0


# ================================================================== #
#  9. End-to-End Pipeline Integration Test
# ================================================================== #

class TestPipelineEndToEnd:
    def test_pipeline_stages_run_end_to_end(self, tmp_path):
        """Test train -> tune -> predict -> evaluate -> validate end-to-end."""
        from business_entity_resolution.pipeline import (
            run_train, run_tune, run_predict, run_evaluate, run_validate
        )

        cfg = Config()
        cfg.paths.models_dir = tmp_path / "models"
        cfg.paths.artifacts_dir = tmp_path / "artifacts"
        cfg.paths.output_dir = tmp_path / "output"
        cfg.paths.metrics_dir = tmp_path / "artifacts" / "metrics"
        cfg.paths.reports_dir = tmp_path / "reports"
        cfg.project_root = tmp_path
        cfg.model.lightgbm["n_estimators"] = 10
        cfg.model.lightgbm["min_child_samples"] = 1

        train_dir = tmp_path / "data" / "train"
        test_dir = tmp_path / "data" / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        cfg.paths.train_source1 = train_dir / "train_source1.tsv"
        cfg.paths.train_source2 = train_dir / "train_source2.tsv"
        cfg.paths.train_source3 = train_dir / "train_source3.tsv"
        cfg.paths.train_ground_truth = train_dir / "train_ground_truth.tsv"

        cfg.paths.test_source1 = test_dir / "test_source1.tsv"
        cfg.paths.test_source2 = test_dir / "test_source2.tsv"
        cfg.paths.test_source3 = test_dir / "test_source3.tsv"

        cfg.training.validation_split = 0.4

        # Create training data (including France as open set and singletons)
        train_s1 = pd.DataFrame([
            {"entity_id": "S1-01", "business_name": "Antigravity AI Ltd", "business_address": "10 Tech Park, Paris", "country": "France"},
            {"entity_id": "S1-02", "business_name": "Bharat Agro Corp", "business_address": "45 Market Rd, Pune", "country": "India"},
            {"entity_id": "S1-03", "business_name": "Singleton Ventures", "business_address": "99 Lone St, NY", "country": "United States"},
            {"entity_id": "S1-04", "business_name": "Parisian Bakery", "business_address": "12 Rue de Rivoli, Paris", "country": "France"},
            {"entity_id": "S1-05", "business_name": "Mumbai Spices Ltd", "business_address": "88 Spice Bazaar, Mumbai", "country": "India"},
        ])
        train_s2 = pd.DataFrame([
            {"entity_id": "S2-01", "business_name": "Antigravity AI", "business_address": "10 Tech Park, Paris", "country": "France"},
            {"entity_id": "S2-02", "business_name": "Bharat Agriculture", "business_address": "45 Market Road, Pune", "country": "India"},
            {"entity_id": "S2-04", "business_name": "Parisian Bakery", "business_address": "12 Rue Rivoli, Paris", "country": "France"},
            {"entity_id": "S2-05", "business_name": "Mumbai Spices", "business_address": "88 Spice Bazaar, Mumbai", "country": "India"},
        ])
        train_s3 = pd.DataFrame([
            {"entity_id": "S3-01", "business_name": "Antigravity Artificial Intelligence", "business_address": "10 Tech Pk, Paris", "country": "France"},
        ])
        train_gt = pd.DataFrame([
            {"source1_entity_id": "S1-01", "matched_entity_ids": "S2-01,S3-01"},
            {"source1_entity_id": "S1-02", "matched_entity_ids": "S2-02"},
            {"source1_entity_id": "S1-03", "matched_entity_ids": ""},  # singleton
            {"source1_entity_id": "S1-04", "matched_entity_ids": "S2-04"},
            {"source1_entity_id": "S1-05", "matched_entity_ids": "S2-05"},
        ])

        train_s1.to_csv(cfg.paths.train_source1, sep="\t", index=False)
        train_s2.to_csv(cfg.paths.train_source2, sep="\t", index=False)
        train_s3.to_csv(cfg.paths.train_source3, sep="\t", index=False)
        train_gt.to_csv(cfg.paths.train_ground_truth, sep="\t", index=False)

        # Create test data
        test_s1 = pd.DataFrame([
            {"entity_id": "T1-01", "business_name": "Antigravity AI", "business_address": "10 Tech Park, Paris", "country": "France"},
            {"entity_id": "T1-02", "business_name": "Unmatched Company", "business_address": "00 Nowhere", "country": "Germany"},
        ])
        test_s2 = pd.DataFrame([
            {"entity_id": "T2-01", "business_name": "Antigravity AI France", "business_address": "10 Tech Park, Paris", "country": "France"},
        ])
        test_s3 = pd.DataFrame([
            {"entity_id": "T3-01", "business_name": "Irrelevant LLC", "business_address": "88 Other Rd", "country": "US"},
        ])

        test_s1.to_csv(cfg.paths.test_source1, sep="\t", index=False)
        test_s2.to_csv(cfg.paths.test_source2, sep="\t", index=False)
        test_s3.to_csv(cfg.paths.test_source3, sep="\t", index=False)

        # Run pipeline stages
        run_train(cfg)
        assert (cfg.paths.models_dir / "val_predictions.csv").exists()
        assert (cfg.paths.models_dir / "val_ground_truth.json").exists()

        run_tune(cfg)
        assert (cfg.paths.models_dir / "selected_threshold.json").exists()
        assert (cfg.paths.metrics_dir / "threshold_sweep.json").exists()

        run_predict(cfg)
        assert (cfg.paths.output_dir / "matching_results.tsv").exists()
        assert (cfg.paths.output_dir / "candidate_pairs.tsv").exists()

        run_evaluate(cfg)

        # Validate outputs
        run_validate(cfg)
        assert validate_outputs(cfg) is True


class TestMaxBlockSizePruning:
    """Verify that uninformative blocks exceeding max_block_size are pruned."""

    def test_large_blocks_pruned(self):
        from business_entity_resolution.blocking import build_block_indices

        cfg = Config()
        cfg.blocking.max_block_size = 3
        # Create pool with 5 records sharing a generic token 'common' and 2 records with 'rare'
        pool = pd.DataFrame([
            {"entity_id": f"E{i}", "business_name": f"common business {i}", "business_name_normalized": f"common business {i}",
             "business_address": "st", "business_address_normalized": "st", "country": "US", "country_normalized": "us", "postal_code": ""}
            for i in range(5)
        ] + [
            {"entity_id": f"R{i}", "business_name": f"rare unique {i}", "business_name_normalized": f"rare unique {i}",
             "business_address": "st", "business_address_normalized": "st", "country": "US", "country_normalized": "us", "postal_code": ""}
            for i in range(2)
        ])

        indices = build_block_indices(pool, cfg.blocking)
        name_idx = indices.get("name_token", {})
        # 'ntok:common' has 5 records > max_block_size 3 -> pruned!
        assert "ntok:common" not in name_idx
        # 'ntok:rare' has 2 records <= 3 -> kept!
        assert "ntok:rare" in name_idx


class TestChunkedStreamingInference:
    """Verify that chunked streaming inference outputs strictly valid files."""

    def test_chunked_predict_matches_schema(self, tmp_path):
        from business_entity_resolution.output_writer import append_chunk_outputs
        from business_entity_resolution.validation import validate_outputs

        cfg = Config()
        cfg.paths.output_dir = tmp_path / "output"
        cfg.paths.test_source1 = tmp_path / "test_s1.tsv"
        cfg.paths.test_source2 = tmp_path / "test_s2.tsv"
        cfg.paths.test_source3 = tmp_path / "test_s3.tsv"

        s1_records = [f"S1-{i:03d}" for i in range(10)]
        pd.DataFrame([{"entity_id": sid, "business_name": "Co", "business_address": "Ad", "country": "US"} for sid in s1_records]).to_csv(cfg.paths.test_source1, sep="\t", index=False)
        pd.DataFrame([{"entity_id": "S2-01", "business_name": "Co", "business_address": "Ad", "country": "US"}]).to_csv(cfg.paths.test_source2, sep="\t", index=False)
        pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]).to_csv(cfg.paths.test_source3, sep="\t", index=False)

        # Chunk 1: first 5
        cands1 = {sid: {"S2-01"} for sid in s1_records[:5]}
        results1 = {sid: {"S2-01"} for sid in s1_records[:2]}
        append_chunk_outputs(results1, cands1, s1_records[:5], cfg, mode="w")

        # Chunk 2: next 5
        cands2 = {sid: set() for sid in s1_records[5:]}
        results2 = {}
        append_chunk_outputs(results2, cands2, s1_records[5:], cfg, mode="a")

        assert validate_outputs(cfg) is True


