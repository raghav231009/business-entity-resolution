"""
tests/test_hardening.py — Tests for production hardening, execution modes, and reproducibility
=============================================================================================
Covers:
1. Environment check: all required dependencies present and operational.
2. Dual execution mode configuration: dev (capped) vs final (unrestricted).
3. SHA-256 hash and reproducibility audit trail utilities.
4. Manifest structure, completeness, and schema validation.
5. End-to-end smoke test execution in isolated temporary paths.
"""

from business_entity_resolution.config import load_config
from business_entity_resolution.reproducibility import (
    check_environment,
    compute_file_sha256,
    compute_code_hash,
    create_manifest,
    run_smoke_test,
)


# ================================================================== #
#  1. Environment Check
# ================================================================== #

class TestEnvironmentCheck:
    def test_all_dependencies_present(self):
        """Verify that check_environment reports all packages operational."""
        all_ok, versions = check_environment()
        assert all_ok is True
        required_pkgs = [
            "pandas",
            "numpy",
            "scikit-learn",
            "rapidfuzz",
            "lightgbm",
            "joblib",
            "pyyaml",
            "pytest",
        ]
        for pkg in required_pkgs:
            assert pkg in versions
            assert versions[pkg] != "MISSING"


# ================================================================== #
#  2. Execution Modes (Dev vs Final)
# ================================================================== #

class TestExecutionModes:
    def test_default_mode_is_dev(self):
        """Default mode must be dev to prevent accidental long runs."""
        cfg = load_config()
        assert cfg.execution.mode == "dev"
        assert cfg.training.max_train_s1_entities == 50000
        assert cfg.training.max_validation_s1_entities == 10000

    def test_explicit_dev_mode(self):
        """Explicit dev mode sets the bounded limits."""
        cfg = load_config(mode="dev")
        assert cfg.execution.mode == "dev"
        assert cfg.training.max_train_s1_entities == 50000
        assert cfg.training.max_validation_s1_entities == 10000

    def test_explicit_final_mode_no_caps(self):
        """Final mode must have None for max entities (full dataset)."""
        cfg = load_config(mode="final")
        assert cfg.execution.mode == "final"
        assert cfg.training.max_train_s1_entities is None
        assert cfg.training.max_validation_s1_entities is None

    def test_unknown_mode_falls_back_to_dev(self):
        """Unknown mode should safely default to dev mode."""
        cfg = load_config(mode="unsupported_mode")
        assert cfg.execution.mode == "dev"
        assert cfg.training.max_train_s1_entities == 50000


# ================================================================== #
#  3. Hashing and Reproducibility Manifest
# ================================================================== #

class TestHashingAndManifest:
    def test_compute_file_sha256(self, tmp_path):
        """Deterministic SHA-256 for a test file."""
        test_file = tmp_path / "sample.txt"
        test_file.write_text("Amazon ML Challenge 2026", encoding="utf-8")
        h1 = compute_file_sha256(test_file)
        h2 = compute_file_sha256(test_file)
        assert len(h1) == 64
        assert h1 == h2

    def test_compute_code_hash(self):
        """Code hash over src directory returns a 64-character SHA-256."""
        cfg = load_config()
        code_h = compute_code_hash(cfg.project_root)
        assert len(code_h) == 64
        assert code_h != "none"

    def test_manifest_structure(self, tmp_path):
        """Manifest contains all required audit fields."""
        cfg = load_config(mode="dev")
        # Direct artifacts to tmp_path
        cfg.paths.artifacts_dir = tmp_path
        manifest = create_manifest(cfg, extra_metrics={"custom_metric": 0.95})

        assert manifest["manifest_version"] == "1.0.0"
        assert manifest["execution_mode"] == "dev"
        assert "timestamp_utc" in manifest
        assert "git_commit" in manifest
        assert "source_code_sha256" in manifest
        assert "config_file_sha256" in manifest
        assert "environment" in manifest
        assert "dataset_files" in manifest
        assert "model" in manifest
        assert "output_files" in manifest
        assert "pipeline_metrics" in manifest
        assert manifest["pipeline_metrics"]["custom_metric"] == 0.95

        saved_file = tmp_path / "final_run_manifest.json"
        assert saved_file.exists()


# ================================================================== #
#  4. Smoke Test Stage
# ================================================================== #

class TestSmokeTestStage:
    def test_smoke_test_runs_and_passes(self):
        """Smoke test exercises end-to-end pipeline in isolated temp paths without errors."""
        cfg = load_config(mode="dev")
        result = run_smoke_test(cfg)
        assert result is True


# ================================================================== #
#  5. Candidate Recall Zero Ground Truth Handling
# ================================================================== #

class TestCandidateRecallHardening:
    def test_zero_true_links_returns_none_not_one(self):
        """Candidate recall must return None when total_true_links == 0, never 1.0."""
        import pandas as pd
        from business_entity_resolution.candidate_generation import _report_candidate_recall

        cfg = load_config(mode="dev")
        cand_df = pd.DataFrame([
            {"source1_id": "S1-A", "candidate_id": "S2-1"},
            {"source1_id": "S1-B", "candidate_id": "S2-2"},
        ])
        # Ground truth has zero true matches (all singletons)
        gt_map = {"S1-A": set(), "S1-B": set()}

        metrics = _report_candidate_recall(cand_df, gt_map, cfg)
        assert metrics["total_true_links"] == 0
        assert metrics["candidate_recall"] is None
        assert metrics["candidate_recall_status"] == "NOT_AVAILABLE"
        assert "Candidate recall not measured" in str(metrics["message"])

    def test_positive_true_links_computes_exact_recall(self):
        """Candidate recall with real links returns exact ratio."""
        import pandas as pd
        from business_entity_resolution.candidate_generation import _report_candidate_recall

        cfg = load_config(mode="dev")
        cand_df = pd.DataFrame([
            {"source1_id": "S1-A", "candidate_id": "S2-1"},
            {"source1_id": "S1-B", "candidate_id": "S2-2"},
        ])
        # S1-A true is S2-1, S1-B true is S2-9 (missed)
        gt_map = {"S1-A": {"S2-1"}, "S1-B": {"S2-9"}}

        metrics = _report_candidate_recall(cand_df, gt_map, cfg)
        assert metrics["total_true_links"] == 2
        assert metrics["true_links_covered"] == 1
        assert metrics["true_links_missed"] == 1
        assert metrics["candidate_recall"] == 0.5
        assert metrics["candidate_recall_status"] == "MEASURED"


# ================================================================== #
#  6. TF-IDF Leak-Free Fitting & Transform
# ================================================================== #

class TestTfidfLeakFree:
    def test_bundle_is_fitted_flag(self):
        """TfidfBundle sets is_fitted=True on fit and uses transform on data."""
        import pandas as pd
        from business_entity_resolution.feature_builder import TfidfBundle

        cfg = load_config(mode="dev")
        bundle = TfidfBundle(cfg.features)
        assert bundle.is_fitted is False

        corpus_names = pd.Series(["Amazon Inc", "Google LLC", "Microsoft Corp"])
        corpus_addrs = pd.Series(["410 Terry Ave", "1600 Amphitheatre Pkwy", "One Microsoft Way"])
        bundle.fit(corpus_names, corpus_addrs)
        assert bundle.is_fitted is True

        # Calling compute_cosine does not alter vectorizer vocabulary (strictly transform)
        assert bundle.name_word_vec is not None
        vocab_size_before = len(bundle.name_word_vec.vocabulary_)
        cand_s1 = ["S1-1"]
        cand_pool = ["S2-1"]
        s1_names = pd.Series(["Apple Inc"], index=["S1-1"])
        pool_names = pd.Series(["Apple Computer"], index=["S2-1"])
        sims = bundle.compute_cosine(
            s1_names, pool_names,
            pd.Series(["S1-1"]), pd.Series(["S2-1"]),
            cand_s1, cand_pool,
            vec_type="name_word",
        )
        assert len(sims) == 1
        assert bundle.name_word_vec is not None
        assert len(bundle.name_word_vec.vocabulary_) == vocab_size_before


# ================================================================== #
#  7. Report Generation No Fallbacks
# ================================================================== #

class TestReportHardening:
    def test_no_hardcoded_metric_fallbacks_in_markdown(self, tmp_path):
        """Report must show NOT_AVAILABLE / NOT_MEASURED instead of 0.9766 / 0.9705 when metrics missing."""
        from business_entity_resolution.reproducibility import generate_final_submission_report

        cfg = load_config(mode="dev")
        # Manifest with missing/null metrics
        manifest = {
            "execution_mode": "dev",
            "dataset_files": {},
            "environment": {"package_versions": {}},
            "model": {"model_type": "lightgbm", "model_file": "entity_matcher.joblib", "model_sha256": "abc"},
            "output_files": {},
            "pipeline_metrics": {
                "max_train_s1_entities": 50000,
                "max_validation_s1_entities": 10000,
                "candidate_recall": None,
                "validation_macro_f05": None,
            },
        }
        report_path = tmp_path / "report.md"
        report_text = generate_final_submission_report(manifest, cfg, report_path)

        assert "0.9766" not in report_text
        assert "0.9705" not in report_text
        assert "NOT_AVAILABLE" in report_text
        assert "NOT_MEASURED" in report_text
