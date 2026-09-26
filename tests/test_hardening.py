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

import json
import tempfile
from pathlib import Path
import pytest

from business_entity_resolution.config import load_config, Config
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
