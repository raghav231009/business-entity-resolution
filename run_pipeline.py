#!/usr/bin/env python
"""
run_pipeline.py — CLI entry-point
===================================
Usage
-----
    python run_pipeline.py --stage all
    python run_pipeline.py --stage train
    python run_pipeline.py --stage predict
    python run_pipeline.py --stage evaluate
    python run_pipeline.py --stage tune
    python run_pipeline.py --stage eda

Use ``--config`` to point to an alternative config file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the ``src/`` directory is on the import path so that
# ``business_entity_resolution`` can be imported without installing.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from business_entity_resolution.config import load_config  # noqa: E402


# ------------------------------------------------------------------ #
# Stages
# ------------------------------------------------------------------ #

VALID_STAGES = (
    "all",
    "eda",
    "train",
    "tune",
    "predict",
    "evaluate",
    "validate",
    "verify",
    "env-check",
    "smoke-test",
    "cap-experiment",
)


def _setup_logging(level: str) -> None:
    """Configure root logger with a readable format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_stage(stage: str, cfg) -> None:
    """Dispatch to the appropriate pipeline stage."""
    if stage == "env-check":
        _stage_env_check()
        return

    if stage == "smoke-test":
        _stage_smoke_test(cfg)
        return

    if stage == "cap-experiment":
        _stage_cap_experiment(cfg)
        return

    if stage in ("all", "eda"):
        _stage_eda(cfg)
        if stage != "all":
            return

    if stage in ("all", "train"):
        _stage_train(cfg)
        if stage != "all":
            return

    if stage in ("all", "tune"):
        _stage_tune(cfg)
        if stage != "all":
            return

    if stage in ("all", "predict"):
        _stage_predict(cfg)
        if stage != "all":
            return

    if stage in ("all", "evaluate"):
        _stage_evaluate(cfg)
        if stage != "all":
            return

    if stage in ("all", "validate"):
        _stage_validate(cfg)
        if stage != "all":
            return

    if stage in ("all", "verify"):
        _stage_verify(cfg)
        if stage != "all":
            return


# ------------------------------------------------------------------ #
# Stage implementations (thin wrappers around pipeline.py)
# ------------------------------------------------------------------ #

def _stage_env_check() -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[0/8] ENV-CHECK — Environment & Dependency Verification")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.reproducibility import check_environment
    ok, _ = check_environment()
    if not ok:
        sys.exit(1)


def _stage_smoke_test(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[SMOKE] SMOKE TEST — Fast Truth-Aware Micro-Pipeline")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_smoke_test
    ok = run_smoke_test(cfg)
    if not ok:
        sys.exit(1)


def _stage_eda(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[1/8] EDA — Exploratory Data Analysis")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_eda
    run_eda(cfg)


def _stage_train(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[2/8] TRAIN — Building model")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_train
    run_train(cfg)


def _stage_tune(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[3/8] TUNE — Threshold tuning")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_tune
    run_tune(cfg)


def _stage_predict(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[4/8] PREDICT — Generating predictions")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_predict
    run_predict(cfg)


def _stage_evaluate(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[5/8] EVALUATE — Computing metrics")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_evaluate
    run_evaluate(cfg)


def _stage_validate(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[6/8] VALIDATE — Checking output files")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_validate
    run_validate(cfg)


def _stage_verify(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[7/8] VERIFY — Comprehensive Submission Verification")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.pipeline import run_verify
    run_verify(cfg)


def _stage_cap_experiment(cfg) -> None:
    logging.getLogger(__name__).info("=" * 60)
    logging.getLogger(__name__).info("[EXP] Candidate Cap Tradeoff Experiment (K = 25, 50, 75, 100, 150)")
    logging.getLogger(__name__).info("=" * 60)
    from business_entity_resolution.candidate_cap_experiment import run_candidate_cap_experiment
    run_candidate_cap_experiment(cfg)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python run_pipeline.py --stage env-check\n"
               "  python run_pipeline.py --mode dev --stage smoke-test\n"
               "  python run_pipeline.py --mode dev --stage all\n"
               "  python run_pipeline.py --mode final --stage all\n"
               "  python run_pipeline.py --mode final --stage verify\n",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        choices=["dev", "final"],
        help="Execution mode: 'dev' for fast testing, 'final' for full competition run. Defaults to 'dev'.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=VALID_STAGES,
        help=f"Pipeline stage to run. Choices: {', '.join(VALID_STAGES)}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to project root).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for dataset files (e.g. data or data_dev).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, data_root=args.data_root, mode=args.mode)
    _setup_logging(cfg.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Pipeline mode  = %s", cfg.execution.mode)
    logger.info("Pipeline stage = %s", args.stage)
    logger.info("Project root   = %s", cfg.project_root)

    _run_stage(args.stage, cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()

