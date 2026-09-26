"""
data_loader.py — Dataset loading & column validation
=====================================================
Responsible ONLY for reading TSV files and validating that
expected columns exist.  No feature engineering happens here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Expected column schemas
# ------------------------------------------------------------------ #
# We define the *minimum* expected columns per source.  Extra columns
# are silently kept so the pipeline never breaks when the challenge
# adds optional fields.

_SOURCE_COLS = {"entity_id", "business_name", "business_address", "country"}
_GROUND_TRUTH_COLS = {"source1_entity_id", "matched_entity_ids"}


# ================================================================== #
#  Container for one split (train or test)
# ================================================================== #

@dataclass
class SourceData:
    """Holds the three source DataFrames (plus optional ground truth)."""
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame
    ground_truth: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        parts = [
            f"  Source 1 : {len(self.source1):>7,} rows",
            f"  Source 2 : {len(self.source2):>7,} rows",
            f"  Source 3 : {len(self.source3):>7,} rows",
        ]
        if self.ground_truth is not None:
            parts.append(f"  Ground-truth rows: {len(self.ground_truth):>7,}")
        return "\n".join(parts)


# ================================================================== #
#  Low-level loaders
# ================================================================== #

def _read_tsv(path: Path, expected_cols: set[str], label: str) -> pd.DataFrame:
    """
    Read a tab-separated file and validate columns.

    Parameters
    ----------
    path : Path
        Absolute path to the TSV file.
    expected_cols : set[str]
        Column names that *must* be present.
    label : str
        Human-readable label for log messages.

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"[{label}] File not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("[%s] Loaded %d rows × %d cols from %s", label, len(df), len(df.columns), path.name)

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[{label}] Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
    return df


# ================================================================== #
#  Public API — individual loaders
# ================================================================== #

def load_source(path: Path, label: str = "source") -> pd.DataFrame:
    """Load a single source TSV (S1, S2, or S3)."""
    return _read_tsv(path, _SOURCE_COLS, label)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """
    Load the ground-truth TSV.

    The ``matched_entity_ids`` column may contain empty strings
    (no matches) or comma-separated IDs.  We keep it as a raw string
    here; parsing into sets is done by ``ground_truth.py``.
    """
    return _read_tsv(path, _GROUND_TRUTH_COLS, "ground_truth")


# ================================================================== #
#  Convenience loaders for full splits
# ================================================================== #

def load_train_data(cfg: Config) -> SourceData:
    """Load all training data (three sources + ground truth)."""
    logger.info("Loading TRAINING data …")
    s1 = load_source(cfg.paths.train_source1, "train_source1")
    s2 = load_source(cfg.paths.train_source2, "train_source2")
    s3 = load_source(cfg.paths.train_source3, "train_source3")
    gt = load_ground_truth(cfg.paths.train_ground_truth)
    data = SourceData(source1=s1, source2=s2, source3=s3, ground_truth=gt)
    logger.info("Training data summary:\n%s", data.summary())
    return data


def load_test_data(cfg: Config) -> SourceData:
    """Load all test data (three sources, no ground truth)."""
    logger.info("Loading TEST data …")
    s1 = load_source(cfg.paths.test_source1, "test_source1")
    s2 = load_source(cfg.paths.test_source2, "test_source2")
    s3 = load_source(cfg.paths.test_source3, "test_source3")
    data = SourceData(source1=s1, source2=s2, source3=s3)
    logger.info("Test data summary:\n%s", data.summary())
    return data
