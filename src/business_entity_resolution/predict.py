"""
predict.py — Score candidate pairs using a trained model
==========================================================
Loads the saved model, applies it to feature columns, and
returns the candidate DataFrame augmented with ``probability``.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import Config
from .train import load_model, load_feature_columns

logger = logging.getLogger(__name__)


def predict(
    feat_df: pd.DataFrame,
    cfg: Config,
    model=None,
    feature_cols=None,
) -> pd.DataFrame:
    """
    Load the trained model and score every candidate pair.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix (from ``feature_builder.build_feature_matrix()``).
    cfg : Config
    model : optional pre-loaded model
    feature_cols : optional pre-loaded feature columns

    Returns
    -------
    pd.DataFrame
        ``feat_df`` with an added ``probability`` column.
    """
    if model is None:
        model = load_model(cfg)
    if feature_cols is None:
        feature_cols = load_feature_columns(cfg)

    # Keep only columns that exist
    available = [c for c in feature_cols if c in feat_df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning("Missing features (will be zero-filled): %s", missing)
        for c in missing:
            feat_df[c] = 0.0

    X = feat_df[feature_cols].astype(float)
    probs = model.predict_proba(X)[:, 1]

    result = feat_df.copy()
    result["probability"] = probs

    logger.info("Scored %d candidate pairs.", len(result))
    logger.info("  Prob stats — min=%.4f  median=%.4f  max=%.4f",
                probs.min(), probs.mean(), probs.max())

    return result
