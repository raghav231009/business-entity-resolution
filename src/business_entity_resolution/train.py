"""
train.py — Model training
===========================
Trains a LightGBM (or CatBoost) classifier on the labelled
feature matrix and saves the model + metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .feature_builder import get_feature_columns

logger = logging.getLogger(__name__)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Config,
    val_gt_map: Optional[dict] = None,
) -> None:
    """
    Train the entity-matching classifier and persist it.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Must contain feature columns + ``label``.
    cfg : Config
    val_gt_map : dict, optional
        If provided, entity-level F0.5 is computed on the validation set.
    """
    feature_cols = get_feature_columns(cfg)
    # Keep only columns that actually exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["label"].astype(int)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["label"].astype(int)

    logger.info("Training %s  (%d features, %d train, %d val)",
                cfg.model.type, len(feature_cols), len(X_train), len(X_val))

    if cfg.model.type == "lightgbm":
        model = _train_lightgbm(X_train, y_train, X_val, y_val, cfg)
    elif cfg.model.type == "catboost":
        model = _train_catboost(X_train, y_train, X_val, y_val, cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")

    # ---- Save model ------------------------------------------------
    models_dir = cfg.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{cfg.model.saved_model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # Save feature names
    meta = {
        "feature_columns": feature_cols,
        "model_type": cfg.model.type,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "train_pos": int(y_train.sum()),
        "train_neg": int((y_train == 0).sum()),
    }
    meta_path = models_dir / f"{cfg.model.saved_model_name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    # Save validation predictions for threshold tuning
    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    val_out = val_df[["source1_id", "candidate_id", "label"]].copy()
    val_out["probability"] = val_probs
    val_path = models_dir / "val_predictions.csv"
    val_out.to_csv(val_path, index=False)
    logger.info("Validation predictions saved to %s", val_path)

    # Persist full validation ground truth for unbiased evaluation
    if val_gt_map is not None:
        val_gt_serializable = {s1: sorted(matches) for s1, matches in val_gt_map.items()}
        val_gt_path = models_dir / "val_ground_truth.json"
        with open(val_gt_path, "w", encoding="utf-8") as f:
            json.dump(val_gt_serializable, f, indent=2)
        logger.info("Validation full ground truth saved to %s (%d entities)", val_gt_path, len(val_gt_map))


# ================================================================== #
#  LightGBM
# ================================================================== #

def _train_lightgbm(X_train, y_train, X_val, y_val, cfg):
    import lightgbm as lgb

    params = dict(cfg.model.lightgbm)
    # Map 'class_weight' → LightGBM's 'is_unbalance'
    if params.pop("class_weight", None) == "balanced":
        params["is_unbalance"] = True

    model = lgb.LGBMClassifier(
        random_state=cfg.random_seed,
        **params,
    )
    fit_kwargs = {}
    if len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "binary_logloss"

    model.fit(
        X_train, y_train,
        **fit_kwargs,
    )

    # Feature importances
    imp = sorted(zip(X_train.columns, model.feature_importances_),
                 key=lambda x: -x[1])
    logger.info("Top-10 features:")
    for name, score in imp[:10]:
        logger.info("  %-30s  %d", name, score)

    return model


# ================================================================== #
#  CatBoost
# ================================================================== #

def _train_catboost(X_train, y_train, X_val, y_val, cfg):
    from catboost import CatBoostClassifier

    params = dict(cfg.model.catboost)
    model = CatBoostClassifier(
        random_seed=cfg.random_seed,
        **params,
    )
    fit_kwargs = {}
    if len(X_val) > 0:
        fit_kwargs["eval_set"] = (X_val, y_val)

    model.fit(
        X_train, y_train,
        **fit_kwargs,
    )
    return model


# ================================================================== #
#  Model loading
# ================================================================== #

def load_model(cfg: Config):
    """Load the persisted model from disk."""
    model_path = cfg.paths.models_dir / f"{cfg.model.saved_model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


def load_feature_columns(cfg: Config, enforce_compatibility: bool = True) -> list[str]:
    """
    Load the feature column names from saved metadata and verify compatibility.
    """
    meta_path = cfg.paths.models_dir / f"{cfg.model.saved_model_name}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    saved_cols = meta["feature_columns"]

    if enforce_compatibility:
        current_cols = get_feature_columns(cfg)
        # Compare sets
        missing = set(saved_cols) - set(current_cols)
        unexpected = set(current_cols) - set(saved_cols)
        if missing or unexpected:
            raise ValueError(
                f"Feature compatibility error: saved model features do not match current pipeline config.\n"
                f"  Missing in current pipeline: {missing}\n"
                f"  Unexpected in current pipeline: {unexpected}"
            )
        if saved_cols != current_cols:
            logger.warning("Feature column order differs between saved model and current config — using saved model order.")

    return saved_cols
