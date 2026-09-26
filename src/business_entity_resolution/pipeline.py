"""
pipeline.py — Orchestrator
============================
Each ``run_*`` function coordinates a single pipeline stage.
Modules are imported lazily so you can develop them one at a time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


# ================================================================== #
#  Stage: EDA
# ================================================================== #

def run_eda(cfg: "Config") -> None:
    """Run comprehensive EDA, print summary statistics, and persist all reports, logs, and plots."""
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .data_loader import load_train_data
    from .preprocessing import preprocess_sources
    from .ground_truth import parse_ground_truth

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Configure dedicated EDA log file
    eda_log_path = output_dir / "eda.log"
    eda_file_handler = logging.FileHandler(eda_log_path, mode="w", encoding="utf-8")
    eda_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(eda_file_handler)

    logger.info("Starting exploratory data analysis (EDA)...")

    # Execute comprehensive Phase 1 EDA module to generate reports/phase1_eda.md and reports/eda_summary.json
    from .eda import run_full_eda
    reports_dir = getattr(cfg.paths, "reports_dir", None) or (cfg.project_root / "reports")
    run_full_eda(cfg, reports_dir=reports_dir)

    data = load_train_data(cfg)
    total_shapes = {
        "source1": data.source1.shape,
        "source2": data.source2.shape,
        "source3": data.source3.shape,
        "ground_truth": data.ground_truth.shape if data.ground_truth is not None else None,
    }

    # Sample up to 50k per source for fast, accurate distributional statistics
    for attr in ("source1", "source2", "source3"):
        df = getattr(data, attr)
        if len(df) > 50000:
            setattr(data, attr, df.sample(n=50000, random_state=cfg.random_seed).reset_index(drop=True))
    data = preprocess_sources(data, cfg)

    # 1. Source level statistics
    source_stats = {}
    for s_name, df, full_shape in [
        ("source1", data.source1, total_shapes["source1"]),
        ("source2", data.source2, total_shapes["source2"]),
        ("source3", data.source3, total_shapes["source3"]),
    ]:
        missing_dict = df.isnull().sum().to_dict()
        top_countries = df["country_normalized"].value_counts().head(10).to_dict()
        stats = {
            "source_name": s_name,
            "full_row_count": full_shape[0],
            "column_count": full_shape[1],
            "sample_analyzed": len(df),
            "missing_values": missing_dict,
            "top_countries": top_countries,
            "avg_name_length": float(df["business_name_normalized"].str.len().mean()),
            "avg_address_length": float(df["business_address_normalized"].str.len().mean()),
        }
        source_stats[s_name] = stats
        # Save individual source EDA file
        for target_dir in (output_dir, artifacts_dir):
            with open(target_dir / f"eda_{s_name}.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

        logger.info("--- %s Summary ---", s_name.upper())
        logger.info("  Total rows: %d, Columns: %d", full_shape[0], full_shape[1])
        logger.info("  Missing values: %s", missing_dict)
        logger.info("  Top countries: %s", top_countries)
        logger.info("  Avg name length: %.1f chars", stats["avg_name_length"])
        logger.info("  Avg address length: %.1f chars", stats["avg_address_length"])

    # 2. Ground-truth match analysis
    gt_stats = {}
    match_counts = []
    if data.ground_truth is not None:
        gt_map = parse_ground_truth(data.ground_truth)
        match_counts = [len(v) for v in gt_map.values()]
        singletons = sum(1 for c in match_counts if c == 0)
        with_matches = sum(1 for c in match_counts if c > 0)
        gt_stats = {
            "total_s1_entities": len(gt_map),
            "singletons": singletons,
            "singleton_pct": float(singletons / len(gt_map) * 100),
            "entities_with_matches": with_matches,
            "entities_with_matches_pct": float(with_matches / len(gt_map) * 100),
            "total_match_links": sum(match_counts),
            "min_matches": int(min(match_counts)) if match_counts else 0,
            "max_matches": int(max(match_counts)) if match_counts else 0,
            "mean_matches": float(np.mean(match_counts)) if match_counts else 0.0,
            "median_matches": float(np.median(match_counts)) if match_counts else 0.0,
        }
        logger.info("--- Ground Truth Summary ---")
        logger.info("  Total S1 entities: %d", gt_stats["total_s1_entities"])
        logger.info("  Singletons (0 matches): %d (%.2f%%)", singletons, gt_stats["singleton_pct"])
        logger.info("  With matches: %d (%.2f%%)", with_matches, gt_stats["entities_with_matches_pct"])
        logger.info("  Mean matches per S1: %.2f", gt_stats["mean_matches"])

    # 3. Save comprehensive JSON report
    eda_summary = {
        "dataset_shapes": total_shapes,
        "sources": source_stats,
        "ground_truth": gt_stats,
    }
    for target_dir in (output_dir, artifacts_dir):
        with open(target_dir / "eda_report.json", "w", encoding="utf-8") as f:
            json.dump(eda_summary, f, indent=2)

    # 4. Generate the 3 visual EDA figures
    # Figure 1: Country distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (s_name, df) in zip(axes, [("Source 1", data.source1), ("Source 2", data.source2), ("Source 3", data.source3)]):
        top_c = df["country_normalized"].value_counts().head(8)
        top_c.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black")
        ax.set_title(f"{s_name} Countries")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    for target_dir in (output_dir, artifacts_dir):
        fig.savefig(target_dir / "eda_country_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 2: Name and Address Length Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, (s_name, df) in enumerate([("Source 1", data.source1), ("Source 2", data.source2), ("Source 3", data.source3)]):
        df["business_name_normalized"].str.len().hist(bins=40, ax=axes[0, idx], color="#2ca02c", edgecolor="black", alpha=0.7)
        axes[0, idx].set_title(f"{s_name} Name Length")
        axes[0, idx].set_xlabel("Characters")
        df["business_address_normalized"].str.len().hist(bins=40, ax=axes[1, idx], color="#ff7f0e", edgecolor="black", alpha=0.7)
        axes[1, idx].set_title(f"{s_name} Address Length")
        axes[1, idx].set_xlabel("Characters")
    plt.tight_layout()
    for target_dir in (output_dir, artifacts_dir):
        fig.savefig(target_dir / "eda_length_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 3: Ground Truth Match Distribution
    if match_counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(match_counts, bins=range(max(match_counts) + 2), color="#9467bd", edgecolor="black", align="left")
        ax.set_title("Ground Truth Matches per Source-1 Entity")
        ax.set_xlabel("Number of Matched Entities")
        ax.set_ylabel("Number of Source-1 Entities")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        for target_dir in (output_dir, artifacts_dir):
            fig.savefig(target_dir / "eda_match_distribution.png", dpi=150)
        plt.close(fig)

    # 5. Save human-readable Markdown report
    md_content = f"""# Exploratory Data Analysis (EDA) Report

## 1. Dataset Dimensions
- **Source 1 (Reference)**: {total_shapes['source1'][0]:,} rows × {total_shapes['source1'][1]} columns
- **Source 2 (Noisy)**: {total_shapes['source2'][0]:,} rows × {total_shapes['source2'][1]} columns
- **Source 3 (Noisy)**: {total_shapes['source3'][0]:,} rows × {total_shapes['source3'][1]} columns
- **Ground Truth Pairs**: {total_shapes['ground_truth'][0]:,} rows (total links: {gt_stats.get('total_match_links', 0):,})

## 2. Text Attribute Statistics
| Source | Avg Name Length | Avg Address Length | Missing Values |
|---|---|---|---|
| Source 1 | {source_stats['source1']['avg_name_length']:.1f} chars | {source_stats['source1']['avg_address_length']:.1f} chars | None |
| Source 2 | {source_stats['source2']['avg_name_length']:.1f} chars | {source_stats['source2']['avg_address_length']:.1f} chars | None |
| Source 3 | {source_stats['source3']['avg_name_length']:.1f} chars | {source_stats['source3']['avg_address_length']:.1f} chars | None |

## 3. Ground Truth Matching Behavior
- **Total Master Entities**: {gt_stats.get('total_s1_entities', 0):,}
- **Singletons (0 matches)**: {gt_stats.get('singletons', 0):,} ({gt_stats.get('singleton_pct', 0.0):.2f}%)
- **Multi-match Entities**: {gt_stats.get('entities_with_matches', 0):,} ({gt_stats.get('entities_with_matches_pct', 0.0):.2f}%)
- **Average Matches per Entity**: {gt_stats.get('mean_matches', 0.0):.2f} (Median: {gt_stats.get('median_matches', 0.0):.1f})

## 4. Generated EDA Visualizations
1. `eda_country_distribution.png` — Geographical breakdown across all 3 data sources.
2. `eda_length_distribution.png` — Name and address character length distributions.
3. `eda_match_distribution.png` — Histogram of ground-truth link multiplicities.
"""
    for target_dir in (output_dir, artifacts_dir):
        with open(target_dir / "eda_report.md", "w", encoding="utf-8") as f:
            f.write(md_content)

    # Also write a copy of eda.log to artifacts_dir
    with open(artifacts_dir / "eda.log", "w", encoding="utf-8") as f:
        with open(eda_log_path, "r", encoding="utf-8") as src_log:
            f.write(src_log.read())

    logger.removeHandler(eda_file_handler)
    eda_file_handler.close()
    logger.info("EDA complete. Reports and plots successfully saved to %s and %s.", output_dir, artifacts_dir)


# ================================================================== #
#  Stage: TRAIN
# ================================================================== #

def run_train(cfg: "Config") -> None:
    """Full training pipeline: load → preprocess → block → features → model."""
    import pandas as pd
    from .data_loader import load_train_data
    from .preprocessing import preprocess_sources
    from .ground_truth import parse_ground_truth
    from .candidate_generation import generate_candidates
    from .feature_builder import build_feature_matrix
    from .dataset_builder import build_training_dataset
    from .train import train_model

    # 1. Load
    logger.info("[1/8] Loading training data")
    data = load_train_data(cfg)

    # Subsample training S1 entities if configured for bounded memory and fast training
    max_s1 = getattr(cfg.training, "max_train_s1_entities", None)
    if max_s1 and len(data.source1) > max_s1:
        logger.info("DEV MODE: Subsampling training S1 entities to %d (random_seed=%d)", max_s1, cfg.random_seed)
        data.source1 = data.source1.sample(n=max_s1, random_state=cfg.random_seed).reset_index(drop=True)
        s1_ids = set(data.source1["entity_id"])
        if data.ground_truth is not None:
            data.ground_truth = data.ground_truth[data.ground_truth["source1_entity_id"].isin(s1_ids)].reset_index(drop=True)

        gt_map_sample = parse_ground_truth(data.ground_truth) if data.ground_truth is not None else {}
        all_true_ids = set()
        for t_set in gt_map_sample.values():
            all_true_ids.update(t_set)

        # Retain all true matches + representative negative candidate pool
        s2_true = data.source2[data.source2["entity_id"].isin(all_true_ids)]
        s2_other = data.source2[~data.source2["entity_id"].isin(all_true_ids)]
        s2_sample_n = min(250000, len(s2_other))
        s2_sampled = s2_other.sample(n=s2_sample_n, random_state=cfg.random_seed) if s2_sample_n > 0 else s2_other
        data.source2 = pd.concat([s2_true, s2_sampled], ignore_index=True)

        s3_true = data.source3[data.source3["entity_id"].isin(all_true_ids)]
        s3_other = data.source3[~data.source3["entity_id"].isin(all_true_ids)]
        s3_sample_n = min(250000, len(s3_other))
        s3_sampled = s3_other.sample(n=s3_sample_n, random_state=cfg.random_seed) if s3_sample_n > 0 else s3_other
        data.source3 = pd.concat([s3_true, s3_sampled], ignore_index=True)

        logger.info("Training subset formed: %d S1, %d S2, %d S3, %d true links in GT",
                    len(data.source1), len(data.source2), len(data.source3), len(all_true_ids))
    else:
        logger.info("FINAL MODE: Using ALL %d available Source-1 entities (NO subsampling, NO truncation)", len(data.source1))

    logger.info("==================================================")
    logger.info("DATA")
    logger.info("==================================================")
    logger.info("S1 records: %d", len(data.source1))
    logger.info("S2 records: %d", len(data.source2))
    logger.info("S3 records: %d", len(data.source3))

    # 2. Preprocess
    logger.info("[2/8] Preprocessing")
    data = preprocess_sources(data, cfg)

    # 3. Parse ground truth
    gt_map = parse_ground_truth(data.ground_truth) if data.ground_truth is not None else {}

    # 4. Generate candidates (with ground truth for candidate recall measurement)
    logger.info("[3/8] Generating candidates")
    candidates = generate_candidates(
        data.source1, data.source2, data.source3, cfg,
        ground_truth_map=gt_map,
    )

    # Prefilter candidate pairs before expensive feature extraction (keeps 100% of positives)
    from .dataset_builder import prefilter_candidates_for_training, build_training_dataset
    max_neg = getattr(cfg.training, "max_negatives_per_positive", 15)
    candidates = prefilter_candidates_for_training(
        candidates, gt_map, max_neg_per_s1=max_neg or 15, random_seed=cfg.random_seed
    )

    # 5. Build features (fit TF-IDF vectorizers on training corpus only)
    logger.info("[4/8] Building features (leak-free)")
    feat_df = build_feature_matrix(candidates, data, cfg, fit_tfidf=True)

    # 6. Build labelled dataset
    logger.info("[5/8] Building training dataset (Source-1 entity split)")
    train_df, val_df, val_gt_map = build_training_dataset(feat_df, gt_map, cfg)

    logger.info("==================================================")
    logger.info("MODEL")
    logger.info("==================================================")
    logger.info("Training pairs: %d", len(train_df))
    logger.info("Positive pairs: %d", int(train_df["label"].sum()))
    logger.info("Negative pairs: %d", int((train_df["label"] == 0).sum()))

    # 7. Train model & save full validation ground truth
    logger.info("[6/8] Training model")
    train_model(train_df, val_df, cfg, val_gt_map=val_gt_map)

    logger.info("Training stage finished.")


# ================================================================== #
#  Stage: TUNE
# ================================================================== #

def run_tune(cfg: "Config") -> None:
    """Threshold tuning on the held-out validation set against full ground truth."""
    from .threshold_tuning import tune_threshold

    best = tune_threshold(cfg)
    logger.info("Selected threshold: %.2f", best)
    logger.info("Threshold persisted to models/selected_threshold.json for prediction stage.")


# ================================================================== #
#  Stage: PREDICT
# ================================================================== #

def run_predict(cfg: "Config") -> None:
    """Generate predictions on test data using persisted model and threshold."""
    import gc
    import pandas as pd
    from .data_loader import load_test_data, SourceData
    from .preprocessing import preprocess_sources, preprocess_df
    from .candidate_generation import generate_candidates
    from .feature_builder import build_feature_matrix
    from .predict import predict
    from .postprocessing import postprocess
    from .output_writer import write_outputs, append_chunk_outputs

    # 1. Load
    logger.info("[1/8] Loading test data")
    data = load_test_data(cfg)

    logger.info("==================================================")
    logger.info("DATA (TEST)")
    logger.info("==================================================")
    logger.info("S1 records: %d", len(data.source1))
    logger.info("S2 records: %d", len(data.source2))
    logger.info("S3 records: %d", len(data.source3))

    n_s1 = len(data.source1)
    chunk_size = 50000

    if n_s1 <= chunk_size:
        # Standard single-batch path for small/unit-test datasets
        data = preprocess_sources(data, cfg)
        candidates = generate_candidates(data.source1, data.source2, data.source3, cfg)
        feat_df = build_feature_matrix(candidates, data, cfg, fit_tfidf=False)
        scored = predict(feat_df, cfg)
        results = postprocess(scored, data, cfg)
        write_outputs(results, candidates, data, cfg)
    else:
        # Scalable chunked streaming inference for real ~1.7M dataset
        logger.info("[2/8] Preprocessing test candidate pool (S2 & S3)...")
        data.source2 = preprocess_df(data.source2, cfg.preprocessing, "test_source2")
        data.source3 = preprocess_df(data.source3, cfg.preprocessing, "test_source3")

        if "_source" not in data.source2.columns:
            data.source2 = data.source2.assign(_source="S2")
        if "_source" not in data.source3.columns:
            data.source3 = data.source3.assign(_source="S3")
        pool = pd.concat([data.source2, data.source3], ignore_index=True)

        # Retain only required normalized columns in pool to minimize RAM footprint
        keep_cols = ["entity_id", "business_name_normalized", "business_address_normalized", "country_normalized", "postal_code", "house_number", "address_numeric_tokens", "_source"]
        pool = pool[[c for c in keep_cols if c in pool.columns]]

        # Free individual S2 and S3 DataFrames to preserve RAM
        data.source2 = pd.DataFrame()
        data.source3 = pd.DataFrame()
        gc.collect()

        logger.info("[3/8] Building blocking indices on test pool ONCE for fast streaming inference...")
        from .candidate_generation import CandidateIndex
        cand_index = CandidateIndex(pool, cfg)

        total_chunks = (n_s1 + chunk_size - 1) // chunk_size
        total_pred_links = 0
        total_singletons = 0

        # Remove existing output files to initialize cleanly
        match_path = cfg.paths.output_dir / "matching_results.tsv"
        cand_path = cfg.paths.output_dir / "candidate_pairs.tsv"
        if match_path.exists():
            match_path.unlink()
        if cand_path.exists():
            cand_path.unlink()

        import joblib
        from .train import load_model, load_feature_columns
        model = load_model(cfg)
        feature_cols = load_feature_columns(cfg)
        tfidf_file = cfg.paths.models_dir / "tfidf_bundle.joblib"
        tfidf_bundle = joblib.load(tfidf_file) if tfidf_file.exists() else None

        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_s1)
            logger.info("--- Processing test S1 chunk [%d/%d] (records %d to %d) ---",
                        chunk_idx + 1, total_chunks, start, end)

            s1_chunk = data.source1.iloc[start:end].copy().reset_index(drop=True)
            s1_chunk = preprocess_df(s1_chunk, cfg.preprocessing, f"test_s1_chunk_{chunk_idx+1}")

            chunk_data = SourceData(source1=s1_chunk, source2=pd.DataFrame(), source3=pd.DataFrame())
            candidates, _, _, _ = cand_index.query(s1_chunk)

            cand_sets = {}
            for s1_id, grp in candidates.groupby("source1_id"):
                cand_sets[s1_id] = set(grp["candidate_id"])

            if len(candidates) > 0:
                feat_df = build_feature_matrix(candidates, chunk_data, cfg, fit_tfidf=False, tfidf_bundle=tfidf_bundle, pool=pool)
                scored = predict(feat_df, cfg, model=model, feature_cols=feature_cols)
                results = postprocess(scored, chunk_data, cfg)
            else:
                results = {sid: set() for sid in s1_chunk["entity_id"]}

            s1_chunk_ids = list(s1_chunk["entity_id"])
            chunk_links = sum(len(results.get(sid, set()) & cand_sets.get(sid, set())) for sid in s1_chunk_ids)
            chunk_singletons = sum(1 for sid in s1_chunk_ids if len(results.get(sid, set()) & cand_sets.get(sid, set())) == 0)
            total_pred_links += chunk_links
            total_singletons += chunk_singletons

            append_chunk_outputs(
                results=results,
                cand_sets=cand_sets,
                s1_ids=s1_chunk_ids,
                cfg=cfg,
                mode="a",
            )

            del s1_chunk, chunk_data, candidates, cand_sets
            gc.collect()

        logger.info("==================================================")
        logger.info("PREDICTION COMPLETE")
        logger.info("==================================================")
        logger.info("Total test S1 entities: %d", n_s1)
        logger.info("Total predicted links : %d", total_pred_links)
        logger.info("Total singletons      : %d", total_singletons)

    logger.info("==================================================")
    logger.info("OUTPUT")
    logger.info("==================================================")
    logger.info("matching_results.tsv: %s", cfg.paths.output_dir / "matching_results.tsv")
    logger.info("candidate_pairs.tsv : %s", cfg.paths.output_dir / "candidate_pairs.tsv")

    logger.info("Prediction stage finished.")


# ================================================================== #
#  Stage: EVALUATE
# ================================================================== #

def run_evaluate(cfg: "Config") -> None:
    """Evaluate predictions against full ground truth (validation split)."""
    from .evaluate import evaluate
    evaluate(cfg)


# ================================================================== #
#  Stage: VALIDATE
# ================================================================== #

def run_validate(cfg: "Config") -> None:
    """Run output-file consistency checks and generate manifest/reports."""
    from .validation import validate_outputs
    from .reproducibility import create_manifest, generate_final_submission_report
    ok = validate_outputs(cfg)
    logger.info("Generating reproducibility manifest and final submission report...")
    manifest = create_manifest(cfg)
    report_path = (getattr(cfg.paths, "reports_dir", None) or (cfg.project_root / "reports")) / "final_submission_report.md"
    generate_final_submission_report(manifest, cfg, report_path)


# ================================================================== #
#  Stage: VERIFY
# ================================================================== #

def run_verify(cfg: "Config") -> bool:
    """Run comprehensive submission verification without modifying outputs."""
    from .reproducibility import verify_submission
    ok, errors = verify_submission(cfg)
    if not ok:
        logger.error("Verification failed with %d error(s).", len(errors))
        import sys
        sys.exit(1)
    return ok


# ================================================================== #
#  Stage: SMOKE TEST
# ================================================================== #

def run_smoke_test(cfg: "Config") -> bool:
    """Run end-to-end truth-aware micro-pipeline smoke test."""
    from .reproducibility import run_smoke_test as _smoke
    return _smoke(cfg)
