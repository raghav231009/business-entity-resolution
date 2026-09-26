# Business Entity Resolution

> A modular ML pipeline for matching business records across multiple noisy data sources — built for the Amazon ML Challenge.

---

## 📋 Problem Statement

Given three independent data sources describing businesses:

- **Source 1** = deduplicated reference/master source
- **Source 2** = noisy source
- **Source 3** = noisy source

For every Source 1 entity, find all matching Source 2 and Source 3 entities. Records may have noisy names, addresses, and countries with abbreviations, typos, token reordering, transliterations, and missing components.

---

## 🏗️ Architecture

```
RAW DATA → LOAD → PREPROCESS → BLOCK → CANDIDATE GENERATION
    → FEATURE ENGINEERING → ML MODEL → THRESHOLD → POST-PROCESS
    → matching_results.tsv + candidate_pairs.tsv → VALIDATION
```

Each step lives in its own Python module. You can change **blocking** without touching features, change **features** without touching training, change the **model** without touching blocking, and change the **threshold** without retraining.

---

## 📁 Folder Structure

```
business_entity_resolution/
│
├── config.yaml                    # All settings (paths, model params, thresholds)
├── requirements.txt               # Python dependencies
├── run_pipeline.py                # CLI entry point
├── README.md                      # This file
├── Documentation_template.md      # Methodology writeup
│
├── data/
│   ├── train/                     # Place training TSVs here
│   └── test/                      # Place test TSVs here
│
├── models/                        # Saved model + metadata
├── artifacts/
│   ├── candidate_cache/           # Cached candidates
│   ├── feature_cache/             # Cached feature matrices
│   └── metrics/                   # Experiment results
│
├── output/
│   ├── matching_results.tsv       # Final submission
│   └── candidate_pairs.tsv        # Candidate set
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   └── 02_error_analysis.ipynb    # Error inspection
│
├── src/business_entity_resolution/
│   ├── config.py                  # Config loader (dataclasses)
│   ├── data_loader.py             # TSV loading + column validation
│   ├── preprocessing.py           # Name/address/country normalisation
│   ├── blocking.py                # 6 blocking strategies
│   ├── candidate_generation.py    # Multi-strategy union + stats
│   ├── ground_truth.py            # Parse ground-truth labels
│   ├── features.py                # Individual feature functions
│   ├── feature_builder.py         # Build feature matrix + TF-IDF
│   ├── dataset_builder.py         # Labelled train/val split
│   ├── train.py                   # LightGBM / CatBoost training
│   ├── evaluate.py                # Entity-level macro F₀.₅
│   ├── threshold_tuning.py        # Threshold sweep
│   ├── predict.py                 # Score candidates
│   ├── postprocessing.py          # Threshold + safety guards
│   ├── output_writer.py           # Write TSV files
│   ├── validation.py              # Output consistency checks
│   └── pipeline.py                # Stage orchestrator
│
└── tests/
    └── test_core.py               # Unit tests
```

---

## ⚙️ Installation & Environment Setup

This project requires Python 3.9+ (tested on Python 3.13) and pinned, reproducible dependencies.

```bash
# 1. Clone / navigate to project root
cd d:\Amazon ML\business_entity_resolution

# 2. Create and activate a clean virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify environment dependencies
python run_pipeline.py --stage env-check
```

---

## 📊 Dataset Placement

Place the challenge TSV files like this:

```
data/train/train_source1.tsv
data/train/train_source2.tsv
data/train/train_source3.tsv
data/train/train_ground_truth.tsv
data/test/test_source1.tsv
data/test/test_source2.tsv
data/test/test_source3.tsv
```

---

## 🚀 Execution Modes

The pipeline explicitly distinguishes between fast development experiments and full-scale challenge submissions:

### Mode A — Development (`--mode dev`, Default)
- **Purpose**: Fast iteration, code debugging, feature engineering experiments, and unit/integration testing.
- **Limits**:
  - `training.max_train_s1_entities`: 50,000
  - `training.max_validation_s1_entities`: 10,000
- **Sampling**: Ground-truth-aware sampling (preserves true matching links and negative distributions).

```bash
# Fast development pipeline run
python run_pipeline.py --mode dev --stage all

# Fast representative micro-pipeline smoke test (~3-5 seconds)
python run_pipeline.py --mode dev --stage smoke-test
```

### Mode B — Final (`--mode final`)
- **Purpose**: Official competition run and final submission generation.
- **Limits**: Strictly **NONE** (`max_train_s1_entities: null`, `max_validation_s1_entities: null`).
- **Data Universe**: Uses 100% of eligible training Source-1 entities (2.2M) and the complete validation population.
- **Audit Trail**: Produces `artifacts/final_run_manifest.json` and `reports/final_submission_report.md` proving code/config/model/data synchronization.

```bash
# Full competition pipeline run
python run_pipeline.py --mode final --stage all

# Verify submission files against manifest without modifying outputs
python run_pipeline.py --mode final --stage verify
```

---

## 🏃 Pipeline Stages

You can run the full end-to-end pipeline or individual stages:

| Stage | Command | Description |
|---|---|---|
| `env-check` | `python run_pipeline.py --stage env-check` | Verify required dependencies & versions |
| `smoke-test` | `python run_pipeline.py --mode dev --stage smoke-test` | Fast end-to-end test on isolated temp paths |
| `eda` | `python run_pipeline.py --stage eda` | Run exploratory data analysis & generate reports |
| `train` | `python run_pipeline.py --mode dev --stage train` | Train entity matching model (LGBM) |
| `tune` | `python run_pipeline.py --stage tune` | Sweep thresholds on full validation set for macro $F_{0.5}$ |
| `predict` | `python run_pipeline.py --stage predict` | Generate test predictions & candidate TSVs |
| `evaluate` | `python run_pipeline.py --stage evaluate` | Evaluate on validation set against full ground truth |
| `verify` | `python run_pipeline.py --mode final --stage verify` | Comprehensive submission audit & constraint check |
| `all` | `python run_pipeline.py --mode dev --stage all` | Run all stages sequentially |

Custom config file can be passed to any stage:
```bash
python run_pipeline.py --stage all --config my_config.yaml
```

---

## 🔧 How Each Component Works

### Preprocessing
- Unicode normalisation (NFKD) + accent stripping
- Lowercase everything
- Replace `&` with `and`
- Strip legal suffixes (`Ltd`, `Inc`, `LLC`, …)
- Expand address abbreviations (`St` → `Street`, …)
- Extract postal codes, house numbers, numeric tokens
- All configurable via `config.yaml`

### Blocking (Multi-Channel Candidate Generation)
Informative single-channel and compound strategies whose union produces the candidate set:
1. **Compound Country + Name token** — highly informative, avoids country-alone explosion
2. **Compound Country + Name prefix** — handles variations with shared prefix
3. **Compound Country + Postal code** — geographic anchor within country
4. **Compound Country + Address token** — street/area anchor within country
5. **Compound Name token + Postal code** — strong cross-source identifier
6. **Compound Name prefix + Postal code** — robust to name suffix/spelling noise
7. **Compound Name token + Address token** — cross-field token co-occurrence
8. **Name token block** — shared business name tokens (min len 2)
9. **Name prefix block** — first N characters of normalised name
10. **Postal code block** — extracted postal / ZIP / PIN code
11. **Address token block** — non-stopword address tokens
12. **Character n-gram block** — shared character n-grams with min shared threshold
13. **Similarity-based pre-ranking** — deterministic similarity ranking before capping (never arbitrary ID truncation)
14. **Candidate recall tracking** — measured and logged directly against ground truth (`artifacts/metrics/candidate_recall.json`)

### Features (~25 features per pair)
| Category | Features |
|----------|----------|
| **Name** | Exact match, fuzzy ratio, partial ratio, token sort, token set, Jaccard, char sim, TF-IDF cosine (word + char via leak-free fitted bundle), length diff, token count diff, common tokens |
| **Address** | Exact match, fuzzy ratio, token sort, Jaccard, char sim, TF-IDF cosine (word + char via leak-free fitted bundle), length diff, common tokens, numeric overlap, postal match, house # match |
| **Country** | Exact match |
| **Source** | Is-S2, Is-S3 indicators |

### Model
- Primary: **LightGBM** classifier with balanced class weights
- Alternative: CatBoost (swap via `config.yaml`)
- Trained on pairs partitioned at the Source-1 entity level to prevent leakage

### Evaluation
- **Macro-averaged entity-level F₀.₅** (exact challenge metric) against **FULL ground truth**
- True matches missed by blocking remain in the ground truth and receive proper recall penalties
- Singleton-aware: empty prediction for true singleton = F₀.₅ of 1.0; false match = 0.0

### Threshold Tuning
- Sweeps configurable threshold values on validation predictions against FULL ground truth
- Selects threshold maximising entity-level macro F₀.₅
- Persists selected threshold to `models/selected_threshold.json`
- Prediction stage automatically loads the tuned threshold from `models/selected_threshold.json`
- Sweep results saved to `artifacts/metrics/threshold_sweep.json`

---

## 🧪 Running Tests & Quality Verification

Run the comprehensive unit, regression, and pipeline test suite:

```bash
# Verify environment and pinned dependencies
python run_pipeline.py --stage env-check

# Run all 68+ unit, hardening, and regression tests
python -m pytest tests/ -q
```

The test suite validates:
- Exact challenge macro $F_{0.5}$ metric computation and singleton scoring
- Full ground truth evaluation (unretrieved true matches remain penalized)
- Candidate recall calculation and explicit `NOT_AVAILABLE` state when zero true links exist
- Candidate cap tradeoff logic and deterministic similarity ranking (no arbitrary ID truncation)
- Leak-free TF-IDF fitting on training split only and transform-only on validation/test
- Submission format, exact column headers, TSV tab delimiters, and subset consistency ($\text{matches} \subseteq \text{candidates}$)
- Reproducibility manifest generation, git commit provenance, and working-tree dirty detection

---

## 🔬 Core Methodology & Hardening Details

### 1. Leak-Free TF-IDF Pipeline
Unlike naive implementations that fit vectorizers across the entire dataset, this pipeline strictly splits Source-1 entities first:
`FULL TRAIN DATA -> SOURCE-1 SPLIT -> TRAIN (fit TF-IDF) -> transform TRAIN / transform VALIDATION`.
Test inference loads the persisted TF-IDF bundle (`models/tfidf_bundle.joblib`) and applies `transform()` only.

### 2. Candidacy & Justified Candidate Cap
- **Multi-channel Blocking**: Evaluates 12+ compound and token blocking keys across normalized fields.
- **Empirical Cap Selection**: Evaluated $K \in [25, 50, 75, 100, 150]$ (`reports/candidate_cap_experiment.md`). $K=50$ captures $85.99\%$ candidate recall with optimal pair density and fast runtime ($0.27$s), while $K=75$ only adds $+0.55\%$ recall at $+43\%$ pair cost.
- **Informative Pre-ranking**: Candidates exceeding $K$ are sorted by composite similarity (token Jaccard, fuzzy ratio, postal match), never alphabetical/ID truncation.
- **Honest Candidate Recall**: Computed strictly on held-out validation sets with known ground truth. If evaluated on zero true links, reports `NOT_AVAILABLE` with explicit diagnostics rather than vacuous `1.0`.

### 3. Evaluation & Threshold Optimization
- **Full Ground Truth**: Evaluated against the complete validation ground truth (`models/val_ground_truth.json`). Candidates missed by blocking are never dropped from truth.
- **Macro-averaged Entity-Level $F_{0.5}$**: Weights precision over recall ($\beta=0.5$). Singletons predicted empty receive $1.0$; false positive matches on singletons receive $0.0$.
- **Tuned Decision Boundary**: Automatically sweeps thresholds to optimize macro $F_{0.5}$ and persists to `models/selected_threshold.json`.

### 4. Git Provenance & Reproducibility Manifest
`artifacts/final_run_manifest.json` records:
- Git repository URL, commit hash, branch, and working-tree clean/dirty status
- SHA-256 hashes of config YAML, source code tree, model weights, dataset files, and output TSVs
- Pinned dependency versions (`requirements.txt`)
- Execution mode (`dev` vs `final`) and timestamps

---

## 🛡️ Important Rules

1. **No external data**: No geocoding APIs, Google Maps, web scraping, or business databases
2. **Open-set country**: Never hard-code country lists — country is treated as a string feature
3. **Model license**: MIT/Apache-2.0 compatible, ≤8B parameters
4. **Reproducibility**: All randomness uses `random_seed` from config

---

## 📄 Output Files

### `output/matching_results.tsv`
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

### `output/candidate_pairs.tsv`
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
```

---

## 📝 License

All code is original. Model uses LightGBM (MIT license).
