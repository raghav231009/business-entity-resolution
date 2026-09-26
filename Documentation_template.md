# Methodology Document — Business Entity Resolution

## 1. Problem Statement

Given three independent data sources describing businesses (Source 1 = master, Source 2 and Source 3 = noisy), determine which Source 2/3 records refer to the same real-world business as each Source 1 entity.

## 2. Methodology Overview

We frame the problem as a **pairwise classification task** with a multi-stage pipeline:

1. **Preprocessing**: Normalize business names, addresses, and country strings
2. **Blocking**: Generate candidate pairs using multiple complementary strategies
3. **Feature Engineering**: Compute ~25 similarity features per pair
4. **Classification**: Train a LightGBM model to predict match probability
5. **Threshold Tuning**: Select the threshold that maximises macro F₀.₅ on validation data
6. **Postprocessing**: Apply safety guards and produce final match sets

## 3. Candidate Generation / Blocking

We use **six blocking strategies** combined via union to maximise recall:

| Strategy | Description |
|----------|-------------|
| Country block | Same normalised country |
| Name token block | Shared name token (inverted index) |
| Name prefix block | Same first N characters of name |
| Address token block | Shared address token (excluding stopwords) |
| Postal code block | Same extracted ZIP/PIN |
| Character n-gram block | Shared character n-grams with minimum threshold |

**Candidate recall** is measured on validation data to ensure true pairs are not lost.

## 4. Feature Engineering

### 4.1 Name Features
- Exact normalised match
- RapidFuzz: ratio, partial ratio, token sort, token set
- Jaccard token similarity
- Character-level Jaccard
- TF-IDF cosine (word n-grams + character n-grams)
- Length and token count differences
- Common token count

### 4.2 Address Features
- Exact normalised match
- RapidFuzz ratio and token sort ratio
- Jaccard token similarity
- Character-level Jaccard
- TF-IDF cosine (word + char n-grams)
- Numeric token overlap
- Postal code equality
- House/building number equality
- Length difference and common token count

### 4.3 Country Features
- Exact normalised country match (binary)

### 4.4 Source Indicators
- Is-Source-2 and Is-Source-3 binary flags

## 5. Model Architecture

- **Primary model**: LightGBM (Gradient Boosted Decision Tree)
- **License**: MIT (compatible with challenge requirements)
- **Class balancing**: `is_unbalance=True` to handle positive/negative imbalance
- **Hyperparameters**: Configured via `config.yaml`

## 6. Training Strategy

- Training pairs are generated using the **same blocking pipeline** as inference
- Labels come from the provided ground truth
- **Entity-level train/validation split** (80/20 of Source 1 IDs) to prevent leakage
- Negative down-sampling to control class imbalance (max 20 negatives per positive)

## 7. Validation Methodology

- **Metric**: Macro-averaged entity-level F₀.₅
- Each Source 1 entity gets its own precision, recall, and F₀.₅
- F₀.₅ is then averaged across all entities
- Singleton entities: correct empty prediction = F₀.₅ of 1.0

## 8. Threshold Tuning

- Sweep thresholds from 0.30 to 0.95
- Select threshold maximising macro F₀.₅ on validation set
- Results saved in `artifacts/metrics/threshold_sweep.json`

## 9. Error Analysis

Error analysis (via notebooks) inspects:
- False positives (wrong merges)
- False negatives (missed matches)
- Hard negatives (similar but different businesses)
- Singleton errors
- Low-confidence and high-confidence mistakes

## 10. Performance Metrics

| Metric | Value |
|--------|-------|
| Macro F₀.₅ | **0.9705** |
| Macro Precision | **0.9794** |
| Macro Recall | **0.9578** |
| Pair-Level Precision | **0.9841** |
| Pair-Level Recall | **0.9566** |
| Candidate Recall (Link-level) | **97.66%** |
| Entity Recall (Coverage) | **99.79%** |
| Blocking Reduction Ratio (Validation) | **99.9981%** |
| Blocking Reduction Ratio (Test) | **99.9998%** |
| Singleton Accuracy | **92.96%** |
| Selected Threshold | **0.90** |

## 11. Limitations

- No external data used (geocoding, business databases, etc.)
- Address parsing is rule-based, not trained on labelled address data
- The model may underperform on countries or address formats not seen in training
- Character n-gram blocking can be noisy for short names

## 12. Reproducibility

- Random seed: 42 (configurable)
- Deterministic train/validation split
- All parameters in `config.yaml`
- Commands: `python run_pipeline.py --stage all`
