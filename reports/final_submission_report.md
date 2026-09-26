# Final Submission Report

## 1. Dataset
The complete official challenge dataset was processed:
- **Train Source 1 (Reference)**: 200.34 MB (591af0e1dfeb65ca...)
- **Train Source 2 (Noisy)**: 466.63 MB (6336c1a055eec79c...)
- **Train Source 3 (Noisy)**: 480.37 MB (67da22f5151898ff...)
- **Train Ground Truth**: 121.13 MB (70bc1d8a16c667e0...)
- **Test Source 1**: 166.91 MB (3d4a32c54c2ca9c5...)
- **Test Source 2**: 485.86 MB (79d906c7497af2ac...)
- **Test Source 3**: 482.56 MB (850942b11d2a4343...)

## 2. Training
- **Execution Mode**: `final`
- **Training S1 Entities Limit**: `None`
- **Validation S1 Entities Limit**: `None`
- **Model Type**: `lightgbm`
- **Saved Model File**: `entity_matcher.joblib` (SHA-256: `ab38ed238f3a83f6...`)
- **Negative Sampling**: Configured max negatives per positive = 20, prioritizing hard negatives.

## 3. Blocking
- **Blocking Strategies Active**:
  - `country + first_name_token`
  - `country + name_prefix_4`
  - `country + postal_code`
  - `name_token + postal_code`
  - `name_prefix + postal_code`
- **Candidate Recall (Link-Level)**: **1.0** (≥ 97.6% true links covered)
- **Candidate Safety Cap**: max 50 candidates per S1 entity with similarity pre-ranking.

## 4. Validation
- **Evaluation Metric**: Exact Challenge Macro-averaged Entity-Level $F_{0.5}$ with exact singleton handling ($F_{0.5} = 1.0$ for true singletons predicted empty, $0.0$ for false positives).
- **Tuned Threshold**: **0.9**
- **Validation Macro F0.5**: **0.9705**

## 5. Test Prediction
- **Test Source 1 Entities**: **1,732,544**
- **Total Predicted Links**: **3,711,726**
- **Predicted Singletons**: **840,175** (48.49%)
- **Test Set Countries Covered**: US, India, and France (open-set country).

## 6. Submission Validation
- **Validator Result**: **PASS**
- **Format**: Strictly tab-separated TSV without index or quotes.
- **Headers**:
  - `matching_results.tsv`: `source1_entity_id	matched_entity_ids`
  - `candidate_pairs.tsv`: `source1_entity_id	candidate_entity_ids`
- **Entity Consistency**: Exactly one row per test S1 entity ID; no duplicates; no omissions; no extras.
- **Strict Subset Property**: $\text{predicted\_matches} \subseteq \text{candidate\_entity\_ids}$ holds for 100% of test entities.
- **Target Membership**: All predicted candidate IDs strictly belong to the test Source-2/Source-3 union.

## 7. Reproducibility
- **Timestamp (UTC)**: `2026-09-26T12:42:41.236427+00:00`
- **Source Code SHA-256**: `f1621eaf12d9b2d22d7828d643ea61e139fe4b7cc636f4782ca5cb0587faa927`
- **Config YAML SHA-256**: `0a8407f650a696d73277fd87b347871945fa12d79a87c690c2acdf4d032537a0`
- **Git Commit**: `none (clean standalone export)`
- **Python**: `3.13.4`
- **Platform**: `Windows-11-10.0.26200-SP0`
- **Key Packages**:
  - `pandas`: 2.3.0
  - `numpy`: 2.3.0
  - `scikit-learn`: 1.9.1
  - `rapidfuzz`: 3.14.6
  - `lightgbm`: 4.7.0
  - `joblib`: 1.6.0
  - `pyyaml`: 6.0.2
  - `pytest`: 9.1.1
- **Output Files**:
  - `matching_results.tsv`: SHA-256 `8b83d55cebec1f0ba1c37e999e7d9d03d5eb9c23bd4f2ac90aa0f4455e995a75`
  - `candidate_pairs.tsv`: SHA-256 `dd61c601b630c3769fc2d30da17cb7aeea8d8c96dcf0d2342b31870cfd480156`
