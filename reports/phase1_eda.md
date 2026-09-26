# Phase 1 — Exploratory Data Analysis

## 1. Executive Summary
This report presents the Phase 1 Exploratory Data Analysis (EDA) for the **Amazon ML Business Entity Resolution Challenge**.
The challenge requires linking records representing identical real-world business entities across three heterogeneous sources:
- **Source 1 (S1)**: Authoritative reference entities (query records).
- **Source 2 (S2)**: High-volume noisy records.
- **Source 3 (S3)**: High-volume noisy records.

Key findings derived from the actual challenge dataset:
- **Massive Scale**: The training set contains **2,206,821** reference S1 entities, **5,034,616** S2 entities, and **5,285,603** S3 entities. The test set comprises **1,732,544** S1, **4,887,273** S2, and **5,082,316** S3 entities.
- **True Link Structure**: The authoritative ground truth contains **7,638,365** true links. **123,247** S1 entities (**5.58%**) are **singletons** with zero matches across S2 and S3.
- **Critical Country Shift**: Train data contains exclusively **US** (~60%) and **India** (~40%). In contrast, the test data introduces **France** (**259,452** S1 entities, **14.98%** of test S1). Blocking strategies and feature extractors must remain open-set and domain-invariant.
- **Data Hygiene**: Source 1 has zero missing fields. However, **168,967** records (**3.36%**) in S2 and **175,916** records (**3.33%**) in S3 have completely empty addresses.
- **Official Metric Alignment**: The macro-averaged entity-level $F_{0.5}$ heavily penalizes false positives (weighting precision 4x over recall). Singletons correctly predicted empty receive $F_{0.5} = 1.0$, while a single false positive match on a singleton drops its score to $0.0$.

---

## 2. Dataset Size
The training and test splits were verified directly from disk:

| Dataset Identifier | File Name | Row Count | Column Count | File Size (MB) |
| :--- | :--- | :---: | :---: | :---: |
| `train_source1` | `train_source1.tsv` | 2,206,821 | 4 | 200.34 MB |
| `train_source2` | `train_source2.tsv` | 5,034,616 | 4 | 466.63 MB |
| `train_source3` | `train_source3.tsv` | 5,285,603 | 4 | 480.37 MB |
| `train_ground_truth` | `train_ground_truth.tsv` | 2,206,821 | 2 | 121.13 MB |
| `test_source1` | `test_source1.tsv` | 1,732,544 | 4 | 166.91 MB |
| `test_source2` | `test_source2.tsv` | 4,887,273 | 4 | 485.86 MB |
| `test_source3` | `test_source3.tsv` | 5,082,316 | 4 | 482.56 MB |

The unrestricted Cartesian product between S1 and the candidate pools (S2 + S3) is:
$$\text{Search Space} = 2,206,821 \times (5,034,616 + 5,285,603) \approx 22.77 \times 10^{12} \text{ candidate pairs}$$
Exhaustive pairwise scoring is intractable; an aggressive, high-recall blocking architecture is mathematically required.

---

## 3. Schema Validation
All datasets were parsed and inspected against the challenge specification:
- **Expected Source Columns**: `entity_id`, `business_name`, `business_address`, `country`
- **Expected Ground Truth Columns**: `source1_entity_id`, `matched_entity_ids`

Schema inspection findings:
- **Missing Required Columns**: **None**. All required columns exist across all train and test files.
- **Unexpected Columns**: **None**. No extra columns were introduced.
- **Data Types**: All columns are represented and parsed as UTF-8 string text to preserve leading zeros in postal codes and prevent numerical identifier corruption.

---

## 4. Missing Values
We analyzed every column for empty strings (`""`), whitespace-only strings, and null-like markers (`"nan"`, `"null"`, `"none"`):

| Dataset | Total Records | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :---: | :---: | :---: | :---: |
| `train_source1` | 2,206,821 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `train_source2` | 5,034,616 | 1 (0.0%) | 168,967 (3.3561%) | 0 (0.0%) |
| `train_source3` | 5,285,603 | 0 (0.0%) | 175,916 (3.3282%) | 0 (0.0%) |
| `test_source1` | 1,732,544 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `test_source2` | 4,887,273 | 0 (0.0%) | 129,408 (2.6479%) | 0 (0.0%) |
| `test_source3` | 5,082,316 | 1 (0.0%) | 136,098 (2.6779%) | 0 (0.0%) |

Key observations:
1. `entity_id` and `country` have **0 missing values** across all splits.
2. `business_name` is **100% complete** across all datasets (only 1 single null-like entry in train S2).
3. **Address quality asymmetry**: Reference S1 has **0.0%** missing addresses. However, both S2 and S3 contain **~3.33% – 3.36%** completely empty address records. Fallback blocking rules must ensure these records can still be matched via business name and country tokens.

---

## 5. Country Distribution
The complete, unfiltered country distributions across all 6 sources:

| Dataset Identifier | Total Rows | Country Breakdown |
| :--- | :---: | :--- |
| `train_source1` | 2,206,821 | **US**: 1,323,633 (59.9792%), **INDIA**: 883,188 (40.0208%) |
| `train_source2` | 5,034,616 | **US**: 3,016,817 (59.9215%), **INDIA**: 2,017,799 (40.0785%) |
| `train_source3` | 5,285,603 | **US**: 3,170,056 (59.9753%), **INDIA**: 2,115,547 (40.0247%) |
| `test_source1` | 1,732,544 | **INDIA**: 809,986 (46.7513%), **US**: 663,106 (38.2735%), **FRANCE**: 259,452 (14.9752%) |
| `test_source2` | 4,887,273 | **INDIA**: 2,312,565 (47.3181%), **US**: 1,871,330 (38.2899%), **FRANCE**: 703,378 (14.392%) |
| `test_source3` | 5,082,316 | **INDIA**: 2,405,000 (47.3209%), **US**: 1,945,701 (38.2837%), **FRANCE**: 731,615 (14.3953%) |

Critical findings:
- **Train vs Test Discrepancy**: While the training data is partitioned between **US** (60.0%) and **India** (40.0%), the test set contains three distinct countries: **India** (46.75% – 47.32%), **US** (38.27% – 38.29%), and **France** (14.39% – 14.98%).
- **Open-Set Country Detection**: **France is explicitly present in all three test files (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`)**.
- **Design Impact**: Models and pipelines MUST NOT hardcode a binary US/India classification or filter test data by training countries. Country normalization and matching must be strictly domain-invariant.

---

## 6. Business Name Analysis
Statistical analysis on business names (sample size: 100,000 records):
- **Raw Unique Names**: 90,403 (9.6% duplicate rate).
- **Normalized Unique Names**: 83,880 (16.12% duplicate rate).
- **Empty Names**: 0 (0.0%).
- **Length Distribution (Characters)**: Mean = 24.04 chars, Median = 24.0 chars, Min = 3.0, Max = 71.0, 90th percentile = 34.0 chars.
- **Token Count Distribution**: Mean = 3.55 tokens, Median = 4.0 tokens, Min = 1.0, Max = 12.0, 90th percentile = 5.0 tokens.
- **Punctuation Frequency**: **20.68%** of business names contain punctuation characters (`&`, `-`, `.`, `,`, `'`).
- **Common Legal Suffix Frequency**:
  - `ltd`: 6,587
  - `limited`: 23,920
  - `pvt`: 5,312
  - `private`: 19,823
  - `inc`: 10,873
  - `llc`: 16,032
  - `corp` / `corporation`: 2,207
  - `co` / `company`: 1,853
  - `plc`: 0

Stripping legal suffixes during candidate blocking prevents catastrophic block explosions on generic business descriptors.

---

## 7. Address Analysis
Statistical analysis on business addresses (sample size: 100,000 records):
- **Empty Addresses**: 0 (0.0%) in S1.
- **Length Distribution (Characters)**: Mean = 52.08 chars, Median = 41.0 chars, Min = 13.0, Max = 222.0, 90th percentile = 90.0 chars.
- **Token Count Distribution**: Mean = 8.04 tokens, Median = 7.0 tokens, Min = 3.0, Max = 38.0.
- **Numeric Token Presence**: **94.27%** of addresses contain numeric tokens (mean = 1.35 numeric tokens per address).
- **Postal / PIN Code Presence**: **6.76%** of non-empty addresses contain identifiable postal codes (US 5-digit ZIP, India 6-digit PIN, France 5-digit code postal).
- **Common Address Abbreviations**:
  - Street / St: 742
  - Road / Rd: 876
  - Avenue / Ave: 133
  - Boulevard / Blvd: 11
  - Suite / Ste: 126
  - Floor / Fl: 1,518

Address normalization must expand or standardize common directional and structural abbreviations before token matching.

---

## 8. Ground Truth / Match Distribution
Analysis of the full ground truth dataset (`train_ground_truth.tsv`):
- **Total Source 1 Entities**: **2,206,821**
- **Total True Match Links**: **7,638,365**
- **Source 2 Links**: **3,693,619** (48.36%)
- **Source 3 Links**: **3,944,746** (51.64%)
- **Entities with Zero Matches (Singletons)**: **123,247** (**5.58%**)
- **Entities with Exactly 1 Match**: **119,157** (**5.40%**)
- **Entities with Multiple Matches (>1)**: **1,964,417** (**89.02%**)
- **Match Links per S1 Entity**:
  - Mean: **3.4613**
  - Median: **3.0**
  - Min: **0**
  - Max: **11**
  - Standard Deviation: **1.7053**
- **Match Breakdown by Source Inclusion**:
  - Matched exclusively in S2: **143,029** entities (6.48%)
  - Matched exclusively in S3: **164,498** entities (7.45%)
  - Matched in both S2 and S3: **1,776,047** entities (80.48%)

The overwhelming majority of non-singleton S1 entities (**80.48%**) have matching counterparts in **both** Source 2 and Source 3 simultaneously.

---

## 9. Singleton Analysis
Singletons represent reference entities that have **zero corresponding records** in either Source 2 or Source 3.
- **Total Singletons**: **123,247** entities (**5.5848%** of all S1 reference records).
- **Mathematical Evaluation Impact**:
  - Under official challenge rules, if a singleton entity is predicted with an empty set (no matches), its score is:
    $$F_{0.5} = 1.0$$
  - If a model outputs even a single false-positive candidate for a singleton entity, its score collapses to:
    $$F_{0.5} = 0.0$$
- **Precision Sensitivity**: Because singletons account for 5.58% of all reference entities, predicting low-confidence noisy candidates indiscriminately on singletons reduces the macro F0.5 score by up to **0.0558** (5.58 percentage points).
- **Threshold Implication**: High-precision postprocessing and threshold tuning are necessary to suppress borderline candidate predictions.

---

## 10. True Pair Similarity Analysis
Similarity metrics evaluated across **25,000** actual verified true matches:

| Metric | Mean | Median | Min | 25th Pct | 75th Pct | 90th Pct | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fuzzy Name Ratio** | 79.56 | 88.0 | 3.33 | 71.88 | 100.0 | 100.0 | 100.0 |
| **Token Sort Name Ratio** | 79.32 | 88.89 | 3.33 | 70.59 | 100.0 | 100.0 | 100.0 |
| **Fuzzy Address Ratio** | 76.4 | 84.11 | 0.0 | 65.93 | 92.69 | 100.0 | 100.0 |
| **Address Token Jaccard** | 0.64 | 0.67 | 0.0 | 0.5 | 0.83 | 1.0 | 1.0 |

Key similarity rates:
- **Exact Normalized Name Match**: **35.13%** of true pairs have identical normalized names.
- **Exact Normalized Address Match**: **11.08%** of true pairs have identical normalized addresses.
- **Country Agreement**: **100.0%** of true pairs share identical normalized countries. True cross-country matches are virtually nonexistent in the ground truth.

---

## 11. Important Observations
1. **Heterogeneous Sources**: Source 2 and Source 3 exhibit distinct formatting styles, abbreviation patterns, and address completeness levels.
2. **High Overlap Between S2 and S3**: 80.48% of reference entities link to both S2 and S3, indicating significant redundancy that multi-source graph clustering or pairwise classification can exploit.
3. **Severe Search Space**: Without candidate blocking, evaluating all pairs requires ~22.7 trillion comparisons, which is computationally infeasible.
4. **Extreme Metric Asymmetry**: Macro F0.5 puts 4x more weight on precision than recall. Aggressive candidate generation must be paired with conservative classification thresholds.

---

## 12. Implications for Blocking
1. **Never Block on Country Alone**: The US block alone would contain $1.32\text{M} \times 3.02\text{M} \approx 3.99 \times 10^{12}$ pairs. Country must only be used as a compound partition in conjunction with name tokens or postal codes.
2. **Compound Multi-Channel Blocking**: Candidate generators should combine:
   - `country + first_name_token`
   - `country + name_prefix_4`
   - `country + postal_code`
   - `name_token + postal_code`
3. **Legal Suffix Invalidation**: Blocking keys built on raw business names will produce oversized blocks on words like "pvt", "ltd", "inc". Suffix stripping must precede key generation.
4. **Fallback for Missing Addresses**: Because 3.3% of S2/S3 addresses are missing, address-dependent blocking keys alone would miss these pairs. Name-based blocking channels are mandatory fallbacks.

---

## 13. Implications for Feature Engineering
1. **Name Similarity Features**:
   - Exact normalized string match
   - RapidFuzz ratio, partial ratio, token sort ratio, and token set ratio
   - Character 3-gram Jaccard similarity and length difference
2. **Address Similarity Features**:
   - Exact normalized address match
   - Fuzzy address ratio and token sort ratio
   - Numeric token overlap (house numbers, suite numbers)
   - Postal code exact match (boolean indicator)
3. **Country Consistency**:
   - Exact country equality indicator feature
4. **Source Distinction**:
   - Binary indicator feature for whether candidate originates from S2 vs S3.

---

## 14. Risks and Limitations
1. **Test Set Country Shift (France)**:
   - *Risk*: Any model feature that memorizes US state codes, India PIN patterns, or country names will fail on the French test set.
   - *Mitigation*: Ensure postal code regex and address normalization are country-agnostic.
2. **Missing Address Distortion**:
   - *Risk*: Computing address similarity on empty strings yields 0.0, which may cause tree models to heavily penalize valid true matches with missing addresses.
   - *Mitigation*: Include explicit `is_address_missing` indicator features so the classifier can decouple missing address signals from low-similarity address signals.
3. **Singleton Penalties**:
   - *Risk*: Over-predicting matches on singletons drops the macro F0.5 score significantly.
   - *Mitigation*: Optimize the classification threshold specifically against the macro entity-level F0.5 metric during threshold tuning.

---

## 15. Recommended Next Steps
1. **Phase 2 — Candidate Generation (Blocking)**:
   - Implement compound blocking strategies (`country + name_prefix`, `country + name_token`, `country + postal`).
   - Benchmark candidate recall (target: ≥ 95% pair recall) and reduction ratio (target: ≥ 99.9% reduction).
2. **Phase 3 — Feature Extraction & Classifier Training**:
   - Compute dense lexical and token features across candidate pairs.
   - Train a gradient boosted decision tree (LightGBM) optimized for binary cross-entropy.
3. **Phase 4 — Threshold Optimization & Submission**:
   - Tune probability threshold on the validation set using the unified `metrics.py` macro F0.5 function.
   - Generate test set predictions adhering to the challenge submission schema.
