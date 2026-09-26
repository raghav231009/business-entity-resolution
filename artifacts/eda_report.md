# Exploratory Data Analysis (EDA) Report

## 1. Dataset Dimensions
- **Source 1 (Reference)**: 2,206,821 rows × 4 columns
- **Source 2 (Noisy)**: 5,034,616 rows × 4 columns
- **Source 3 (Noisy)**: 5,285,603 rows × 4 columns
- **Ground Truth Pairs**: 2,206,821 rows (total links: 7,638,365)

## 2. Text Attribute Statistics
| Source | Avg Name Length | Avg Address Length | Missing Values |
|---|---|---|---|
| Source 1 | 18.6 chars | 48.9 chars | None |
| Source 2 | 22.0 chars | 44.1 chars | None |
| Source 3 | 22.1 chars | 44.4 chars | None |

## 3. Ground Truth Matching Behavior
- **Total Master Entities**: 2,206,821
- **Singletons (0 matches)**: 123,247 (5.58%)
- **Multi-match Entities**: 2,083,574 (94.42%)
- **Average Matches per Entity**: 3.46 (Median: 3.0)

## 4. Generated EDA Visualizations
1. `eda_country_distribution.png` — Geographical breakdown across all 3 data sources.
2. `eda_length_distribution.png` — Name and address character length distributions.
3. `eda_match_distribution.png` — Histogram of ground-truth link multiplicities.
