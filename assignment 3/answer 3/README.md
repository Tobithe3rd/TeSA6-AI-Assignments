# Milestone 3 — Customer Segmentation

No target label (`Churned` absent). Four numeric features describe 20,000 customers: `AvgOrderValue`, `PurchaseFrequency`, `NumProductCategoriesShoppedIn`, `AvgDiscountUsed`. Unsupervised clustering (K-Means) discovers natural segments.

## Files

- `milestone3_segmentation.ipynb` — Full clustering workflow.
- `dataset used/milestone-3-customer-segments.csv` — Raw dataset (20,000 rows, 4 numeric columns, 0 nulls, 0 duplicates, no label).
- `README.md` — This file.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook milestone3_segmentation.ipynb
```
Dataset path in notebook: `assignment 4/dataset used/milestone-3-customer-segments.csv` (relative to repo root; if running from `assignment 3/`, adjust accordingly).

## Actual results from execution

### Dataset
- Rows: 20,000; Columns: 4; Nulls: 0; Duplicates: 0; No label column.

### k selection (k = 2..10)
- Best by Silhouette (highest): **k = 4** (silhouette = 0.5966)
- Best by Davies-Bouldin (lowest): **k = 4** (DB = 0.5423)
- Metrics agree at k=4; elbow plot supports same region.

### Final model
- KMeans, k=4, `n_init=10`, `random_state=42`.
- GMM comparison: Adjusted Rand Index = 0.9766 (very high agreement = stable clusters).

### Cluster sizes
| Cluster | Count | Percent |
|---------|------:|--------:|
| 0       | 5803  | 29.0%  |
| 1       | 5898  | 29.5%  |
| 2       | 3242  | 16.2%  |
| 3       | 5057  | 25.3%  |

### Feature means per cluster (original units)
| Cluster | AvgOrderValue | PurchaseFrequency | NumProductCategoriesShoppedIn | AvgDiscountUsed |
|---------|--------------:|------------------:|------------------------------:|----------------:|
| 0       | 35.15         | 12.07             | 3.03                          | 0.100           |
| 1       | 199.81        | 2.01              | 1.99                          | 0.050           |
| 2       | 26.38         | 1.31              | 1.12                          | 0.028           |
| 3       | 44.97         | 3.00              | 2.01                          | 0.349           |

### ANOVA (F-statistic, p-value) — all p < 0.001
- AvgOrderValue: F = 114206.65
- PurchaseFrequency: F = 55305.90
- NumProductCategoriesShoppedIn: F = 6611.91
- AvgDiscountUsed: F = 43846.75

### Cluster profiles (auto-generated from actual means)
- Cluster 0: AvgOrderValue=low, PurchaseFrequency=high, NumProductCategoriesShoppedIn=high, AvgDiscountUsed=low | 5803 (29.0%)
- Cluster 1: AvgOrderValue=high, PurchaseFrequency=low, NumProductCategoriesShoppedIn=avg, AvgDiscountUsed=low | 5898 (29.5%)
- Cluster 2: AvgOrderValue=low, PurchaseFrequency=low, NumProductCategoriesShoppedIn=low, AvgDiscountUsed=low | 3242 (16.2%)
- Cluster 3: AvgOrderValue=low, PurchaseFrequency=low, NumProductCategoriesShoppedIn=avg, AvgDiscountUsed=high | 5057 (25.3%)

### Robustness
- Adjusted Rand Index (KMeans vs GMM at k=4): 0.9766

### Placeholders for profile names/descriptions
- **Cluster 0** — Name: *(fill)* — Description: high-frequency, multi-category, low-order-value shoppers with low discounts
- **Cluster 1** — Name: *(fill)* — Description: premium high-value buyers with low frequency and low discounts
- **Cluster 2** — Name: *(fill)* — Description: low-value, low-engagement, minimal-discount customers
- **Cluster 3** — Name: *(fill)* — Description: moderate-value, moderate-frequency, high-discount users

## Outputs saved
- `outputs/customer_segments_labeled.csv` (original df + `Cluster` column)
- `outputs/charts/histograms.png`
- `outputs/charts/correlation_heatmap.png`
- `outputs/charts/elbow_silhouette.png`
- `outputs/charts/davies_bouldin.png`
- `outputs/charts/pca_scatter.png`
- `outputs/charts/feature_boxplots.png`
- `outputs/charts/cluster_profile_heatmap.png`
