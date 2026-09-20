# Milestone 2 — UrbanCart: Predicting Customer Churn (Retention Team)

UrbanCart's retention team wants to flag customers likely to churn before they go quiet, using `MonthsActive`, `AvgOrderValue`, `NumOrdersLastQuarter`, `DaysSinceLastOrder`, and `SupportTicketsFiled`. Only ~7.8% of customers churned (`Churned = 1`), so this is a significantly imbalanced binary classification problem — accuracy alone would mislead (a model predicting "not churned" for everyone scores ~92.2% accuracy while finding zero churners).

## Files

- `milestone-2-churn.ipynb` — Full workflow: exploration, stratified split, two classifiers (Logistic Regression, Random Forest), imbalance-aware evaluation (precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC), recommendation with metric-priority reasoning, and feature importance.
- `dataset used/milestone-2-customer-churn.csv` — Raw dataset (20,000 rows, 6 columns, zero missing values).
- `README.md` — This file.

## How to run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. From `assignment 3/answer 3/`, open and run top to bottom:
   ```bash
   jupyter notebook milestone-2-churn.ipynb
   ```
   The CSV path (`dataset used/milestone-2-customer-churn.csv`) is relative to this folder.

## Actual results (from held-out test set, n = 4,000, stratified ~7.8% churn)

### Metrics per model (positive class = churned)

| Metric            | Logistic Regression | Random Forest |
|-------------------|--------------------:|--------------:|
| Precision (churn) | 0.1971              | 0.5741        |
| Recall (churn)    | 0.6518              | 0.0990        |
| F1 (churn)        | 0.3027              | 0.1689        |
| ROC-AUC           | 0.7670              | 0.7204        |
| PR-AUC            | 0.3520              | 0.2698        |

### Confusion matrices (actual test-set counts)

**Logistic Regression:** TN=2856, FP=831, FN=109, TP=204  
**Random Forest:** TN=3664, FP=23, FN=282, TP=31

### Feature importance (Random Forest, actual values)

From the printed table:

- `DaysSinceLastOrder`: 0.3595 (strongest driver)
- `AvgOrderValue`: 0.2821
- `MonthsActive`: 0.2076
- `NumOrdersLastQuarter`: 0.0981
- `SupportTicketsFiled`: 0.0526

## Why accuracy was not used

With only ~7.8% churned rows, a naive "always predict not churned" model would score ~92.2% accuracy while finding zero actual churners — completely useless for retention. All evaluation here uses precision, recall, F1, confusion matrix, ROC-AUC, and PR-AUC, which meaningfully reflect performance on the rare positive class.

## Metric priority for this use case

A **false negative** (missing an actual churner, so retention never reaches them) is worse than a **false positive** (wasting a retention outreach on someone who stays). Therefore **recall should be weighted more heavily** than precision. The recommendation favors the model that achieves the best balance through F1, backed by higher recall, rather than whichever had marginally higher accuracy.

## Final recommendation

Recommend **Logistic Regression**. Although Random Forest has higher precision (0.574 vs 0.197), it misses almost all real churners — recall is only 0.099 (9.9%) vs Logistic Regression's 0.652 (65.2%). For retention, catching at-risk customers matters more than avoiding a few false alarms. Logistic Regression also achieves higher F1 (0.303 vs 0.169), higher ROC-AUC (0.767 vs 0.720), and higher PR-AUC (0.352 vs 0.270), confirming it separates churners more reliably overall. Its interpretability is an additional benefit: coefficients show which features increase or decrease churn risk directly.
