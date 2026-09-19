# UrbanCart — Milestone 1: Predicting Next Month Customer Spend

UrbanCart's finance team wants to predict `NextMonthSpend` (a continuous dollar value) per customer to anticipate spend drops. The model uses four features — `MonthsActive`, `AvgOrderValue`, `NumOrdersLastQuarter`, and `Region` — to forecast how much each customer will spend in the coming month. A stronger predictive model lets the team flag at-risk customers earlier and adjust retention or marketing spend accordingly.

## Files

- `milestone-1-regression.ipynb` — Full workflow notebook: exploration, train/test split, two regression models (Linear Regression and Random Forest), evaluation with MAE / RMSE / R², interpretation, recommendation, and feature importance.
- `information used/milestone-1-customer-spend.csv` — Raw dataset (~20,000 rows, 5 columns). No cleaning was needed: zero missing values across all rows, so the CSV was used as-is for modeling.
- `README.md` — This file.

## How to run

1. Install dependencies (all standard scientific-ML packages):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. From this `Assignment 2/` folder, open and run the notebook top to bottom:
   ```bash
   jupyter notebook milestone-1-regression.ipynb
   ```
   The CSV load path (`information used/milestone-1-customer-spend.csv`) is relative to this folder, so the notebook must be run from here.

## Actual results (from held-out test set, n = 4,000)

- **Linear Regression:** MAE $9.85, RMSE $12.33, R² 0.830
- **Random Forest:** MAE $10.56, RMSE $13.15, R² 0.807

**Recommendation:** Linear Regression. It outperforms Random Forest on every metric (lower MAE by $0.71, lower RMSE by $0.82, higher R² by 0.023), and because the simpler model is also fully interpretable (each coefficient directly shows a feature's effect on spend), there is no accuracy-vs-interpretability tradeoff to weigh — the simpler model wins on both counts.

## Key finding — feature importance (Random Forest, actual values)

From the printed feature importance table (`AvgOrderValue`: 0.671, `NumOrdersLastQuarter`: 0.166, `MonthsActive`: 0.096, combined `Region` categories: ~0.066):

- `AvgOrderValue` is by far the strongest driver (~67% of model weight).
- `NumOrdersLastQuarter` is the second most important (~17%).
- `MonthsActive` is third (~10%).
- `Region` contributes very little combined (~6.6%).

This aligns with the regression result: order value and frequency dominate spend prediction, while geography plays a minor role.
