# Primetrade.ai Internship Round 0

## What this is

I built a notebook that looks at whether the Bitcoin Fear & Greed Index actually tells us anything useful about how traders perform on Hyperliquid. The data is small (~100 account-day rows after aggregation) so I had to be careful with the ML side of things.

## Files

- `intern3.ipynb` - the main notebook, runs on Google Colab
- `fear_greed_index.csv` - Bitcoin Fear & Greed Index data (upload to `/content/` in Colab)
- `historical_data.csv` - Hyperliquid trader data (upload to `/content/` in Colab)

## What the notebook does

1. **Data cleaning** - parses timestamps (the sentiment data uses unix seconds, trader data uses milliseconds), handles missing values, merges on date
2. **Feature engineering** - daily aggregation per account, rolling 3d/7d windows for PnL and win rate, interaction features (FGI x leverage, etc.)
3. **EDA** - win rate by sentiment, PnL distributions, heatmaps across leverage/volume segments, time series plots
4. **Statistical tests** - Kruskal-Wallis, Mann-Whitney U, Spearman correlation, bootstrap confidence intervals. Non-parametric throughout since the data isn't normally distributed
5. **ML modeling** - tried Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and a soft voting ensemble. Used SMOTE to handle class imbalance, LOO-CV for honest evaluation on small data, and bootstrap AUC CIs
6. **Interactive dashboard** - Plotly dashboard with FGI over time, win rates, PnL boxes, L/S ratio
7. **Trading rules** - 5 data-driven rules covering position sizing, leverage management, volume filters, model signals, and rolling performance monitoring

## How to run

1. Open `intern3.ipynb` in Google Colab
2. Upload `fear_greed_index.csv` and `historical_data.csv` to `/content/`
3. Run all cells

Plots get saved to `/content/outputs/`.

## Tech stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- XGBoost, Scikit-learn, imbalanced-learn (SMOTE)
- SciPy (stats)
