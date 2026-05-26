# Retail Demand Forecasting System

An end-to-end retail demand forecasting system built on the Walmart M5 Competition dataset. This project covers the full data science lifecycle — from data engineering and EDA to feature engineering, modeling, evaluation and deployment.

---

## Live Demo
[Click here to view the deployed dashboard](https://soumya-160405-retail-demand-forecasting.streamlit.app)

---

## Project Overview
Retail demand forecasting is one of the most critical problems in supply chain and operations. This project builds a forecasting system that predicts daily food sales for Walmart's CA1 store using 5 years of historical sales data.

**Business Question:** How many units of a food product will CA1 store sell in the next 28 days?

---

## Dataset
- **Source:** M5 Forecasting Competition (Kaggle)
- **Store:** CA1, California
- **Category:** FOODS
- **Time Period:** January 2011 — April 2016
- **Size:** 2.7 million rows after preprocessing
- **Files used:**
  - `sales_train_validation.csv` — daily sales per product
  - `calendar.csv` — dates, holidays, SNAP days
  - `sell_prices.csv` — weekly product prices

---

## Project Structure

    retail-demand-forecast/
    │
    ├── app/
    │   ├── app.py                  # Streamlit dashboard
    │   └── models/                 # Saved pickle files
    │       ├── lgb_model.pkl
    │       ├── prophet_model.pkl
    │       └── test_data.pkl
    │
    ├── notebooks/
    │   └── 01_data_loading.ipynb   # Full project notebook
    │
    ├── requirements.txt
    └── README.md

---

## Methodology

### Phase 1 — Data Engineering
- Loaded and merged 3 datasets into a single 2.7M row DataFrame
- Handled missing values — event columns filled with No_Event, prices filled with per-item median
- Converted wide format sales data to long format
- Filtered to CA1 store and FOODS category

### Phase 2 — Exploratory Data Analysis
- Total daily sales trend (2011–2016)
- Sales by day of week — Sunday peak confirmed
- Sales by month — Summer demand peak in August
- SNAP day impact — clear sales uplift on payout days
- Holiday impact — store closures drag event day averages down
- Price vs sales — strong inverse relationship confirmed

### Phase 3 — Feature Engineering
19 features engineered from scratch:
- **Lag features:** lag_7, lag_28
- **Rolling statistics:** rolling_mean_7, rolling_mean_28, rolling_std_7
- **Time features:** day_of_week, day_of_month, week_of_year, month, year, is_weekend
- **Price features:** price_change, price_change_pct
- **Event features:** is_event, is_snap, is_superbowl, is_thanksgiving, is_christmas, is_easter

### Phase 4 — Modeling
Three models compared on a chronological train/test split:
- **Train:** February 2011 — December 2015
- **Test:** January 2016 — April 2016 (115 days)

| Model | Description |
|-------|-------------|
| LightGBM | Gradient boosting with all 19 engineered features |
| Prophet | Meta's business forecasting tool — automatic seasonality |
| SARIMA | Classical statistical baseline — order (1,1,1)(1,1,1,7) |

### Phase 5 — Evaluation

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **LightGBM** | **15.66** | **19.80** | **35.56%** |
| Prophet | 22.43 | 27.62 | 38.56% |
| SARIMA | 56.48 | 59.61 | 124.11% |

**LightGBM wins across all metrics.** Feature engineering gave it a decisive advantage over Prophet and SARIMA.

### Phase 6 — Streamlit Dashboard
Interactive dashboard with:
- Model performance KPI cards
- Actual vs predicted sales chart with confidence bands
- Model comparison charts (MAE, RMSE, MAPE)
- Residual analysis and error distribution
- EDA insights (day of week, month, SNAP)
- Feature importance chart
- 28 day future forecast with confidence bands
- Interactive prediction panel — enter custom inputs and get instant predictions

---

## Key Insights
- **Weekly seasonality:** Sunday and Saturday drive 28–37% above average sales
- **Summer peak:** Sales build from June through August driven by back-to-school demand
- **SNAP effect:** Food stamp payout days create a reliable and predictable demand uplift
- **Top features:** rolling_mean_7, lag_7 and rolling_std_7 are the strongest predictors
- **Price elasticity:** Strong inverse relationship — lower priced products dominate volume

---

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas, NumPy | Data engineering |
| Matplotlib, Seaborn | EDA visualizations |
| LightGBM | Primary forecasting model |
| Prophet | Business forecasting model |
| Statsmodels | SARIMA baseline model |
| Scikit-learn | Evaluation metrics |
| Plotly | Interactive dashboard charts |
| Streamlit | Dashboard deployment |

---

## How to Run Locally
```bash
# Clone the repository
git clone https://github.com/soumya-160405/retail-demand-forecasting.git
cd retail-demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
cd app
streamlit run app.py
```

---

## Author
**Soumya Patil**  
Data Analytics Portfolio Project — 2025

