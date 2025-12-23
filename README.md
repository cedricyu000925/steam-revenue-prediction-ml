# 🎮 Steam Game Revenue Prediction – End-to-End Analytics & ML Project

> **Goal:** Build a realistic, end‑to‑end analytics and ML project that answers real business questions about Steam game revenue, using SQL, BigQuery, Python, XGBoost, SHAP, Looker Studio, and a deployed Flask API.

---

## 1. Business Problem & Context

The Steam store hosts thousands of games with widely varying prices, genres, discount strategies, and platform support. Publishers and product teams face questions such as:

- Which **game characteristics** (genre, price, platform support, DLC, discounts) drive **higher revenue**?
- How should we **price** and **discount** games to maximize long‑term revenue instead of just short‑term sales spikes?
- Which **genres and platforms** are worth prioritizing for future releases?

This project builds a **revenue prediction and analysis system** to:

- Quantify how different features impact a game’s **estimated revenue**
- Provide **actionable recommendations** for pricing and discount strategies
- Offer an **API** that product or marketing tools can call to estimate revenue for new or existing games

> 🔧 Data is based on a real Steam dataset and engineered **revenue proxy**, approximating revenue using available fields (e.g. price and engagement‑related proxies).

---

## 2. Project Architecture (End‑to‑End Flow)

**High‑level pipeline:**

1. **Data ingestion & cleaning** in Google Colab (Python, pandas).
2. **Upload to Google BigQuery** for scalable SQL analytics.
3. **Exploratory Data Analysis (EDA)** and business insights using BigQuery + Python.
4. **Feature engineering & model training** (XGBoost regression).
5. **Model evaluation & explainability** (SHAP, per‑genre performance).
6. **Dashboards** in Google Looker Studio for business users.
7. **Model deployment** as a Flask API (running locally, designed for cloud deployment).
8. **Performance optimization & latency measurement** for the API model.

---

## 3. Data, EDA & Key Insights

### 3.1 Data Sources

- **Primary dataset:** Steam game metadata (title, genre, price, tags, platform support, etc.).
- **Rows:** **42497**.
- **Time coverage:** **From year 1997 to 2024**.
- **Target:** Engineered **revenue proxy** based on price and proxies for popularity/engagement.

> The full raw dataset is not included in the repo for size/licensing reasons; a **small sample** and the **data dictionary** are provided for reference. **(https://www.kaggle.com/datasets/amanbarthwal/steam-store-data)**

### 3.2 EDA (Exploratory Data Analysis)

EDA was done in a combination of **BigQuery SQL** and **Python notebooks** to answer:

- How is **price** distributed across genres?
- Which **genres** dominate the catalog vs. which dominate **revenue**?
- How do **discounts** vary by genre and price segment?
- Are there obvious **data quality issues** (duplicates, missing values, outliers)?

> <img width="1784" height="692" alt="Screenshot 2025-12-19 133715" src="https://github.com/user-attachments/assets/22dea623-c5c9-4f00-80f7-38f383fffa98" />

> <img width="1199" height="801" alt="Screenshot 2025-12-19 133821" src="https://github.com/user-attachments/assets/df9dec17-0e8a-430f-98e3-55b2ebd5951c" />

---

## 4. Google Cloud & BigQuery

The cleaned dataset was uploaded to **Google BigQuery** to leverage:

- **Scalable SQL** for large‑table analysis.
- **Reusable views** for dashboards and modelling.
- **Separation** between raw, cleaned, and feature tables.

> <img width="1808" height="720" alt="Screenshot 2025-12-19 134605" src="https://github.com/user-attachments/assets/c11b12b3-0633-4d65-a931-759fd1dca60d" />

---

## 5. Dashboards (Google Looker Studio)

Business‑friendly dashboards were built in **Google Looker Studio** to let non‑technical stakeholders explore:

- Overall **estimated revenue** by **genre**, **engagement (in the form of reviews)**, and **price tiers**.
- Trends by **release year** and **publisher**.

### 5.1 Main Dashboard

>   <img width="1075" height="807" alt="Screenshot 2025-12-19 140402" src="https://github.com/user-attachments/assets/a7b78b29-d0f2-40e7-9a39-7deb3154c07a" />
>   “Executive Overview - At-a-glance visualization of the dataset used”

### 5.2 Pricing and Revenue Page

>   <img width="1077" height="805" alt="Screenshot 2025-12-19 141735" src="https://github.com/user-attachments/assets/98bf0c04-383c-42c9-9a9e-c13fae8efb0c" />
> “Performance of different games - Price and Median Review Counts”

### 5.3 Engagement Drivers
>   <img width="1076" height="808" alt="Screenshot 2025-12-19 141936" src="https://github.com/user-attachments/assets/44e3fa3e-d680-47a6-a4bb-62aeb9207f56" />
> “How diffrent features impact the engagement of games of different price tiers?”

> **Live dashboard:** **https://lookerstudio.google.com/reporting/9269dbaf-6611-4939-b230-7edd85007020**

---

## 6. Modelling – XGBoost Revenue Predictor

### 6.1 Problem Framing

- **Task:** Regression – predict **log(revenue proxy)**, then transform back to the original scale.
- **Unit of prediction:** One game.
- **Inputs:**
  - Price attributes: `final_price`, `discount_pct_clean`, `is_on_sale`.
  - Platform flags: `win_support`, `mac_support`, `linux_support`, `multi_platform`.
  - Content attributes: `has_dlc`, `dlc_available`, `is_early_access`, `is_free_to_play`.
  - Encoded genres: one‑hot `genre_*` features.

### 6.2 Model & Features

Model trained in **Google Colab** using **XGBoost**:

- **Algorithm:** `XGBRegressor`.
- **Key hyperparameters:**
  - `n_estimators`: **300**
  - `max_depth`: **6**
  - `learning_rate`: **0.01**
  - `subsample`: **0.7**
  - `colsample_bytree`: **0.9**

> **Tuning approach:** Hyperparameters were selected using **GRID SEARCH HYPERPARAMETER TUNING**

### 6.3 Performance

- **R²:** 0.3135
- **RMSE (₹):** ₹1001.9M
- **MAE (₹):** ₹103.2M

- Model performs best on **casual** games; struggles more with **RPG**

## 7. Model Deployment – Flask API

The trained XGBoost model is deployed via a **Flask API** (non‑notebook Python app) that can run in a terminal and be integrated into other services.

## 8. Optimization & Latency Measurement

A key part of this project is treating the model like a **real service**, not just a notebook experiment.

### Latency Testing

In `test_latency.py`, the API is called repeatedly for multiple game scenarios to measure:

- **Mean latency**.
- **P95 latency**.

Results (per game):

- **Average XGBoost latency:** 2.862.
- **P95 latency:** 6.978.

> <img width="511" height="196" alt="Screenshot 2025-12-23 194830" src="https://github.com/user-attachments/assets/59d6da2b-8aa0-469a-88b4-0a768348bbc1" />

## 9. How to Run This Project

### 9.1 Clone & Install

git clone https://github.com/cedricyu000925/steam-revenue-prediction-ml.git

cd steam-revenue-prediction-ml

pip install -r requirements.txt


### 9.2 Run the API

cd api

python app.py

The API will start on: http://localhost:5000


### 9.3 Test the API

python test_api.py

python test_latency.py

---

## 10. What This Project Demonstrates

This project is designed as a **portfolio‑ready, real‑world style** piece to show:

- ✅ **End‑to‑end thinking:** From raw data → insights → model → API → dashboard.
- ✅ **Cloud awareness:** BigQuery as a warehouse for analytics and dashboarding.
- ✅ **SQL & analytics:** Non‑trivial queries to answer concrete business questions.
- ✅ **Machine Learning:** XGBoost regression, evaluation, and genre‑level performance analysis (also trained two different ML models for comparison).
- ✅ **Explainability:** SHAP to explain predictions in business‑friendly terms.
- ✅ **Deployment mindset:** Flask API, latency measurement, and model optimization attempts.
- ✅ **Storytelling:** Clear narrative from business problem to technical solution and impact.

---

## 11. Contact

If you’d like to discuss this project or the approach:

- **Name:** Cedric Yu
- **Location:** Hong Kong
- **Email:** cedricyu80@gmail.com
- **LinkedIn:** www.linkedin.com/in/shiu-kong-yu-971556105
- **Portfolio/Website:** Coming soon!

