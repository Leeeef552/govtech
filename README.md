Here’s a **cleanly formatted README** version, with consistent headings, code blocks, bulleting, and section separation for readability:

---

# HDB Resale & BTO Price Analysis and Prediction

## 📌 Overview

This project leverages **HDB resale and BTO price data from data.gov.sg** to build an end-to-end system for:

1. **Data Ingestion** – Collecting and preprocessing public housing data.
2. **Analysis** – Natural language query handling via LLMs to extract insights from the SQL database.
3. **Prediction** – Training and serving an ML model (XGBoost) to predict HDB resale prices.
4. **Deployment** – Hosting analysis and prediction endpoints via FastAPI.

The system integrates **LLMs as a natural language interface**, enabling users to either query the database for insights or request price predictions.

---

## 📊 Data Sources

* **BTO Prices**: [BTO Pricing Dataset](https://data.gov.sg/datasets/d_2d493bdcc1d9a44828b6e71cb095b88d/view)
* **Resale Prices**: [Resale Flat Prices Dataset](https://data.gov.sg/collections/189/view)

**Notes**:

* Data was ingested into an **SQLite database** with two tables: `bto_prices` and `resale_prices`.
* Columns were normalized (e.g., lowercased) for consistency.
* Attempted feature enrichment with transport accessibility (nearest MRT/LRT/bus stops, density). This is **work-in-progress** in `notebooks/data_ingestion.ipynb`.

---

## 🗄️ Database Schema

**BTO Prices Table**:

```sql
CREATE TABLE bto_prices (
    _id INTEGER PRIMARY KEY AUTOINCREMENT,
    financial_year TEXT,
    room_type TEXT,
    town TEXT,
    min_selling_price REAL,
    max_selling_price REAL,
    min_selling_price_less_ahg_shg REAL,
    max_selling_price_less_ahg_shg REAL
);
```

**Resale Prices Table**:

```sql
CREATE TABLE resale_prices (
    _id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT,
    town TEXT,
    flat_type TEXT,
    flat_model TEXT,
    block TEXT,
    street_name TEXT,
    storey_range TEXT,
    floor_area_sqm REAL,
    lease_commence_date TEXT,
    resale_price REAL
);
```

---

## 🤖 Model Training

Notebook: `notebooks/model_training.ipynb`

**Preprocessing**:

* One-hot encoding categorical variables.
* Feature engineering:

  * `flat_age` = transaction year − lease commencement year
  * `remaining_lease` = 99 years − `flat_age`
* (Planned) Transport-related features for accessibility.

**Model**:

* XGBoost with hyperparameter tuning (Optuna).
* Training set: \~250k rows sampled (from \~950k total).
* Considered subgroup models (e.g., by town or decade).

**Performance**:

* Final RMSE ≈ **24,000 SGD**
* Error margin: \~3–8% of actual resale price.

---

## 🏗️ System Architecture

### FastAPI Service

* `/predict` → ML model endpoint for price prediction.
* `/analysis` → SQL-based analysis endpoint.

### LLM Integration

The system leverages **Gemini LLM** as the NLP entry point for database queries and predictions.

**Core Capabilities**:

* **SQL Query Generation**: Up to 3 attempts for valid SQL → execution → validation.
* **Prediction Handling**: Extracts features, normalizes inputs, applies defaults where needed.
* **Tool Orchestration**:

  * Mode selection (analysis vs. prediction)
  * Iterative query refinement
  * Synthesizes multiple tool outputs into final user response

**System Modes**:

1. **Analysis Mode (`/analysis`)**

   * Generates SQL queries from user intent
   * Executes & validates queries (up to 3 retries)
   * Returns structured answers

2. **Prediction Mode (`/predict`)**

   * Extracts intent and features for ML model
   * Handles variable inconsistencies with LLM support
   * Provides fallback defaults for missing values

---

## 🚀 Running the System

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start FastAPI server:

   ```bash
   python main.py
   ```

   * Runs `uvicorn` with endpoints `/predict` and `/analysis`.

3. Test:

   * Edit sample queries in `main.py`
   * Or send HTTP requests to endpoints.

---

## ⚠️ Limitations & Future Improvements

1. **Latency**

   * Multiple LLM calls per query (up to 10 worst case).

2. **Feature Gaps**

   * No income-based affordability analysis.
   * Transport accessibility features not yet included.

3. **Data Considerations**

   * Historical data may reduce accuracy.
   * Possible benefit from **town-specific models**.

4. **Scalability**

   * Hosting multiple model endpoints increases complexity.

5. **Future Work**

   * Enrich data with amenities, transport, schools.
   * Optimize LLM orchestration to reduce latency.
   * Add affordability analysis tied to income groups.

---

## 📂 Repository Structure

```
.
├── api/                     # Core logic for analysis, prediction, orchestration
│   ├── analyst.py
│   ├── orchestrator_tool.py
│   ├── predictor.py
│   └── synthesizer.py
│
├── archive/                 # Archived experiments / old files
├── data/                    # Raw data & database
│   ├── *.csv
│   └── hdb_prices.db
│
├── model/                   # Trained XGBoost artifacts
├── notebooks/               # Jupyter notebooks
│   ├── data_ingestion.ipynb
│   └── model_training.ipynb
│
├── notes/                   # Documentation notes
├── server/                  # FastAPI routes
│   └── app.py
├── utils/                   # Helper functions
│
├── .env
├── .gitignore
├── README.md
├── main.py                  # FastAPI runner
└── requirements.txt
```

---
