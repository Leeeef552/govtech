Here’s a clean **README draft** for your system based on the architecture you described:

---

# 🏠 HDB Price Intelligence System

This project provides a natural language interface for analyzing and predicting HDB flat prices. The system routes user queries through a layered architecture that combines **LLM-based query understanding**, **machine learning (XGBoost)** for predictions, and **SQL-based analytics** for data-driven insights.

---

## 🔑 Main Entry Point

The main entry point to the system is **a user query into the LLM**.
The query is first classified into one of two categories:

1. **Prediction-based** queries → Estimate/project HDB flat prices.
2. **Analysis-based** queries → Retrieve, analyze, and present insights from the database.

---

## 🧠 Query Types

### 1. Prediction

* **Goal:** Estimate resale price of HDB flats based on user-specified variables (flat type, location, size, etc.).
* **Pipeline:**

  1. LLM interprets the query and extracts relevant variables.
  2. Data cleaning & preparation (rule-based, automated).
  3. Feature engineering for categorical, text, and numeric variables.
  4. Price prediction using **XGBoost** (with AutoML handling standard preprocessing).
  5. Results fed into synthesizer LLM → outputs projected price with natural language explanation.

---

### 2. Analysis

* **Goal:** Retrieve and analyze structured data to answer factual or exploratory questions.
* **Pipeline:**

  1. LLM converts natural language query → SQL query.
  2. SQL query executed on HDB database.
  3. Raw data returned.
  4. Data passed to synthesizer LLM → generates **analytical textual output** backed with the extracted data.

---

## 🏗️ System Architecture

```
User Query
     │
     ▼
+-----------------+
|  Classifier LLM |
+-----------------+
     │
 ┌────┴────┐
 ▼         ▼
Prediction Analysis
(ML Model) (SQL-based)
     │         │
     ▼         ▼
   Results  Retrieved Data
     │         │
     └────┬────┘
          ▼
   Synthesizer LLM
          │
          ▼
   Natural Language
   Output & Feedback
```

---

## 📌 Key Points

* **Prediction Layer:**

  * Uses XGBoost with AutoML preprocessing.
  * Handles structured, categorical, text, and numeric inputs.
  * Focused on price forecasting.

* **Analysis Layer:**

  * Translates natural queries into SQL.
  * Retrieves and summarizes data from database.
  * Produces interpretable text outputs with supporting figures.

* **Final Synthesizer:**

  * Converts raw predictions or analysis results into coherent, user-friendly explanations.
  * Ensures outputs are backed by reasoning and data.

---

## 🚀 Example Queries

* **Prediction:**

  > "What is the expected resale price for a 5-room flat in Ang Mo Kio built in 1995 with 90 years left on lease?"
  > → ML model predicts price with explanation.

* **Analysis:**

  > "Show me the trend of executive flat prices in Tampines over the last 5 years."
  > → SQL query retrieves data, LLM synthesizes into textual + tabular/chart-based analysis.

---

✅ With this design, the system intelligently distinguishes between **prediction tasks** and **analysis tasks**, ensuring the right tools (ML or SQL) are applied, while the LLM provides seamless natural language interaction.

---

