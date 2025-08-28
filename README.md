# 🏠 HDB Price Intelligence System (Updated)

This project provides a natural language interface for analyzing and recommending **HDB flat prices**. The system leverages **a unified pretrained model** that integrates both **prediction** and **analysis** tasks, ensuring consistency and simplicity.

---

## 🔑 Main Entry Point

The main entry point remains **a user query into the LLM**.
The query is classified into one of two high-level intents:

1. **Price Recommendation (BTO/Resale)** → Gauge BTO prices based on resale benchmarks.
2. **Data Analysis** → Retrieve, explore, and explain insights from the database.

---

## 🧠 Query Types

### 1. Price Recommendation (BTO & Resale)

* **Goal:** Estimate **resale prices** and use them as benchmarks to derive **BTO price recommendations**.
* **Pipeline:**

  1. LLM interprets the query and extracts relevant variables (flat type, location, lease, size, etc.).
  2. Unified pretrained model estimates **resale price**.
  3. Resale price → transformed into **BTO recommendation** using predefined adjustment rules (e.g., subsidies, launch discounts).
  4. Synthesizer LLM explains the reasoning clearly to the user.

---

### 2. Analysis

* **Goal:** Provide data-driven insights and trends on HDB prices.
* **Pipeline:**

  1. LLM interprets the query into structured SQL.
  2. SQL query retrieves relevant resale transaction data.
  3. Unified pretrained model applies feature extraction/representation for consistency.
  4. Synthesizer LLM generates clear, data-backed explanations (tables, charts, narratives).

---

## 🏗️ Unified System Architecture

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
Recommendation  Analysis
(BTO via Resale) (SQL-based)
     │              │
     └──────┬───────┘
            ▼
   Unified Pretrained Model
            │
            ▼
     Synthesizer LLM
            │
            ▼
   Natural Language Output
```

---

## 📌 Key Points

* **Unified Pretrained Model:**

  * Single model trained on historical resale transactions.
  * Outputs **resale prices** directly.
  * Provides consistent feature understanding for both prediction and analysis.

* **BTO Price Recommendation:**

  * Anchored on resale benchmarks.
  * Adjustment layer applies subsidies/discounts to resale baseline → BTO pricing gauge.

* **Analysis Layer:**

  * Uses SQL for data retrieval.
  * Unified model ensures consistent interpretation of features.
  * Synthesizer explains results clearly.

---

## 🚀 Example Queries

* **BTO Recommendation:**

  > "What’s the estimated BTO launch price for a 4-room flat in Punggol today?"
  > → System predicts resale benchmark, applies adjustments, and outputs recommended BTO price with explanation.

* **Resale Price:**

  > "What is the expected resale price for a 5-room flat in Ang Mo Kio built in 1995 with 90 years left on lease?"
  > → Unified model predicts resale price and explains.

* **Analysis:**

  > "Show me the trend of executive flat prices in Tampines over the last 5 years."
  > → SQL query fetches data, model processes features, synthesizer outputs charts + explanation.

---

