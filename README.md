# Analytics in the Age of AI — Telco Churn Demo

A hands-on, reproducible Databricks demo: *Reducing Telco Churn with the Databricks
Data Intelligence Platform* — one free dataset walked end-to-end through Databricks'
current (2026) AI/ML feature set. Reproducible on **Databricks Free Edition** (serverless).

## Dataset
**Telco Customer Churn** (IBM sample, ~7,043 rows × 21 columns, public domain) —
[Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Optionally extended with a free-text `CustomerFeedback` column for Vector Search / agent demos.

## Layout
```
analytics-age-of-ai/
├── notebooks/
│   ├── 01_churn_steps_1-6.py   # Unity Catalog → Feature Store → AutoML → MLflow → Registry → Serving
│   └── 02_churn_steps_7-9.py   # Vector Search, Lakebase, Databricks App deploy
├── app/                        # Databricks App (Streamlit): app.py, app.yaml, requirements.txt
└── agent/                      # Omnigent churn_explainer.yaml + steps 10-14 walkthrough
```

## Run it
1. Create a catalog + schema in your workspace (the notebooks default to `telco_churn` / `churn`).
2. Import the notebooks (Databricks source format — `# COMMAND ----------` / `# MAGIC %md`)
   via Repos or `databricks workspace import`, and run `01_...` then `02_...`.
3. Deploy the app from `app/` with the Databricks CLI (`databricks apps deploy`).

> Uncertain 2026 Beta/Preview syntax is flagged `[VERIFY AGAINST CURRENT DOCS]`.

## Naming migration (old → 2026)
| Old term | 2026 term |
|---|---|
| Mosaic AI Gateway | Unity AI Gateway |
| Mosaic AI Agent Framework | Omnigent |
| Genie | Genie One |
| Genie Space | Genie Agent-enabled Genie Space |
