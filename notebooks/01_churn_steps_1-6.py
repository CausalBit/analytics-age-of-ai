# Databricks notebook source
# MAGIC %md
# MAGIC # Reducing Telco Churn — Part 1 (Steps 1–6)
# MAGIC ### Unity Catalog → Feature Store → AutoML → MLflow → Model Registry → Model Serving
# MAGIC
# MAGIC **Course:** Analytics in the Age of AI · **Guest lecture:** Databricks (2026)
# MAGIC
# MAGIC Reproducible on **Databricks Free Edition** (serverless). One dataset — the IBM
# MAGIC **Telco Customer Churn** sample (~7,043 rows × 21 cols, public domain) — carried
# MAGIC end-to-end. Cells are short and commented for a mixed-background audience.
# MAGIC
# MAGIC > Metrics are left as `[ACCURACY]%`-style placeholders to fill in live.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — names in one place
# MAGIC Everything uses Unity Catalog's three-level namespace: `catalog.schema.object`.

# COMMAND ----------

# Central config — edit CATALOG/SCHEMA to match your Free Edition workspace.
CATALOG = "telco_churn"          # a catalog you can CREATE in (USE CATALOG + CREATE)
SCHEMA  = "churn"                # created below if missing
TABLE   = "customers"            # raw ingested Delta table
FEATURE_TABLE = "customer_features"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.telco_churn_model"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Working in {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Unity Catalog: ingest the CSV into a governed Delta table
# MAGIC Unity Catalog is the unified governance layer: access control, lineage, auditing.
# MAGIC We ingest the public Telco CSV, register it as a Delta table, and it's immediately
# MAGIC discoverable/governed in Catalog Explorer.
# MAGIC
# MAGIC KB: `knowledge-base/unity-catalog-overview.md`, `catalogs.md`

# COMMAND ----------

# The Telco Customer Churn CSV (IBM sample). Options to load it:
#   (a) Upload the Kaggle CSV to a UC Volume and point CSV_PATH at it, OR
#   (b) Use the public raw URL below (works on Free Edition serverless).
CSV_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

import pandas as pd
pdf = pd.read_csv(CSV_URL)          # 7,043 × 21
print(pdf.shape)
pdf.head()

# COMMAND ----------

# TotalCharges arrives as text with blanks for brand-new customers — coerce to numeric.
pdf["TotalCharges"] = pd.to_numeric(pdf["TotalCharges"], errors="coerce").fillna(0.0)

# Write to a governed Delta table in Unity Catalog.
sdf = spark.createDataFrame(pdf)
(sdf.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.{TABLE}"))

print(f"Registered {CATALOG}.{SCHEMA}.{TABLE}")
display(spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC **Governance talking points (show in Catalog Explorer, no code needed):**
# MAGIC - Three-level namespace `main.telco_churn.customers`
# MAGIC - Lineage tab — this table now traces downstream to features + model
# MAGIC - `GRANT SELECT ON TABLE ... TO \`data-analysts\`` for row/column-governed access
# MAGIC - Audit log captures every read/write

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Feature Store: engineer reusable features
# MAGIC Create a Feature Engineering table in Unity Catalog. Same features used at
# MAGIC training and inference → no training/serving skew. The online store is backed by
# MAGIC **Lakebase** (Step 8).
# MAGIC
# MAGIC KB: `knowledge-base/feature-store.md`

# COMMAND ----------

from pyspark.sql import functions as F

base = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE}")

# A few teaching-friendly engineered features:
services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

features = (
    base
    .withColumn("tenure_bucket",
                F.when(F.col("tenure") < 12, "0-1yr")
                 .when(F.col("tenure") < 24, "1-2yr")
                 .when(F.col("tenure") < 48, "2-4yr")
                 .otherwise("4yr+"))
    # count of "Yes" add-on services
    .withColumn("num_services",
                sum((F.col(s) == "Yes").cast("int") for s in services))
    # spend ratio: total vs. monthly*tenure (loyalty/anomaly signal)
    .withColumn("spend_ratio",
                F.when(F.col("tenure") > 0,
                       F.col("TotalCharges") / (F.col("MonthlyCharges") * F.col("tenure")))
                 .otherwise(F.lit(1.0)))
    .select("customerID", "tenure", "tenure_bucket", "num_services",
            "MonthlyCharges", "TotalCharges", "spend_ratio", "Contract",
            "InternetService", "PaymentMethod")
)
display(features.limit(5))

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create (or overwrite) the feature table keyed on customerID.
fe.create_table(
    name=f"{CATALOG}.{SCHEMA}.{FEATURE_TABLE}",
    primary_keys=["customerID"],
    df=features,
    description="Reusable Telco churn features: tenure buckets, service counts, spend ratio.",
)
print(f"Feature table {CATALOG}.{SCHEMA}.{FEATURE_TABLE} created.")
# [VERIFY AGAINST CURRENT DOCS] exact FeatureEngineeringClient signature on your DBR.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — AutoML: classification predicting `Churn`
# MAGIC AutoML searches algorithms + hyperparameters, generates a leaderboard and a
# MAGIC readable best-model notebook. `primary_metric` for classification defaults to `f1`.
# MAGIC
# MAGIC KB: `knowledge-base/automl-classification.md`, `automl-api-reference.md`

# COMMAND ----------

# Training frame = label + features (join on customerID). Convert label to 0/1.
train_df = (
    base.select("customerID", (F.col("Churn") == "Yes").cast("int").alias("Churn"))
        .join(features, on="customerID", how="inner")
        .drop("customerID")          # id is not a predictor
)
display(train_df.limit(5))

# COMMAND ----------

from databricks import automl

summary = automl.classify(
    dataset=train_df,
    target_col="Churn",
    primary_metric="f1",       # options: log_loss, precision, accuracy, roc_auc
    timeout_minutes=15,        # keep short for a live demo (min 5)
)

print("Best trial:", summary.best_trial.model_description)
print("Best F1 (val):", summary.best_trial.metrics.get("val_f1_score"))
print("Best-model notebook:", summary.best_trial.notebook_url)
# Live: open the generated data-exploration + best-model notebooks. Best F1 ≈ [ACCURACY].

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — MLflow: tracking & comparing trials
# MAGIC Every AutoML trial is an MLflow run in one experiment. Compare params/metrics,
# MAGIC and note autologging captured signatures + artifacts automatically.
# MAGIC
# MAGIC KB: `knowledge-base/mlflow-tracking.md`, `mlflow-experiments.md`

# COMMAND ----------

import mlflow

experiment_id = summary.experiment.experiment_id
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.val_f1_score DESC"],
    max_results=10,
)
display(runs[["run_id", "metrics.val_f1_score", "metrics.val_roc_auc_score",
              "tags.mlflow.runName"]])
# Talking point: the Experiments UI compares these runs visually (parallel-coords plot).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Unity Catalog Model Registry: register & version the winner
# MAGIC Point MLflow at UC (`databricks-uc`) and register with the three-level name.
# MAGIC Use an **alias** (e.g. `Champion`) as a mutable pointer for deployment.
# MAGIC
# MAGIC KB: `knowledge-base/manage-model-lifecycle-uc.md`

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

best_run_id = summary.best_trial.mlflow_run_id
model_uri = f"runs:/{best_run_id}/model"

mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"Registered {MODEL_NAME} version {mv.version}")

# Mark this version as the deployment target.
from mlflow.tracking import MlflowClient
MlflowClient().set_registered_model_alias(MODEL_NAME, "Champion", mv.version)
print(f"Alias 'Champion' -> version {mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Model Serving: real-time REST endpoint
# MAGIC Deploy the Champion as a serverless REST endpoint, then query live churn-risk
# MAGIC scores. In Part 2 we add online feature lookups (Feature Store + Lakebase).
# MAGIC
# MAGIC KB: `knowledge-base/model-serving.md`, `model-serving-endpoints.md`

# COMMAND ----------

# Create/refresh a serving endpoint from the registered model's Champion alias.
# The SDK is the reproducible path; the UI (Serving > Create) is the demo-friendly path.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput,
)

w = WorkspaceClient()
ENDPOINT = "telco-churn-endpoint"

served = ServedEntityInput(
    entity_name=MODEL_NAME,
    entity_version=mv.version,
    scale_to_zero_enabled=True,       # cost-friendly for a classroom
    workload_size="Small",
)
try:
    w.serving_endpoints.create(
        name=ENDPOINT,
        config=EndpointCoreConfigInput(served_entities=[served]),
    )
    print(f"Creating endpoint {ENDPOINT} (a few minutes to become Ready)...")
except Exception as e:
    print("Endpoint may already exist — update its config in the UI/API instead.", e)
# [VERIFY AGAINST CURRENT DOCS] exact serving SDK fields on your workspace version.

# COMMAND ----------

# Once the endpoint is Ready, query it with a sample customer's features.
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
sample = train_df.drop("Churn").limit(3).toPandas()

response = client.predict(
    endpoint=ENDPOINT,
    inputs={"dataframe_split": sample.to_dict(orient="split")},
)
print("Live churn-risk predictions:", response)
# In the app (Part 2) a "customer care" user sees this score per customer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap (Steps 1–6)
# MAGIC Governed table → reusable features → AutoML leaderboard → tracked/compared runs →
# MAGIC versioned Champion model → live REST endpoint. **Part 2** adds Vector Search over
# MAGIC customer feedback, a Lakebase Postgres backend, and a Databricks App.
