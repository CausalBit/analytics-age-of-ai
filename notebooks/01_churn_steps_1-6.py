# Databricks notebook source
# MAGIC %md
# MAGIC # Reducing Telco Churn — Part 1 (Steps 1–6)
# MAGIC ### Unity Catalog → Features → Model Training → MLflow → Model Registry → Model Serving
# MAGIC
# MAGIC **Course:** Analytics in the Age of AI · **Guest lecture:** Databricks (2026)
# MAGIC
# MAGIC Reproducible on **Databricks Free Edition** (serverless). One dataset — the IBM
# MAGIC **Telco Customer Churn** sample (~7,043 rows × 21 cols, public domain) — carried
# MAGIC end-to-end. Cells are short and commented for a mixed-background audience.
# MAGIC
# MAGIC > **Serverless-native.** Free Edition has no ML-runtime clusters, so this notebook
# MAGIC > uses scikit-learn (not classic AutoML) and a plain Unity Catalog Delta feature
# MAGIC > table (not the Feature Engineering client). The story — governed data → reusable
# MAGIC > features → tracked training → versioned Champion → live endpoint — is identical.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — upgrade MLflow for Free Edition
# MAGIC Free Edition serverless ships MLflow ~2.11, whose UC artifact-upload path hits an
# MAGIC S3 access-denied when registering a model. MLflow ≥ 2.20 fixes it. Upgrade first,
# MAGIC then restart Python so the new version is used.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow-skinny[databricks]>=2.20" "mlflow>=2.20"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Names in one place
# MAGIC Everything uses Unity Catalog's three-level namespace: `catalog.schema.object`.

# COMMAND ----------

# Central config — edit CATALOG/SCHEMA to match your Free Edition workspace.
CATALOG = "telco_churn"          # a catalog you can CREATE in (USE CATALOG + CREATE)
SCHEMA  = "churn"                # created below if missing
TABLE   = "customers"            # raw ingested Delta table
FEATURE_TABLE = "customer_features"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.telco_churn_model"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
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
# MAGIC - Three-level namespace `telco_churn.churn.customers`
# MAGIC - Lineage tab — this table now traces downstream to features + model
# MAGIC - `GRANT SELECT ON TABLE ... TO \`data-analysts\`` for row/column-governed access
# MAGIC - Audit log captures every read/write

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Features: engineer reusable features into a governed Delta table
# MAGIC On an ML runtime you'd use the **Feature Engineering client** for a managed feature
# MAGIC table with an online store. Free Edition is serverless, so we write the same
# MAGIC engineered features to a plain **Unity Catalog Delta table** — still governed,
# MAGIC lineage-tracked, and reusable at training and inference (no training/serving skew).

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

# Persist as a governed Delta feature table keyed on customerID.
(features.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.{FEATURE_TABLE}"))

spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.{FEATURE_TABLE} IS "
          "'Reusable Telco churn features: tenure buckets, service counts, spend ratio.'")
print(f"Feature table {CATALOG}.{SCHEMA}.{FEATURE_TABLE} created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Train a classification model predicting `Churn`
# MAGIC On an ML runtime, **AutoML** (`databricks.automl`) would search algorithms +
# MAGIC hyperparameters and emit a leaderboard + best-model notebook. That API needs an
# MAGIC ML cluster (not available on Free Edition serverless), so here we train a
# MAGIC scikit-learn pipeline directly — the same "features → fitted classifier →
# MAGIC f1 score" idea AutoML automates. Swap this cell for `automl.classify(...)` on a
# MAGIC paid ML-runtime workspace.

# COMMAND ----------

# Assemble the training frame: label + features, joined on customerID.
train_sdf = (
    base.select("customerID", (F.col("Churn") == "Yes").cast("int").alias("Churn"))
        .join(features, on="customerID", how="inner")
        .drop("customerID")          # id is not a predictor
)
train_pdf = train_sdf.toPandas()
print("Training frame:", train_pdf.shape)
display(train_sdf.limit(5))

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

y = train_pdf["Churn"]
X = train_pdf.drop(columns=["Churn"])

cat_cols = ["tenure_bucket", "Contract", "InternetService", "PaymentMethod"]
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])
# RandomForest: strong, interpretable, and its components are skops-trusted (UC model
# registration validates artifacts with skops; some classifiers' internal loss objects
# are rejected as untrusted).
clf = Pipeline([("pre", pre),
                ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — MLflow: track the run, params, and metrics
# MAGIC MLflow autologging captures params, metrics, the model signature, and artifacts.
# MAGIC The Experiments UI compares runs visually — the same tracking layer AutoML uses
# MAGIC under the hood for every trial.

# COMMAND ----------

import mlflow, mlflow.sklearn
from mlflow.models.signature import infer_signature

# On serverless (Spark Connect), set the tracking + registry URIs explicitly BEFORE
# any run starts. Otherwise MlflowClient() tries to resolve the registry URI from a
# Spark conf (spark.mlflow.modelRegistryUri) that is blocked for reads on serverless.
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# We also log explicitly rather than using mlflow.sklearn.autolog(), whose Spark
# integration reads the same blocked conf.
with mlflow.start_run(run_name="telco-churn-gbc") as run:
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    mlflow.log_params({
        "model": "RandomForestClassifier",
        "n_estimators": 200,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "cat_cols": ",".join(cat_cols),
    })
    metrics = {
        "test_f1_score": f1_score(y_test, preds),
        "test_roc_auc_score": roc_auc_score(y_test, proba),
        "test_accuracy": accuracy_score(y_test, preds),
    }
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(
        clf, artifact_path="model",
        signature=signature, input_example=X_test.head(3),
    )
    run_id = run.info.run_id

print("Run:", run_id)
print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
# Live: open the Experiments UI to compare this run's params/metrics. Best F1 ≈ [ACCURACY].

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Unity Catalog Model Registry: register & version the winner
# MAGIC Point MLflow at UC (`databricks-uc`) and register with the three-level name.
# MAGIC Use an **alias** (e.g. `Champion`) as a mutable pointer for deployment.

# COMMAND ----------

# Registry URI already set to databricks-uc at the top of the notebook.
model_uri = f"runs:/{run_id}/model"
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
# MAGIC scores. Model Serving is available on Free Edition.

# COMMAND ----------

# Create (or update) a serving endpoint from the registered model's version.
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

# Does the endpoint already exist? (list is more reliable than catching get's 404.)
existing = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT in existing:
    print(f"Endpoint {ENDPOINT} exists — updating to version {mv.version} ...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT, served_entities=[served])
else:
    print(f"Creating endpoint {ENDPOINT} (a few minutes to become Ready)...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT,
        config=EndpointCoreConfigInput(served_entities=[served]),
    )
print(f"Endpoint {ENDPOINT} is ready on version {mv.version}.")

# COMMAND ----------

# Query the live endpoint with sample customers.
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
sample = X_test.head(3)
response = client.predict(
    endpoint=ENDPOINT,
    inputs={"dataframe_split": sample.to_dict(orient="split")},
)
print("Live churn-risk predictions:", response)
# In the app (Part 2) a "customer care" user sees this score per customer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap (Steps 1–6)
# MAGIC Governed table → reusable features → tracked training → versioned Champion model →
# MAGIC live REST endpoint. **Part 2** adds Vector Search over customer feedback, a Lakebase
# MAGIC Postgres backend, and a Databricks App.
# MAGIC
# MAGIC > On a paid **ML-runtime** workspace, swap Step 2 for the Feature Engineering client
# MAGIC > and Step 3 for `databricks.automl.classify(...)` — the rest is unchanged.
