# Databricks notebook source
# MAGIC %md
# MAGIC # Reducing Telco Churn — Part 2 (Steps 7–9)
# MAGIC ### Mosaic AI Vector Search → Lakebase → Databricks Apps
# MAGIC
# MAGIC Builds on Part 1. Uses the community variant of the dataset that adds a free-text
# MAGIC `CustomerFeedback` column so we can demo retrieval + an app.
# MAGIC
# MAGIC > Several fields marked `[VERIFY AGAINST CURRENT DOCS]` — Lakebase / Vector Search
# MAGIC > APIs evolve; confirm against the KB source links before the live run.

# COMMAND ----------

CATALOG, SCHEMA = "telco_churn", "churn"
spark.sql(f"USE CATALOG {CATALOG}"); spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Mosaic AI Vector Search: embed `CustomerFeedback`
# MAGIC Create a Delta-Sync vector index over the feedback text so an agent/app can
# MAGIC retrieve "why customers are unhappy" semantically.
# MAGIC
# MAGIC KB: `knowledge-base/vector-search-unstructured-retrieval.md`,
# MAGIC `vector-search-terraform-index.md`

# COMMAND ----------

# A source table with feedback text. If your CSV lacks it, synthesize a few rows so the
# demo runs; swap for the real beatafaron/telco-customer-churn-realistic-feedback variant.
from pyspark.sql import functions as F

feedback = (
    spark.table("customers")
    .select("customerID", "Contract", "Churn")
    .withColumn("CustomerFeedback",
                F.expr("""
                    CASE
                      WHEN Churn = 'Yes' THEN 'Frustrated with rising monthly charges and poor support.'
                      ELSE 'Generally satisfied, happy with the service reliability.'
                    END
                """))
)
feedback.write.format("delta").mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable("customer_feedback")           # CDF required for Delta-Sync index
display(spark.table("customer_feedback").limit(5))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
VS_ENDPOINT = "telco-vs-endpoint"
VS_INDEX = f"{CATALOG}.{SCHEMA}.feedback_index"

# 1) An endpoint hosts the index.
try:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
except Exception as e:
    print("Endpoint may already exist:", e)

# 2) A Delta-Sync index that auto-embeds the feedback text with a Databricks-hosted model.
index = vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=VS_INDEX,
    source_table_name=f"{CATALOG}.{SCHEMA}.customer_feedback",
    pipeline_type="TRIGGERED",
    primary_key="customerID",
    embedding_source_column="CustomerFeedback",
    embedding_model_endpoint_name="databricks-gte-large-en",  # a hosted embedding model
)
print("Vector index creating:", VS_INDEX)
# [VERIFY AGAINST CURRENT DOCS] embedding model endpoint name + client signature.

# COMMAND ----------

# Semantic query once the index has synced.
results = index.similarity_search(
    query_text="customers upset about price and support",
    columns=["customerID", "CustomerFeedback", "Churn"],
    num_results=3,
)
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Lakebase: managed Postgres backend / state store
# MAGIC Lakebase is serverless Postgres integrated with the lakehouse (autoscaling,
# MAGIC branching, UC sync). The app (Step 9) and agent (Part 3) use it as their
# MAGIC transactional store — e.g. logging every churn lookup a care agent performs.
# MAGIC
# MAGIC KB: `knowledge-base/lakebase-projects.md`, `lakebase-product.md`

# COMMAND ----------

# Provisioning is largely UI/CLI-driven. Reproducible CLI (run in a terminal / `%sh`):
#
#   databricks database create-database-instance telco-lakebase --capacity CU_1
#
# Then connect with any Postgres driver using an OAuth token as the password.
# [VERIFY AGAINST CURRENT DOCS] instance CLI flags + auth flow.

# COMMAND ----------

# Example connection + schema for the app's state (interaction log).
# Fill LAKEBASE_HOST from the instance details; token via WorkspaceClient.
CARE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS care_interactions (
    id           BIGSERIAL PRIMARY KEY,
    customer_id  TEXT NOT NULL,
    churn_score  DOUBLE PRECISION,
    note         TEXT,
    created_at   TIMESTAMPTZ DEFAULT now()
);
"""
print(CARE_LOG_DDL)
# In practice: psycopg.connect(host=LAKEBASE_HOST, dbname=..., user=..., password=<oauth token>, sslmode='require')
# then cur.execute(CARE_LOG_DDL). We also sync the UC `customer_feedback` table into
# Lakebase (one-click UC sync) so the app can read feedback history via SQL.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Databricks Apps: "customer care" front end
# MAGIC A small Streamlit app (see `app/` in the repo) that, for a looked-up customer:
# MAGIC 1. calls the **Model Serving** endpoint for a live churn-risk score,
# MAGIC 2. reads **feedback history** from **Lakebase**, and
# MAGIC 3. logs the interaction back to Lakebase.
# MAGIC
# MAGIC KB: `knowledge-base/databricks-apps-key-concepts.md`, `databricks-apps-product.md`
# MAGIC
# MAGIC Deploy from a terminal:
# MAGIC ```
# MAGIC databricks apps create telco-care
# MAGIC databricks sync ./app /Workspace/Users/<you>/telco-care
# MAGIC databricks apps deploy telco-care --source-code-path /Workspace/Users/<you>/telco-care
# MAGIC ```
# MAGIC The app declares its serving endpoint + Lakebase instance as resources in `app.yaml`.
