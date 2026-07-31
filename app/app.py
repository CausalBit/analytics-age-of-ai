"""Telco 'Customer Care' — Databricks App (Streamlit).

For a looked-up customer, shows a live churn-risk score from the Model Serving
endpoint and their feedback history from Lakebase, and logs the interaction.

Runs as a Databricks App (serverless). Config/resources are declared in app.yaml.
KB: knowledge-base/databricks-apps-key-concepts.md
"""
import os
import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient

# Resource names come from app.yaml -> injected as env vars.
SERVING_ENDPOINT = os.environ.get("SERVING_ENDPOINT", "telco-churn-endpoint")
LAKEBASE_HOST = os.environ.get("LAKEBASE_HOST", "")
LAKEBASE_DB = os.environ.get("LAKEBASE_DB", "databricks_postgres")

st.set_page_config(page_title="Telco Customer Care", page_icon="📞", layout="centered")
st.title("📞 Telco Customer Care — Churn Risk")

w = WorkspaceClient()


def get_churn_score(features: dict) -> float:
    """Query the Model Serving endpoint for a single customer's churn probability."""
    resp = w.serving_endpoints.query(
        name=SERVING_ENDPOINT,
        dataframe_split={"columns": list(features.keys()),
                         "data": [list(features.values())]},
    )
    # Endpoint returns predictions; adapt to your model's output shape.
    return float(resp.predictions[0])


def get_feedback_history(customer_id: str) -> pd.DataFrame:
    """Read feedback rows synced into Lakebase (Postgres) for this customer."""
    if not LAKEBASE_HOST:
        return pd.DataFrame([{"note": "Lakebase not configured in this environment."}])
    import psycopg
    # OAuth token acts as the Postgres password for Lakebase.
    token = w.config.oauth_token().access_token
    with psycopg.connect(host=LAKEBASE_HOST, dbname=LAKEBASE_DB,
                         user=w.current_user.me().user_name, password=token,
                         sslmode="require") as conn:
        return pd.read_sql(
            "SELECT customerid, customerfeedback, churn "
            "FROM customer_feedback WHERE customerid = %s", conn, params=(customer_id,))


def log_interaction(customer_id: str, score: float, note: str) -> None:
    if not LAKEBASE_HOST:
        return
    import psycopg
    token = w.config.oauth_token().access_token
    with psycopg.connect(host=LAKEBASE_HOST, dbname=LAKEBASE_DB,
                         user=w.current_user.me().user_name, password=token,
                         sslmode="require") as conn:
        conn.execute(
            "INSERT INTO care_interactions (customer_id, churn_score, note) "
            "VALUES (%s, %s, %s)", (customer_id, score, note))
        conn.commit()


# --- UI -------------------------------------------------------------------
customer_id = st.text_input("Customer ID", value="7590-VHVEG")

with st.expander("Customer features (demo inputs)"):
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total charges", 0.0, 10000.0, 840.0)
    num_services = st.slider("Num add-on services", 0, 8, 2)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Assess churn risk"):
    features = {
        "tenure": tenure, "tenure_bucket": "0-1yr" if tenure < 12 else "1-2yr",
        "num_services": num_services, "MonthlyCharges": monthly,
        "TotalCharges": total, "spend_ratio": 1.0, "Contract": contract,
        "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
    }
    try:
        score = get_churn_score(features)
        st.metric("Churn risk", f"{score:.0%}")
        st.progress(min(max(score, 0.0), 1.0))
        st.subheader("Feedback history")
        st.dataframe(get_feedback_history(customer_id))
        log_interaction(customer_id, score, note=f"Care lookup for {customer_id}")
    except Exception as e:
        st.error(f"Lookup failed: {e}")
