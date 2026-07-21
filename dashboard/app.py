"""Churn platform dashboard.

A Streamlit web app that reads the serving database (Postgres) and shows
churn risk and complaint activity per US state. It never touches the raw
data or the models, only the gold tables the pipeline produced.

Run locally:
    streamlit run dashboard/app.py
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Customer Churn Platform", layout="wide")


@st.cache_resource
def get_engine():
    """One database connection, reused across reruns.

    Host and credentials come from the environment so the same code works
    locally (localhost:6432) and in the container (host qs-postgres:5432).
    """
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "6432")
    user = os.environ.get("DB_USER", "churn")
    password = os.environ.get("DB_PASSWORD", "churnpass")
    name = os.environ.get("DB_NAME", "churn")
    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    )


@st.cache_data(ttl=600)
def run_query(sql: str) -> pd.DataFrame:
    """Run a query and cache the result for 10 minutes."""
    return pd.read_sql(sql, get_engine())


def us_map(df: pd.DataFrame, value: str, label: str, scale: str):
    """Choropleth of the US colored by `value`.

    locationmode="USA-states" tells Plotly the `state_code` column holds
    two-letter state codes, which is why dim_state keeps them.
    """
    fig = px.choropleth(
        df,
        locations="state_code",
        locationmode="USA-states",
        color=value,
        scope="usa",
        color_continuous_scale=scale,
        hover_name="state_name",
        labels={value: label},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


st.title("Customer Churn Platform")
st.caption("Churn risk and consumer complaints across US states")

# Headline numbers. Each fact table is queried on its own, never joined
# to the other, so the different grains do not fan out.
kpis = run_query(
    """
    SELECT count(*)               AS customers,
           avg(churned::float)    AS churn_rate,
           avg(churn_probability) AS avg_probability
    FROM fact_churn_prediction
    """
).iloc[0]
complaints_total = run_query(
    "SELECT sum(complaint_count) AS n FROM fact_complaints"
).iloc[0]["n"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers scored", f"{int(kpis['customers']):,}")
c2.metric("Churn rate", f"{kpis['churn_rate']:.0%}")
c3.metric("Avg churn probability", f"{kpis['avg_probability']:.2f}")
c4.metric("Complaints", f"{int(complaints_total):,}")

# Slider controls how many rows the detail tables show. The maps always
# show every state.
top_n = st.sidebar.slider("Rows in the tables", 5, 51, 10)

churn_by_state = run_query(
    """
    SELECT s.state_code,
           s.state_name,
           count(*)                 AS customers,
           avg(f.churn_probability) AS avg_probability,
           avg(f.churned::float)    AS churn_rate
    FROM fact_churn_prediction f
    JOIN dim_state s ON s.state_sk = f.state_sk
    GROUP BY s.state_code, s.state_name
    ORDER BY customers DESC
    """
)
complaints_by_state = run_query(
    """
    SELECT s.state_code,
           s.state_name,
           sum(c.complaint_count) AS complaints
    FROM fact_complaints c
    JOIN dim_state s ON s.state_sk = c.state_sk
    GROUP BY s.state_code, s.state_name
    ORDER BY complaints DESC
    """
)

churn_tab, complaints_tab = st.tabs(["Churn risk", "Complaints"])

with churn_tab:
    st.subheader("Average churn probability by state")
    st.plotly_chart(
        us_map(churn_by_state, "avg_probability", "Churn probability", "Reds"),
        use_container_width=True,
    )
    st.dataframe(churn_by_state.head(top_n), use_container_width=True)

with complaints_tab:
    st.subheader("Complaint volume by state")
    st.plotly_chart(
        us_map(complaints_by_state, "complaints", "Complaints", "Blues"),
        use_container_width=True,
    )
    st.dataframe(complaints_by_state.head(top_n), use_container_width=True)
