import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("outputs/results.csv")

df = load_data()

st.title("💳 Fraud Detection Dashboard")

# =========================
# 🔢 KPI CARDS
# =========================
st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", len(df))
col2.metric("High Risk Users", len(df[df["risk_level"] == "HIGH"]))
col3.metric("Avg Risk Score", round(df["risk_score"].mean(), 3))
col4.metric("Avg Anomaly Score", round(df["anomaly_score"].mean(), 3))

# =========================
# 🚨 HIGH RISK TABLE
# =========================
st.subheader("🚨 High-Risk Users")

high_risk = df[df["risk_level"] == "HIGH"].sort_values(by="risk_score", ascending=False)

st.dataframe(high_risk)

# =========================
# 📊 VISUALS (ROW 1)
# =========================
col1, col2 = st.columns(2)

# Pie chart (Risk Levels)
with col1:
    fig_pie = px.pie(
        df,
        names="risk_level",
        title="Risk Level Distribution",
        color="risk_level",
        color_discrete_map={
            "HIGH": "red",
            "MEDIUM": "orange",
            "LOW": "green"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Histogram (Risk Score)
with col2:
    fig_hist = px.histogram(
        df,
        x="risk_score",
        nbins=50,
        title="Risk Score Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# 📊 VISUALS (ROW 2)
# =========================
col1, col2 = st.columns(2)

# Scatter plot (Anomaly vs Risk)
with col1:
    fig_scatter = px.scatter(
        df,
        x="anomaly_score",
        y="risk_score",
        color="risk_level",
        size="risk_score",
        hover_data=["user", "intent"],
        title="Anomaly vs Risk Analysis"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Intent distribution
with col2:
    intent_counts = df["intent"].value_counts().reset_index()
    intent_counts.columns = ["intent", "count"]

    fig_bar = px.bar(
        intent_counts,
        x="count",
        y="intent",
        orientation="h",
        title="Intent Distribution"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# 📋 FULL DATA TABLE
# =========================
st.subheader("📋 All Data")

# Highlight high risk rows
def highlight(row):
    if row["risk_level"] == "HIGH":
        return ["background-color: red"] * len(row)
    elif row["risk_level"] == "MEDIUM":
        return ["background-color: orange"] * len(row)
    else:
        return [""] * len(row)

st.dataframe(df.style.apply(highlight, axis=1))

# =========================
# 🔽 DOWNLOAD BUTTON
# =========================
st.download_button(
    "📥 Download Results CSV",
    df.to_csv(index=False),
    file_name="fraud_results.csv",
    mime="text/csv"
)

# =========================
# 🧠 INSIGHT PANEL
# =========================
st.subheader("🧠 Insights")

st.info(f"""
📌 Key Observations:

- {len(df[df["risk_level"] == "HIGH"])} users are classified as HIGH risk  
- Average risk score is {round(df["risk_score"].mean(), 3)}  
- High anomaly scores strongly correlate with high risk  

👉 Use the **User page** from the sidebar to drill down into individual behavior.
""")

# =========================
# 🔗 NAVIGATION HINT
# =========================
st.success("👉 Go to 'User' page from sidebar to explore individual users")