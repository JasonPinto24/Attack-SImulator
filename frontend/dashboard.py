import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", layout="wide")

df = pd.read_csv("outputs/results.csv")

st.title("💳 Fraud Detection Dashboard")

# 🚨 High Risk
st.subheader("🚨 High-Risk Users")
high_risk = df[df["risk_level"] == "HIGH"]
st.dataframe(high_risk)

# 📊 Risk Level Distribution
st.subheader("📊 Risk Level Distribution")
st.bar_chart(df["risk_level"].value_counts())

# 📊 Anomaly Distribution
st.subheader("📊 Anomaly Score Distribution")
fig, ax = plt.subplots()
ax.hist(df["anomaly_score"], bins=50)
st.pyplot(fig)

# 📋 Full Data
st.subheader("📋 All Data")
st.dataframe(df)

st.info("👉 Go to 'User' page from sidebar to explore individual users")