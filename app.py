import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("💳 Fraud Detection Dashboard")

df = pd.read_csv("outputs/results.csv")

st.subheader("🚨 High-Risk Users")

high_risk = df[df["risk_level"] == "HIGH"]

st.dataframe(high_risk.sort_values(by="risk_score", ascending=False))

st.subheader("📊 Anomaly Score Distribution")

fig, ax = plt.subplots()
ax.hist(df["anomaly_score"], bins=50)

st.pyplot(fig)

st.subheader("📊 Risk Level Distribution")

st.bar_chart(df["risk_level"].value_counts())

st.subheader("📊 Risk Score Distribution")

fig, ax = plt.subplots()
ax.hist(df["risk_score"], bins=50)

st.pyplot(fig)

st.sidebar.title("👤 User Explorer")

user_id = st.sidebar.selectbox("Select User", df["user"].unique())

user_data = df[df["user"] == user_id]

st.subheader(f"User: {user_id}")

st.dataframe(user_data)
st.metric("Avg Risk Score", round(user_data["risk_score"].mean(), 3))
st.metric("Max Risk Level", user_data["risk_level"].mode()[0])