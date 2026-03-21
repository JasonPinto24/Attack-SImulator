import streamlit as st
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Insider Threat Detection",
    layout="wide"
)

st.title("🔐 Insider Threat Detection Dashboard")
st.markdown("AI-powered detection using Isolation Forest + GMM + SHAP")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("output/results.csv", nrows=500)

# =========================
# CLEAN DATA
# =========================
numeric_cols = [
    "combined_score", "email_count",
    "psych_risk_score", "active_days",
    "future_threat", "is_insider"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Level",
    options=df["risk_level"].dropna().unique(),
    default=df["risk_level"].dropna().unique()
)

filtered_df = df[df["risk_level"].isin(risk_filter)]

# =========================
# METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Users", len(filtered_df))
col2.metric("🚨 High Risk Users", len(filtered_df[filtered_df["risk_level"] == "HIGH"]))
col3.metric("⚠️ Future Threats", int(filtered_df["future_threat"].sum()))


st.markdown("---")

# =========================
# 🔥 TOP RISK USERS
# =========================
st.subheader("🔥 Top Risky Users")

top_users = filtered_df.sort_values(
    by="combined_score", ascending=False
).head(10)

st.dataframe(top_users)

# =========================
# 📊 RISK BREAKDOWN
# =========================
st.subheader("📊 Risk Level Breakdown")

risk_counts = filtered_df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

st.table(risk_counts)

# =========================
# 🚨 CRITICAL ALERTS
# =========================
st.subheader("🚨 Critical Alerts")

threshold = filtered_df["combined_score"].quantile(0.95)

critical_users = filtered_df[
    filtered_df["combined_score"] >= threshold
]

if not critical_users.empty:
    st.error(f"🚨 {len(critical_users)} CRITICAL USERS DETECTED")

    st.dataframe(
        critical_users[["user", "combined_score", "risk_level"]]
        .sort_values(by="combined_score", ascending=False)
    )
else:
    st.success("✅ No critical threats detected")

# =========================
# ⚡ INSIDER DISTRIBUTION
# =========================
st.subheader("⚡ Insider vs Normal Users")

insider_counts = filtered_df["is_insider"].value_counts().reset_index()
insider_counts.columns = ["Type", "Count"]

st.table(insider_counts)

# =========================
# 🔍 USER ANALYSIS
# =========================
st.subheader("🔍 Analyze Individual User")

selected_user = st.selectbox(
    "Select User",
    filtered_df["user"].unique()
)

user_data = filtered_df[
    filtered_df["user"] == selected_user
].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.write("### 📌 User Details")
    st.json({
        "Email Count": int(user_data["email_count"]),
        "Active Days": int(user_data["active_days"]),
        "Psych Risk": float(user_data["psych_risk_score"]),
        "Risk Level": user_data["risk_level"]
    })

with col2:
    st.write("### ⚠️ Risk Scores")
    st.json({
        "Is Insider": int(user_data["is_insider"]),
        "Future Threat": int(user_data["future_threat"]),
        "Combined Score": float(user_data["combined_score"])
    })

# =========================
# 🧠 SHAP (USER LEVEL)
# =========================
st.subheader("🧠 Why is this user risky?")

if all(col in filtered_df.columns for col in [
    "shap_email_count",
    "shap_active_days",
    "shap_psych_risk_score"
]):
    shap_values = {
        "Email Count": user_data["shap_email_count"],
        "Active Days": user_data["shap_active_days"],
        "Psych Risk": user_data["shap_psych_risk_score"]
    }

    sorted_shap = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for feature, impact in sorted_shap:
        if impact > 0:
            st.write(f"🔴 {feature} increases risk by {round(impact, 3)}")
        else:
            st.write(f"🟢 {feature} reduces risk by {round(abs(impact), 3)}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("⚡ Built by Team DHURANDAR")