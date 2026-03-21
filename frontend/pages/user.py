import streamlit as st
import pandas as pd

st.set_page_config(page_title="User Profile", layout="wide")

# Load data
df = load_data("output/results.csv")

st.title("👤 User Profile Explorer")

# Sidebar selector
st.sidebar.title("🔍 Select User")

user_id = st.sidebar.selectbox("Select User", df["user"].unique())

user_data = df[df["user"] == user_id]

# Main content
st.subheader(f"User: {user_id}")

# Show table
st.dataframe(user_data)

# Ensure data exists before calculations
if not user_data.empty:
    
    # Get most risky transaction
    top = user_data.sort_values("risk_score", ascending=False).iloc[0]

    # 📊 Metrics
    st.subheader("📊 Risk Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", round(top["risk_score"], 3))
    col2.metric("Anomaly Score", round(top["anomaly_score"], 3))
    col3.metric("Risk Level", top["risk_level"])

    # 🧾 Clean Details
    st.markdown("### 🧾 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**👤 User:** {top['user']}")
        st.markdown(f"**🎯 Intent:** {top['intent']}")

    with col2:
        st.markdown(f"**⚠️ Risk Level:** {top['risk_level']}")
        st.markdown(f"**📊 Risk Score:** {round(top['risk_score'], 3)}")
        st.markdown(f"**📈 Anomaly Score:** {round(top['anomaly_score'], 3)}")

    # 🧠 Explanation
    st.subheader("🧠 Explanation")

    st.info(f"""
    🚨 This user is classified as **{top['risk_level']} risk**

    Reasons:
    - High anomaly score: {round(top['anomaly_score'], 3)}
    - Risk score: {round(top['risk_score'], 3)}
    - Detected intent: {top['intent']}

    Interpretation:
    This behavior deviates significantly from normal transaction patterns.
    """)

else:
    st.warning("No data available for selected user.")
