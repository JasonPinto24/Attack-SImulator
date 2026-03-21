import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

import shap

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("output/features.csv")

# Features used for model
feature_cols = ["email_count", "active_days", "psych_risk_score"]
X = df[feature_cols]

# =========================
# STEP 2: NORMALIZE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# STEP 3: ISOLATION FOREST (CURRENT INSIDER)
# =========================
iso = IsolationForest(contamination=0.1, random_state=42)
df["anomaly_label"] = iso.fit_predict(X_scaled)

df["is_insider"] = (df["anomaly_label"] == -1).astype(int)

# =========================
# STEP 4: GMM (FUTURE RISK)
# =========================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

log_probs = gmm.score_samples(X_scaled)

# Convert to positive risk score
risk_values = -log_probs.reshape(-1, 1)

risk_scaler = MinMaxScaler()
df["future_risk_score"] = risk_scaler.fit_transform(risk_values)

# Future threat label
threshold = df["future_risk_score"].quantile(0.8)
df["future_threat"] = (df["future_risk_score"] > threshold).astype(int)

# =========================
# STEP 5: COMBINED SCORE
# =========================
df["combined_score"] = (
    0.6 * df["is_insider"] +
    0.4 * df["future_risk_score"]
)

# Risk levels
def risk_label(x):
    if x < 0.3:
        return "LOW"
    elif x < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

df["risk_level"] = df["combined_score"].apply(risk_label)

# =========================
# STEP 6: SHAP EXPLANATION
# =========================
try:
    # Use TreeExplainer (better for tree models)
    explainer = shap.TreeExplainer(iso)

    shap_values = explainer.shap_values(X_scaled)

    # Convert SHAP values into DataFrame
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{col}" for col in feature_cols]
    )

    # Add SHAP values to main df
    df = pd.concat([df.reset_index(drop=True), shap_df], axis=1)

    print("✅ SHAP values computed successfully")

except Exception as e:
    print("⚠️ SHAP failed, skipping explanations:", e)

# =========================
# STEP 7: SAVE RESULTS
# =========================
df.to_csv("output/results.csv", index=False)

print("✅ DONE: results saved to output/results.csv")

# =========================
# STEP 8: SHOW TOP RISKS
# =========================
print("\n🔥 Top Risky Users:")
print(df.sort_values(by="combined_score", ascending=False).head(10))