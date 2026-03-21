import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_csv("output/features.csv")

# Use only numeric features
X = df[["email_count", "active_days", "psych_risk_score"]]

# =========================
# STEP 2: NORMALIZE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# STEP 3: CURRENT INSIDER (Isolation Forest)
# =========================
iso = IsolationForest(contamination=0.1, random_state=42)
df["anomaly_label"] = iso.fit_predict(X_scaled)

# Convert to 0/1
df["is_insider"] = (df["anomaly_label"] == -1).astype(int)

# =========================
# STEP 4: FUTURE RISK (GMM)
# =========================
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)

# Lower probability = more abnormal
probs = gmm.score_samples(X_scaled)

df["future_risk_score"] = -probs

# =========================
# STEP 5: FUTURE THREAT LABEL
# =========================
threshold = df["future_risk_score"].quantile(0.8)
df["future_threat"] = (df["future_risk_score"] > threshold).astype(int)

# =========================
# STEP 6: COMBINED SCORE (🔥 KEY PART)
# =========================
df["combined_score"] = (
    0.6 * df["is_insider"] +
    0.4 * (df["future_risk_score"] / df["future_risk_score"].max())
)

# =========================
# STEP 7: FINAL RISK LEVEL
# =========================
def risk_label(x):
    if x < 0.3:
        return "LOW"
    elif x < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

df["risk_level"] = df["combined_score"].apply(risk_label)

# =========================
# STEP 8: SAVE RESULTS
# =========================
df.to_csv("output/results.csv", index=False)

print("✅ DONE: results saved to output/results.csv")

# =========================
# OPTIONAL: SHOW TOP RISKS
# =========================
print("\n🔥 Top Risky Users:")
print(df.sort_values(by="combined_score", ascending=False).head())