import pandas as pd

# STEP 1: LOAD DATA
email = pd.read_csv(
    "data/processed/email.csv",
    usecols=["user", "date"],   # need date for active_days
    nrows=200000
)

psy = pd.read_csv("data/processed/psychometric.csv")

# STEP 2: FIX COLUMN NAME (if needed)
if "user_id" in psy.columns:
    psy = psy.rename(columns={"user_id": "user"})

# STEP 3: CLEAN DATE
email["date"] = pd.to_datetime(email["date"], errors="coerce")

# STEP 4: EMAIL FEATURES
email_features = email.groupby("user").agg(
    email_count=("user", "count"),
    active_days=("date", "nunique")
).reset_index()

# STEP 5: HIGH EMAIL ACTIVITY FLAG
email_features["high_email_activity"] = (email_features["email_count"] > 50).astype(int)

# STEP 6: PSYCHOMETRIC FEATURE
# ⚠️ Replace 'risk_score' with actual column name if different
if "risk_score" in psy.columns:
    psy_features = psy[["user", "risk_score"]].rename(columns={"risk_score": "psych_risk_score"})
else:
    # fallback: take first numeric column
    numeric_cols = psy.select_dtypes(include="number").columns
    psy_features = psy[["user", numeric_cols[0]]].rename(columns={numeric_cols[0]: "psych_risk_score"})

# STEP 7: MERGE
features = pd.merge(email_features, psy_features, on="user", how="left")

# STEP 8: HANDLE MISSING VALUES
features["psych_risk_score"] = features["psych_risk_score"].fillna(0)

# STEP 9: FINAL FORMAT (EXACT MATCH)
features = features[
    ["user", "email_count", "active_days", "high_email_activity", "psych_risk_score"]
]

# STEP 10: SAVE
features.to_csv("output/features.csv", index=False)

print("✅ DONE: features.csv created (new format)")