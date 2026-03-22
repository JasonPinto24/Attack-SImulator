import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("output/results.csv")

# =========================
# CLEAN DATA
# =========================
numeric_cols = [
    "combined_score",
    "is_insider"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

# =========================
# CREATE PREDICTIONS
# =========================
# You can adjust threshold
threshold = df["combined_score"].quantile(0.9)

df["predicted"] = (df["combined_score"] >= threshold).astype(int)

# =========================
# TRUE LABELS
# =========================
y_true = df["is_insider"]
y_pred = df["predicted"]

# =========================
# METRICS
# =========================
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# =========================
# OUTPUT
# =========================
print("\n===== MODEL PERFORMANCE =====")
print(f"F1 Score   : {f1:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"Accuracy   : {accuracy:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)