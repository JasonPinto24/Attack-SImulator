import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("Loading data...")
df = pd.read_csv("data/processed/features.csv")

users = df['user']
X_raw = df.drop('user', axis=1)


# -----------------------------
# 2. NORMALIZE DATA
# -----------------------------
print("Scaling data...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)


# -----------------------------
# 3. BUILD AUTOENCODER
# -----------------------------
print("Building model...")

input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)

decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')


# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
print("Training model...")
autoencoder.fit(X, X, epochs=30, batch_size=8, verbose=1)


# -----------------------------
# 5. ANOMALY DETECTION
# -----------------------------
print("Calculating anomaly scores...")

reconstructed = autoencoder.predict(X)
error = np.mean((X - reconstructed) ** 2, axis=1)

df['anomaly_score'] = error


# -----------------------------
# 6. SMART RISK SCORING
# -----------------------------
print("Calculating risk scores...")

def calculate_risk(row):
    risk = 0

    # strong signal
    if row['anomaly_score'] > 0.05:
        risk += 0.6

    # night behavior
    if row['is_night'] > 2:
        risk += 0.2

    # file spike
    if row['file_count'] > 50:
        risk += 0.5

    # email spike
    if row['email_count'] > 50:
        risk += 0.3

    return risk


df['risk_score'] = df.apply(calculate_risk, axis=1)


# -----------------------------
# 7. CLASSIFY RISK
# -----------------------------
def classify_risk(score):
    if score > 0.8:
        return "HIGH"
    elif score > 0.4:
        return "MEDIUM"
    else:
        return "LOW"


df['risk_level'] = df['risk_score'].apply(classify_risk)


# -----------------------------
# 8. INTENT DETECTION
# -----------------------------
def detect_intent(row):
    if row['file_count'] > 100:
        return "Possible Data Theft"
    elif row['is_night'] > 5:
        return "Unusual Working Hours"
    elif row['email_count'] > 100:
        return "Data Leakage Risk"
    else:
        return "Normal"


df['intent'] = df.apply(detect_intent, axis=1)


# -----------------------------
# 9. SAVE OUTPUT
# -----------------------------
print("Saving results...")

output = df[['user', 'anomaly_score', 'risk_score', 'risk_level', 'intent']]

output.to_csv("outputs/results.csv", index=False)


# -----------------------------
# DONE
# -----------------------------
print("\n✅ MODEL PIPELINE COMPLETED")
print(output.head())