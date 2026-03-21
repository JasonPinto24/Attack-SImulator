import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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
# VAE PARAMETERS
# -----------------------------
input_dim = X.shape[1]
latent_dim = 2

# -----------------------------
# ENCODER
# -----------------------------
inputs = tf.keras.Input(shape=(input_dim,))
h = layers.Dense(8, activation='relu')(inputs)
h = layers.Dense(4, activation='relu')(h)

z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)


# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


z = layers.Lambda(sampling)([z_mean, z_log_var])


# -----------------------------
# DECODER
# -----------------------------
decoder_h = layers.Dense(4, activation='relu')
decoder_h2 = layers.Dense(8, activation='relu')
decoder_out = layers.Dense(input_dim, activation='sigmoid')

h_decoded = decoder_h(z)
h_decoded = decoder_h2(h_decoded)
outputs = decoder_out(h_decoded)


# -----------------------------
# VAE MODEL
# -----------------------------
vae = tf.keras.Model(inputs, outputs)


# -----------------------------
# LOSS FUNCTION (FIXED)
# -----------------------------
reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
reconstruction_loss = tf.reduce_mean(reconstruction_loss) * input_dim

kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
)

vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')


# -----------------------------
# TRAIN
# -----------------------------
vae.fit(X, X, epochs=30, batch_size=8, verbose=1)
# -----------------------------
# 5. ANOMALY DETECTION
# -----------------------------
print("Calculating anomaly scores...")

reconstructed = vae.predict(X)
error = np.mean((X - reconstructed) ** 2, axis=1)

df['anomaly_score'] = error


# -----------------------------
# 6. SMART RISK SCORING (UPDATED)
# -----------------------------
print("Calculating risk scores...")

def calculate_risk(row):
    risk = 0

    # anomaly signal
    if row['anomaly_score'] > 0.05:
        risk += 0.6

    # high email activity (binary flag)
    if row['high_email_activity'] == 1:
        risk += 0.4

    # very high email count
    if row['email_count'] > 100:
        risk += 0.5

    # psychological risk (important new feature)
    if row['psych_risk_score'] > 0.7:
        risk += 0.7

    # inactive then spike behavior
    if row['active_days'] < 5 and row['email_count'] > 50:
        risk += 0.5

    return risk


df['risk_score'] = df.apply(calculate_risk, axis=1)


# -----------------------------
# 7. CLASSIFY RISK
# -----------------------------
def classify_risk(score):
    if score > 1.2:
        return "HIGH"
    elif score > 0.6:
        return "MEDIUM"
    else:
        return "LOW"


df['risk_level'] = df['risk_score'].apply(classify_risk)


# -----------------------------
# 8. INTENT DETECTION (UPDATED)
# -----------------------------
def detect_intent(row):
    if row['psych_risk_score'] > 0.8:
        return "Potential Insider Threat (Behavioral Risk)"
    elif row['email_count'] > 150:
        return "Possible Data Exfiltration"
    elif row['high_email_activity'] == 1:
        return "Unusual Communication Spike"
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
print("\n✅ MODEL UPDATED FOR NEW FEATURES")
print(output.head())