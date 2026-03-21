import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# -------- 1. Sample Data --------
data = pd.DataFrame({
    "login_count": [5, 6, 7, 5, 100, 6, 7, 120],
    "file_access": [20, 22, 19, 21, 500, 20, 23, 600]
})

X = data.values.astype(np.float32)

# -------- 2. Isolation Forest --------
iso = IsolationForest(contamination=0.2)
iso.fit(X)
data["iso_score"] = -iso.decision_function(X)  # higher = more abnormal

# -------- 3. VAE Model --------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc21 = nn.Linear(8, latent_dim)
        self.fc22 = nn.Linear(8, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 8)
        self.fc4 = nn.Linear(8, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -------- 4. Train VAE --------
model = VAE(input_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

X_tensor = torch.tensor(X)

for epoch in range(100):
    optimizer.zero_grad()
    recon, mu, logvar = model(X_tensor)
    loss = loss_function(recon, X_tensor, mu, logvar)
    loss.backward()
    optimizer.step()

# -------- 5. VAE Anomaly Score --------
with torch.no_grad():
    recon, _, _ = model(X_tensor)
    vae_errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

data["vae_score"] = vae_errors

# -------- 6. Combine Scores --------
# normalize scores
data["iso_norm"] = (data["iso_score"] - data["iso_score"].min()) / (data["iso_score"].max() - data["iso_score"].min())
data["vae_norm"] = (data["vae_score"] - data["vae_score"].min()) / (data["vae_score"].max() - data["vae_score"].min())

# combined score
data["combined_score"] = 0.5 * data["iso_norm"] + 0.5 * data["vae_norm"]

# -------- 7. Current Insider --------
threshold = np.percentile(data["combined_score"], 80)
data["current_insider"] = data["combined_score"] > threshold

# -------- 8. Future Insider (trend) --------
data["future_risk"] = data["combined_score"].rolling(2).mean().fillna(0)
data["future_insider"] = data["future_risk"] > threshold

print(data)