import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache" # jenkins UID shenanigans...
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch._dynamo.disable()
# config
EPOCHS      = int(os.environ.get("EPOCHS", 30))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 64))
LR          = float(os.environ.get("LR", 1e-3))
MODEL_PATH  = "./artifacts/model.pt"
FEATURES_PATH = "./artifacts/features.txt"

TEXT_COLS = ["Film_title", "Director", "Cast", "Countries",
             "Description", "Studios", "Genres"]

def load_features(df):
    drop = [c for c in TEXT_COLS if c in df.columns] + ["Average_rating"]
    return df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])

# data
print("Loading data...")
train = pd.read_csv("./artifacts/train.csv")
dev   = pd.read_csv("./artifacts/dev.csv")

X_train = load_features(train).values.astype(np.float32)
y_train = train["Average_rating"].values.astype(np.float32)

X_dev = load_features(dev).values.astype(np.float32)
y_dev = dev["Average_rating"].values.astype(np.float32)

# save
feature_names = load_features(train).columns.tolist()
with open(FEATURES_PATH, "w") as f:
    f.write("\n".join(feature_names))

print(f"Features: {len(feature_names)}  |  Train rows: {len(X_train)}  |  Dev rows: {len(X_dev)}")

# dataloader
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
dev_ds   = TensorDataset(torch.from_numpy(X_dev),   torch.from_numpy(y_dev))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)

# model
class RatingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model  = RatingMLP(X_train.shape[1]).to(device)
print(f"Model on: {device}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# train
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)

    train_loss /= len(train_ds)

    model.eval()
    dev_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dev_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            dev_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
    dev_loss /= len(dev_ds)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_MSE={train_loss:.4f}  dev_MSE={dev_loss:.4f}")

# save
os.makedirs("./artifacts", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")