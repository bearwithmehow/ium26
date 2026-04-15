import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

MODEL_PATH    = "./artifacts/model.pt"
FEATURES_PATH = "./artifacts/features.txt"
OUTPUT_PATH   = "./artifacts/predictions.csv"

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

with open(FEATURES_PATH) as f:
    feature_names = [line.strip() for line in f.readlines()]

print("Loading test data...")
test = pd.read_csv("./artifacts/test.csv")

X_test = test[feature_names].values.astype(np.float32)
print(f"Test rows: {len(X_test)}  |  Features: {len(feature_names)}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model  = RatingMLP(len(feature_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

with torch.no_grad():
    X_tensor = torch.from_numpy(X_test).to(device)
    preds = model(X_tensor).cpu().numpy()

results = pd.DataFrame({
    "Film_title":        test["Film_title"].values,
    "Actual_rating":     test["Average_rating"].values,
    "Predicted_rating":  preds
})

results.to_csv(OUTPUT_PATH, index=False)
print(f"\nPredictions saved → {OUTPUT_PATH}")
print(results.head(10).to_string(index=False))