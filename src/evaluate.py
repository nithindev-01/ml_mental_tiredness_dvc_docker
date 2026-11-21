import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error,
)

TARGET = "mental_tiredness_score"

# Load best model (from train stage) and test data (from prepare stage)
model = joblib.load("models/model.pkl")
test = pd.read_csv("data/processed/test.csv")

X_test = test.drop(TARGET, axis=1)
y_test = test[TARGET]

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
metrics = {
    "rmse": float(np.sqrt(mse)),
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "mse": float(mse),
    "r2": float(r2_score(y_test, y_pred)),
    "median_ae": float(median_absolute_error(y_test, y_pred)),
    "explained_variance": float(explained_variance_score(y_test, y_pred)),
    "max_error": float(max_error(y_test, y_pred)),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Test metrics for the best model:")
for k, v in metrics.items():
    print(f"  {k:20s} {v:.4f}")
print("Saved: metrics.json")