import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "mental_tiredness_score"

# Read settings from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)["prepare"]

data_path = params["data_path"]
test_size = params["test_size"]
random_state = params["random_state"]

# Load
data = pd.read_csv(data_path)
print("Raw shape:", data.shape)

# Clean
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
data = data.drop_duplicates()

# Feature engineering
data["deep_sleep_hours"] = data["sleep_hours"] * (data["deep_sleep_pct"] / 100.0)
data["workload_intensity"] = data["workload_score"] * data["context_switch_count"]

# Train / test split
train, test = train_test_split(data, test_size=test_size, random_state=random_state)

# Save
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Saved: data/processed/train.csv and data/processed/test.csv")