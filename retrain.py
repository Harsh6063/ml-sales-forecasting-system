# retrain.py

import os
import joblib
import numpy as np
import pandas as pd

from ingest import load_data, merge_data
from features import build_features
from drift import split_data, detect_drift

from train import train_models


# -----------------------------
# Check if Retraining Needed
# -----------------------------
def check_drift_and_retrain(df, threshold=0.2):
    old_data, new_data = split_data(df)

    features = ["Sales", "Customers", "Promo", "SchoolHoliday"]

    drift_report = detect_drift(old_data, new_data, features, threshold)

    # If any feature crosses threshold → retrain
    drift_detected = any(v > threshold for v in drift_report.values())

    return drift_detected, new_data


# -----------------------------
# Retrain Pipeline
# -----------------------------
def retrain_pipeline():
    print("\n🚀 Starting Retraining Pipeline...\n")

    # Load data
    train, test, store = load_data()
    df = merge_data(train, store)

    # Sort by time
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])

    # Check drift
    drift_detected, new_data = check_drift_and_retrain(df)

    if not drift_detected:
        print("✅ No significant drift detected. No retraining needed.")
        return

    print("\n🚨 Drift detected → Retraining model...\n")

    # Use full updated dataset (or just new_data if you want incremental)
    X, y = build_features(df)

    # Train new model
    model = train_models(X, y)

    # Save new model
    save_model(model)

    print("\n✅ Retraining complete. New model deployed.")


# -----------------------------
# Save Model
# -----------------------------
def save_model(model, path="models/best_model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model updated at {path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    retrain_pipeline()