# retrain.py

import os
import joblib
import pandas as pd

from ingest import load_data, merge_data
from features import build_features
from train import train_models

from evidently import Report
from evidently.presets import DataDriftPreset


# -----------------------------
# Split Data
# -----------------------------
def split_data(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    return df.iloc[:split], df.iloc[split:]


# -----------------------------
# Get Drift Score from Evidently
# -----------------------------
def get_drift_score(old_data, new_data):

    old_data = old_data.select_dtypes(include=["number"])
    new_data = new_data.select_dtypes(include=["number"])

    report = Report(metrics=[DataDriftPreset()])

    my_eval=report.run(
        reference_data=old_data,
        current_data=new_data
    )

    # Save report
    my_eval.save_html("drift_report.html")

    # Extract drift info
    result = my_eval.dict()
    metrics = result.get("metrics", [])

    drift_share = 0.0
    drift_count = 0

    for m in metrics:
        if "DriftedColumnsCount" in m.get("metric_name", ""):
            drift_share = m["value"]["share"]
            drift_count = m["value"]["count"]

    dataset_drift = drift_share > 0.3

    print(f"\n📊 Drifted Columns: {int(drift_count)}")
    print(f"📊 Drift Share: {drift_share:.2f}")
    print(f"📊 Dataset Drift: {dataset_drift}")

    return dataset_drift, drift_share


# -----------------------------
# Retraining Pipeline
# -----------------------------
def retrain_pipeline(threshold=0.3):
    print("\n🚀 Starting Retraining Pipeline...\n")

    # Load data
    train, test, store = load_data()
    df = merge_data(train, store)

    # Sort by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])

    # Split
    old_data, new_data = split_data(df)

    # Drift check
    dataset_drift, drift_share = get_drift_score(old_data, new_data)

    # Decision logic
    if drift_share < threshold:
        print("\n✅ Drift below threshold → No retraining needed.")
        return

    print("\n🚨 Drift above threshold → Retraining model...\n")

    # Build features
    X, y = build_features(df)

    # Train new model
    model = train_models(X, y)

    # Save model
    save_model(model)

    print("\n✅ Retraining complete. Model updated.")


# -----------------------------
# Save Model
# -----------------------------
def save_model(model, path="models/best_model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at {path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    retrain_pipeline()