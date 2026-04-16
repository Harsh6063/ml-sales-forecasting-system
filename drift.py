# drift.py

import pandas as pd
import numpy as np

from ingest import load_data, merge_data


# -----------------------------
# Split Data (Old vs New)
# -----------------------------
def split_data(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)

    old_data = df.iloc[:split]
    new_data = df.iloc[split:]

    return old_data, new_data


# -----------------------------
# Detect Drift (Mean Shift)
# -----------------------------
def detect_drift(old_data, new_data, features, threshold=0.2):
    drift_report = {}

    print("\n🔍 Checking Drift...\n")

    for col in features:
        if col not in old_data.columns:
            continue

        old_mean = old_data[col].mean()
        new_mean = new_data[col].mean()

        # Avoid division by zero
        if old_mean == 0:
            continue

        change = abs(new_mean - old_mean) / abs(old_mean)

        drift_report[col] = change

        print(f"{col}: change = {round(change, 3)}")

        if change > threshold:
            print(f"🚨 Drift detected in {col}!")

    return drift_report


# -----------------------------
# Simulate Drift (Optional)
# -----------------------------
def simulate_drift(df):
    df = df.copy()

    # Increase sales artificially (festival spike)
    df["Sales"] = df["Sales"] * 1.5

    # Increase promo frequency
    if "Promo" in df.columns:
        df["Promo"] = 1

    return df


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load data
    train, test, store = load_data()
    df = merge_data(train, store)

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])

    # Split into old vs new
    old_data, new_data = split_data(df)

    # OPTIONAL: simulate drift
    new_data = simulate_drift(new_data)

    # Features to monitor
    features = [
        "Sales",
        "Customers",
        "Promo",
        "SchoolHoliday"
    ]

    # Detect drift
    drift_report = detect_drift(old_data, new_data, features)