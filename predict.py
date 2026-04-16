# predict.py

import pandas as pd
import joblib
import numpy as np

from features import build_features
from ingest import load_data, merge_data


# -----------------------------
# Load Model
# -----------------------------
def load_model(path="models/best_model.pkl"):
    model = joblib.load(path)
    print("Model loaded successfully")
    return model


# -----------------------------
# Make Predictions
# -----------------------------
def make_predictions(model, df):
    # Build features
    X, y = build_features(df)

    # Predict
    preds = model.predict(X)

    # If you used log transform, uncomment this:
    # preds = np.expm1(preds)

    results = X.copy()
    results["Actual"] = y.values
    results["Predicted"] = preds

    return results


# -----------------------------
# Save Predictions
# -----------------------------
def save_predictions(df, path="data/predictions.csv"):
    df.to_csv(path, index=False)
    print(f"Predictions saved at {path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load data
    train, test, store = load_data()

    # Use train data for evaluation (since test has no Sales)
    df = merge_data(train, store)

    # Load model
    model = load_model()

    # Predict
    results = make_predictions(model, df)

    print(results.head())

    # Save
    save_predictions(results)