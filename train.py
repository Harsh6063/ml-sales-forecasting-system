
import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from features import build_features
from ingest import load_data, merge_data

import joblib


# -----------------------------
# SAFE MAPE (FIXED)
# -----------------------------
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


# -----------------------------
# METRICS
# -----------------------------
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, mape, r2


# -----------------------------
# TRAIN MODELS
# -----------------------------
def train_models(X, y):

    # Time Series Split (IMPORTANT)
    tscv = TimeSeriesSplit(n_splits=3)

    models = {
        "LinearRegression": LinearRegression(),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=10
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            n_jobs=-1
        ),

        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        ),
    }

    best_model = None
    best_mape = float("inf")

    mlflow.set_experiment("sales_forecasting")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            print(f"\nTraining {name}...")

            mape_scores = []
            rmse_scores = []
            mae_scores = []
            r2_scores = []

            # Cross-validation
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                rmse, mae, mape, r2 = evaluate(y_val, preds)

                mape_scores.append(mape)
                rmse_scores.append(rmse)
                mae_scores.append(mae)
                r2_scores.append(r2)

            # Average metrics
            avg_rmse = np.mean(rmse_scores)
            avg_mae = np.mean(mae_scores)
            avg_mape = np.mean(mape_scores)
            avg_r2 = np.mean(r2_scores)

            print(f"{name} RMSE: {avg_rmse}")
            print(f"{name} MAE: {avg_mae}")
            print(f"{name} MAPE: {avg_mape}")
            print(f"{name} R2: {avg_r2}")

            # MLflow logging
            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", avg_rmse)
            mlflow.log_metric("mae", avg_mae)
            mlflow.log_metric("mape", avg_mape)
            mlflow.log_metric("r2", avg_r2)

            mlflow.sklearn.log_model(model, name)

            # Best model selection
            if avg_mape < best_mape:
                best_mape = avg_mape
                best_model = model
                best_name = name

    print("\n🏆 Best Model:", best_name)
    return best_model


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model, path="models/best_model.pkl"):
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at {path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train, test, store = load_data()
    df = merge_data(train, store)

    X, y = build_features(df)
    
    #os.makedirs("models", exist_ok=True)
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    print("✅ Feature columns saved")

    model = train_models(X, y)
    save_model(model)