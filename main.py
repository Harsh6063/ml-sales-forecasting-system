# main.py

import streamlit as st
import pandas as pd
import joblib
import datetime

from ingest import load_data, merge_data


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sales AI", layout="wide")

st.title("🛍️ AI Sales Prediction System")
st.markdown("Smart forecasting with automatic feature generation")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# -----------------------------
# Load Data (for history)
# -----------------------------
@st.cache_data
def load_full_data():
    train, test, store = load_data()
    df = merge_data(train, store)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_full_data()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📥 Input")

store = st.sidebar.number_input("Store ID", 1, 1115, 1)

selected_date = st.sidebar.date_input(
    "Select Date",
    value=datetime.date.today()
)

promo = st.sidebar.selectbox("Promo Running?", [0, 1])
state_holiday = st.sidebar.selectbox("State Holiday?", [0, 1])
school_holiday = st.sidebar.selectbox("School Holiday?", [0, 1])

# -----------------------------
# Extract Date Features
# -----------------------------
date = pd.to_datetime(selected_date)

day = date.day
month = date.month
year = date.year
day_of_week = date.weekday()
week_of_year = date.isocalendar().week

# -----------------------------
# Get Store History
# -----------------------------
store_df = df[df["Store"] == store].sort_values("Date")

# Filter past data only
history = store_df[store_df["Date"] < date]

# -----------------------------
# AUTO Feature Engineering
# -----------------------------
def get_lag_features(history):

    if len(history) < 14:
        return None  # Not enough data

    lag_1 = history.iloc[-1]["Sales"]
    lag_7 = history.iloc[-7]["Sales"]
    lag_14 = history.iloc[-14]["Sales"]

    rolling_mean_7 = history["Sales"].tail(7).mean()
    rolling_std_7 = history["Sales"].tail(7).std()

    return lag_1, lag_7, lag_14, rolling_mean_7, rolling_std_7


features = get_lag_features(history)

# -----------------------------
# UI Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Input Summary")
    st.write(f"Store: {store}")
    st.write(f"Date: {selected_date}")
    st.write(f"Promo: {promo}")

with col2:
    st.subheader("⚡ Prediction")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Sales"):

    if features is None:
        st.error("❌ Not enough historical data for this store")
    else:
        lag_1, lag_7, lag_14, rolling_mean_7, rolling_std_7 = features

        input_dict = {
            "Store": store,
            "Promo": promo,
            "StateHoliday": state_holiday,
            "SchoolHoliday": school_holiday,
            "day": day,
            "month": month,
            "year": year,
            "day_of_week": day_of_week,
            "week_of_year": week_of_year,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
        }

        input_df = pd.DataFrame([input_dict])

        # Align columns (VERY IMPORTANT)
        feature_cols = joblib.load("models/feature_columns.pkl")
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        prediction = model.predict(input_df)[0]

        with col2:
            st.metric("💰 Predicted Sales", f"{prediction:.2f}")

        # -----------------------------
        # Insights
        # -----------------------------
        st.subheader("📊 Insights")

        st.write(f"Yesterday Sales: {lag_1:.0f}")
        st.write(f"Last Week Sales: {lag_7:.0f}")
        st.write(f"Trend (7-day avg): {rolling_mean_7:.0f}")
        
        st.subheader("📈 Sales Trend")


        if promo == 1:
            st.success("📈 Promotion active → higher expected sales")

        if rolling_mean_7 > lag_7:
            st.info("📊 Upward trend detected")
        else:
            st.warning("📉 Sales trend is flat or decreasing")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ML + Streamlit + Auto Feature Engineering")