# main.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Sales Forecast AI",
    layout="wide"
)

st.title("🛍️ AI Sales Prediction System")
st.markdown("Predict retail sales using trained ML model")

# -----------------------------
# Load Model + Columns
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("models/feature_columns.pkl")

model = load_model()
feature_cols = load_columns()


# -----------------------------
# Sidebar Inputs (Better UX)
# -----------------------------
st.sidebar.header("📥 Input Parameters")

store = st.sidebar.number_input("Store ID", 1, 1115, 1)

promo = st.sidebar.selectbox("Promo Running?", [0, 1])
state_holiday = st.sidebar.selectbox("State Holiday?", [0, 1])
school_holiday = st.sidebar.selectbox("School Holiday?", [0, 1])

st.sidebar.markdown("### 📅 Date Info")
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.number_input("Year", value=2024)
day_of_week = st.sidebar.selectbox("Day of Week (0=Mon)", list(range(7)))
week_of_year = st.sidebar.slider("Week of Year", 1, 52, 25)

st.sidebar.markdown("### 📊 Sales History")
lag_1 = st.sidebar.number_input("Yesterday Sales", value=5000)
lag_7 = st.sidebar.number_input("Last Week Sales", value=4800)
lag_14 = st.sidebar.number_input("14 Days Ago Sales", value=4700)

rolling_mean_7 = st.sidebar.number_input("7-Day Avg", value=4900)
rolling_std_7 = st.sidebar.number_input("7-Day Std Dev", value=200)


# -----------------------------
# Build Input Data
# -----------------------------
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

# Align columns with training
input_df = input_df.reindex(columns=feature_cols, fill_value=0)


# -----------------------------
# Main UI Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("⚡ Prediction")


# -----------------------------
# Predict Button
# -----------------------------
if st.button("🚀 Predict Sales"):

    prediction = model.predict(input_df)[0]

    # If you used log transform:
    # prediction = np.expm1(prediction)

    with col2:
        st.metric(
            label="💰 Predicted Sales",
            value=f"{prediction:.2f}"
        )

    # Extra insight
    st.subheader("📊 Insights")

    if promo == 1:
        st.success("📈 Promotion is active → sales likely higher")

    if state_holiday == 1:
        st.warning("🎉 Holiday detected → demand may spike")

    if lag_1 > lag_7:
        st.info("📊 Recent trend is increasing")

    else:
        st.info("📉 Sales trend is stable/decreasing")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using ML + Streamlit")