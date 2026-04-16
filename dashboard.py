# dashboard.py

import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
@st.cache_data
def load_data():
    return pd.read_csv("data/predictions.csv")


df = load_data()

st.title("📊 Sales Forecasting Dashboard")

# -----------------------------
# Basic Stats
# -----------------------------
st.subheader("📈 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Sales", f"{df['Actual'].mean():.0f}")
col2.metric("Avg Prediction", f"{df['Predicted'].mean():.0f}")
col3.metric("Avg Error", f"{abs(df['Actual'] - df['Predicted']).mean():.0f}")

# -----------------------------
# Plot: Actual vs Predicted
# -----------------------------
st.subheader("📉 Actual vs Predicted")

sample = df.sample(500)

fig, ax = plt.subplots()
ax.plot(sample["Actual"].values, label="Actual")
ax.plot(sample["Predicted"].values, label="Predicted")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Error Distribution
# -----------------------------
st.subheader("📊 Error Distribution")

df["Error"] = abs(df["Actual"] - df["Predicted"])

fig2, ax2 = plt.subplots()
ax2.hist(df["Error"], bins=50)
st.pyplot(fig2)

# -----------------------------
# Drift Alert (Simple)
# -----------------------------
st.subheader("🚨 Drift Monitoring")

old_mean = df["Actual"][: int(len(df)*0.8)].mean()
new_mean = df["Actual"][int(len(df)*0.8):].mean()

change = abs(new_mean - old_mean) / old_mean

st.write(f"Drift Change: {change:.2f}")

if change > 0.2:
    st.error("🚨 Drift Detected!")
else:
    st.success("✅ No Significant Drift")