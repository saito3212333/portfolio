import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

st.set_page_config(page_title="Sales Forecast Evaluation", layout="wide")

# --- 1. Simulation: Generating Synthetic Sales Data ---
@st.cache_data
def generate_sales_data():
    days = 90
    dates = pd.date_range(start='2026-01-01', periods=days)

    # Base trend + Weekly seasonality (higher sales on weekends)
    base = np.linspace(500, 800, days)
    weekly = np.array([0, 0, 0, 0, 50, 150, 200] * (days // 7 + 1))[:days]
    actual = base + weekly + np.random.normal(0, 20, days)
    return pd.DataFrame({'Date': dates, 'Actual': actual})

df = generate_sales_data()

# --- 2. Sidebar: Interactive Error Simulation ---
st.sidebar.header("Forecast Settings")
bias = st.sidebar.slider("Systemic Bias (MAE Impact)", -100, 100, 20)
outlier = st.sidebar.slider("Promo Miss Magnitude (RMSE Impact)", 0, 500, 0)
outlier_idx = st.sidebar.slider("Outlier Day", 0, 89, 45)

# Generate Predicted Values
df['Predicted'] = df['Actual'] + bias + np.random.normal(0, 30, len(df))
df.loc[outlier_idx, 'Predicted'] += outlier # Injecting a major forecast error

# --- 3. Metric Calculations (Guided by Best Practices) ---
# Implementation Tip: Always remove NaNs before calculation
temp_df = df.dropna(subset=['Actual', 'Predicted'])

y_true = temp_df['Actual'].values
y_pred = temp_df['Predicted'].values

# WMAPE: 1-line calculation using NumPy
# Formula: sum(|actual - pred|) / sum(actual)
wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# RMSE: Penalizes large outliers
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE: Intuitive average drift
mae = mean_absolute_error(y_true, y_pred)

# Max Error: Identifying the worst failure
max_err = max_error(y_true, y_pred)

# --- 4. Dashboard UI ---
st.title("📈 Sales Forecast Performance Lab")
st.markdown("Compare actual vs. predicted sales and evaluate model 'Reliability' vs. 'Risk'.")

# Display metrics as cards
cols = st.columns(4)
cols[0].metric("WMAPE (Overall)", f"{wmape:.1%}", help="The 'General Pass Grade' for model quality")
cols[1].metric("RMSE (Risk)", f"${rmse:.1f}", help="Monitors 'fatal risks' caused by large outliers")
cols[2].metric("MAE (Average)", f"${mae:.1f}", help="Average 'daily strength' reported to the field")
cols[3].metric("Max Error", f"${max_err:.1f}", help="Captures the worst-case scenario/anomaly")

# Visualization
st.subheader("Sales Trend: Actual vs. Predicted")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Date'], df['Actual'], label="Actual Sales", color="#2E86C1", lw=2)
ax.plot(df['Date'], df['Predicted'], label="Predicted Sales", color="#E67E22", linestyle="--")
ax.fill_between(df['Date'], df['Actual'], df['Predicted'], color='gray', alpha=0.1, label="Forecast Error")
ax.set_ylabel("Sales Volume ($)")
ax.legend()
st.pyplot(fig)

with st.expander("Technical Definitions"):
    st.write("The formulas used for these metrics are:")
    st.latex(r"WMAPE = \frac{\sum |Actual - Predicted|}{\sum Actual}")
    st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum (Actual - Predicted)^2}")