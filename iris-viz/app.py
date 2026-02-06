import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Page Setup
st.set_page_config(page_title="Iris ML Dashboard", layout="wide")

# Data Loading
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# --- Sidebar ---
st.sidebar.header("Configuration")

# 1. Model Selection
model_type = st.sidebar.selectbox("Select Model", ["LDA", "LightGBM"])

# 2. Feature Selection (Pick 2 for the 2D Boundary Plot)
features = df.columns[:-1].tolist()
x_feat = st.sidebar.selectbox("X-Axis (Feature 1)", features, index=2)
y_feat = st.sidebar.selectbox("Y-Axis (Feature 2)", features, index=3)

# 3. Accuracy Metric (The "Card")
X = df[[x_feat, y_feat]]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_type == "LDA":
    model = LDA()
else:
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)

model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")
st.sidebar.metric(label="Test Accuracy", value=f"{acc:.2%}")

# --- Main Panel ---
st.title("🌸 Iris Classification Analysis")
st.write(f"Visualizing decision boundaries for **{model_type}** using **{x_feat}** and **{y_feat}**.")

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))

# Create Meshgrid for boundaries
x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot background
ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

# Plot points
for i, color in enumerate(["red", "blue", "green"]):
    idx = np.where(y == i)
    ax.scatter(X.iloc[idx[0], 0], X.iloc[idx[0], 1], c=color,
               label=target_names[i].capitalize(), edgecolors='k', alpha=0.8)

ax.set_xlabel(x_feat)
ax.set_ylabel(y_feat)
ax.set_title(f"Decision Boundary: {model_type}")
ax.legend(title="Species")

st.pyplot(fig)

# Data Table
with st.expander("Show Raw Dataset"):
    st.dataframe(df, use_container_width=True)