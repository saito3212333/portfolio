import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

st.set_page_config(page_title="Iris ML Dashboard", layout="wide")

@st.cache_data
def load_data():
    iris = load_iris()
    # ULTIMATE FIX: Remove EVERYTHING except letters, numbers, and underscores
    clean_feature_names = [re.sub(r'[^a-zA-Z0-9_]', '', name.replace(" ", "_")) for name in iris.feature_names]
    df = pd.DataFrame(iris.data, columns=clean_feature_names)
    df['target'] = iris.target
    return df, iris.target_names, clean_feature_names

df, target_names, features = load_data()

# --- Sidebar ---
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Select Model", ["LDA", "LightGBM"])
x_feat = st.sidebar.selectbox("X-Axis", features, index=2)
y_feat = st.sidebar.selectbox("Y-Axis", features, index=3)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.9, 0.3)

# --- Analysis Logic ---
def get_analysis(model_type, x_f, y_f, t_size):
    X = df[[x_f, y_f]]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42)

    if model_type == "LDA":
        model = LDA()
    else:
        # LightGBM prefers simple names and limited threads on free cloud tiers
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, X, y

# --- Execute & Display ---
try:
    model, acc, X, y = get_analysis(model_type, x_feat, y_feat, test_size)

    # Metric Card with session state logic
    if 'prev_acc' not in st.session_state:
        st.session_state.prev_acc = acc
    delta = acc - st.session_state.prev_acc
    st.sidebar.metric(label="Test Accuracy", value=f"{acc:.2%}", delta=f"{delta:.2%}" if delta != 0 else None)
    st.session_state.prev_acc = acc

    # Main Plot
    st.title("🌸 Iris Classification Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    for i, color in enumerate(["red", "blue", "green"]):
        idx = np.where(y == i)
        ax.scatter(X.iloc[idx[0], 0], X.iloc[idx[0], 1], c=color, label=target_names[i])

    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Try refreshing or changing the features.")