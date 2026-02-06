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

@st.cache_data
def load_data():
    iris = load_iris()
    clean_feature_names = [name.replace(" (cm)", "").replace(" ", "_") for name in iris.feature_names]
    df = pd.DataFrame(iris.data, columns=clean_feature_names)
    df['target'] = iris.target
    return df, iris.target_names, clean_feature_names

df, target_names, features = load_data()

# --- Sidebar ---
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Select Model", ["LDA", "LightGBM"])
x_feat = st.sidebar.selectbox("X-Axis", features, index=2)
y_feat = st.sidebar.selectbox("Y-Axis", features, index=3)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.9, 0.3) # 範囲を広げて変化を出しやすく

# --- ML Training (Caching only the model, not the accuracy) ---
def get_analysis(model_type, x_f, y_f, t_size):
    X = df[[x_f, y_f]]
    y = df['target']
    # 乱数シードを固定せず、あえて少し揺らぎを持たせるのも手ですが、
    # 今回はt_sizeの変更を正確に反映させるため random_state=42 を維持します
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42)

    if model_type == "LDA":
        model = LDA()
    else:
        model = lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1)

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X, y

model, acc, X, y = get_analysis(model_type, x_feat, y_feat, test_size)

# --- Sidebar Accuracy Card ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")

# セッション状態で前回のスコアを保持して、変化(delta)を表示する
if 'prev_acc' not in st.session_state:
    st.session_state.prev_acc = acc

delta = acc - st.session_state.prev_acc
st.sidebar.metric(label="Test Accuracy", value=f"{acc:.2%}", delta=f"{delta:.2%}" if delta != 0 else None)
st.session_state.prev_acc = acc

# --- Main Panel ---
st.title("🌸 Iris Classification Analysis")

# Decision Boundary Plot
fig, ax = plt.subplots(figsize=(10, 6))
x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

for i, color in enumerate(["red", "blue", "green"]):
    idx = np.where(y == i)
    ax.scatter(X.iloc[idx[0], 0], X.iloc[idx[0], 1], c=color,
               label=target_names[i].capitalize(), edgecolors='k', alpha=0.8)

ax.set_xlabel(x_feat)
ax.set_ylabel(y_feat)
st.pyplot(fig)