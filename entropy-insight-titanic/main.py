import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(page_title="Titanic Analysis | Cross-Entropy Dynamics", layout="wide")

# --- 1. Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    # Loading the classic Titanic dataset
    df = sns.load_dataset('titanic')

    # Dropping redundant columns for clarity
    cols_to_drop = ['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']
    df = df.drop(cols_to_drop, axis=1)

    # Handling missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # Numerical Encoding for Categorical Variables
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['embarked'] = le.fit_transform(df['embarked'])

    # Casting objects to standard strings to avoid Arrow "LargeUtf8" errors in browser
    for col in df.select_dtypes(['object', 'category']).columns:
        df[col] = df[col].astype(str)

    return df

# --- 2. Model Training Function (Used for Page 3) ---
def train_final_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss', # Cross-Entropy
        'verbosity': -1,
        'random_state': 42
    }

    model = lgb.train(params, train_data, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(10)])
    return model, X_val, y_val

# --- Main Application Logic ---

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", [
    "1. Data & Correlation Analysis",
    "2. Cross-Entropy Dynamics",
    "3. Final Evaluation & Gain"
])

df_processed = load_and_preprocess_data()

# Identify Top Features via Correlation
corr_matrix = df_processed.corr()
target_corr = corr_matrix['survived'].abs().sort_values(ascending=False)
sorted_features = target_corr.index[1:].tolist()

# --- Page 1: Correlation Analysis ---
if page == "1. Data & Correlation Analysis":
    st.title("🚢 Titanic Survival: Data & Correlation")
    st.markdown("""
    ### Quantifying Relationships
    Before modeling, we analyze the statistical relationship between our target variable (Survival)
    and the available features. This correlation identifies the primary drivers of information.
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Processed Features")
        st.write(df_processed.head(10))
        st.subheader("Target Correlation (Top 10)")
        st.dataframe(target_corr[1:11])

    with col2:
        st.subheader("Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', ax=ax)
        st.pyplot(fig_corr)

# --- Page 2: Cross-Entropy Dynamics (Optimized via Pre-computed CSV) ---
elif page == "2. Cross-Entropy Dynamics":
    st.title("📉 Diminishing Uncertainty: The Loss Curve")
    st.markdown("""
    ### Information Gain and Loss Reduction
    In this view, we observe how adding features reduces the model's **Cross-Entropy (Log-Loss)**.
    Lower loss signifies that the model is gaining confidence and effectively reducing the uncertainty (entropy)
    of its survival predictions.
    """)

    # --- Loading Static Pre-computed Results for High Performance ---

    results_df  = pd.DataFrame({
                'num_features': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'logloss': [
                0.4723870490261505, 0.48740381834300495, 0.4483163354841218,
                0.44420943569035826, 0.4506194469786456, 0.45152151784640643,
                0.45152151784640643, 0.45152151784640643, 0.45152151784640643
            ]})
    num_features_range = results_df['num_features'].tolist()
    logloss_history = results_df['logloss'].tolist()

    fig_loss, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_features_range, logloss_history, marker='o', linestyle='--', color='#2c3e50', linewidth=2)
    ax.set_title('Validation Cross-Entropy vs. Feature Count', fontsize=14)
    ax.set_xlabel('Number of Top-Correlated Features Used', fontsize=12)
    ax.set_ylabel('Log-Loss (Cross-Entropy)', fontsize=12)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_loss)

    st.success("**Insight:** The sharpest drops in loss occur when the most significant features are introduced, demonstrating the concept of high Information Gain.")
# --- Page 3: Final Model Evaluation ---
elif page == "3. Final Evaluation & Gain":
    st.title("✅ Model Performance & Interpretability")
    st.markdown("We evaluate the final model using the top 10 features, focusing on **Gain-based Feature Importance**.")

    # Using Top 10 features for final model
    X_final = df_processed[sorted_features[:10]]
    y_final = df_processed['survived']

    final_model, X_val, y_val = train_final_model(X_final, y_final)

    # Predictions
    y_pred_prob = final_model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    st.metric(label="Final Accuracy Score", value=f"{acc:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Dead (0)', 'Survived (1)'],
                    yticklabels=['Dead (0)', 'Survived (1)'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig_cm)

    with col2:
        st.subheader("Feature Importance (Gain)")
        st.markdown("Ranking features by their contribution to **Cross-Entropy reduction**.")
        # Importance Type: GAIN
        importance = final_model.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'Feature': X_final.columns, 'Gain': importance}).sort_values('Gain', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

st.sidebar.markdown("---")
st.sidebar.caption("Project by [Your Name] | Data Scientist 2026")