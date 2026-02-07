import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(page_title="Titanic Analysis | Cross-Entropy Approach", layout="wide")

# --- Function Definitions ---

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

    # Ensure all remaining objects are encoded
    for col in df.columns:
        if df[col].dtype == 'object':
             df[col] = le.fit_transform(df[col].astype(str))

    return df

def train_lgbm_get_loss(X, y, num_features):
    # Selecting the Top N correlated features
    top_n_cols = X.columns[:num_features]
    X_subset = X[top_n_cols]

    X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Parameter setting focusing on Cross-Entropy (Binary Logloss)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'random_state': 42,
        'boosting_type': 'gbdt'
    }

    evals_result = {}
    model = lgb.train(params, train_data, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(10), lgb.record_evaluation(evals_result)])

    # Extracting the best Validation LogLoss
    best_logloss = model.best_score['valid_0']['binary_logloss']
    return best_logloss, model, X_val, y_val

# --- Main Logic ---

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Correlation Analysis",
    "2. Cross-Entropy Dynamics",
    "3. Final Model Evaluation"
])

# Data Loading
df_processed = load_and_preprocess_data()

# Global Correlation Calculation
corr_matrix = df_processed.corr()
target_corr = corr_matrix['survived'].abs().sort_values(ascending=False)
sorted_features = target_corr.index[1:].tolist()

# --- Page 1: Correlation Analysis ---
if page == "1. Correlation Analysis":
    st.title("🚢 Titanic: Data Preparation & Correlation")
    st.markdown("""
    ### Objective
    In this first step, we convert all categorical features into numerical values to quantify the relationships
    between our variables and the target: **Survival**.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Preprocessed Dataset")
        st.dataframe(df_processed.head(10))
        st.write(f"Shape: {df_processed.shape}")

        st.subheader("Top 10 Correlations")
        st.write(target_corr[1:11])

    with col2:
        st.subheader("Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', ax=ax)
        st.pyplot(fig_corr)

# --- Page 2: Cross-Entropy Dynamics ---
elif page == "2. Cross-Entropy Dynamics":
    st.title("📉 Diminishing Uncertainty: Cross-Entropy Analysis")
    st.markdown("""
    ### The Core Concept
    As we add more informative features to our model, the **Cross-Entropy (LogLoss)** should decrease.
    This indicates that the model is becoming less "surprised" by the actual outcomes and is gaining predictive confidence.
    """)

    max_features = min(10, len(sorted_features))
    num_features_range = list(range(2, max_features + 1))
    logloss_history = []

    X_full = df_processed[sorted_features]
    y_full = df_processed['survived']

    with st.spinner('Iteratively training models...'):
        for n_feat in num_features_range:
            loss, _, _, _ = train_lgbm_get_loss(X_full, y_full, n_feat)
            logloss_history.append(loss)

    # Visualization
    fig_loss, ax = plt.subplots(figsize=(10, 6))
    ax.plot(num_features_range, logloss_history, marker='s', linestyle='--', color='#2c3e50')
    ax.set_title('Cross-Entropy Reduction vs. Number of Features', fontsize=14)
    ax.set_xlabel('Number of Top Features Used', fontsize=12)
    ax.set_ylabel('Validation LogLoss (Binary Cross-Entropy)', fontsize=12)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig_loss)

    st.success("""
    **Insight:** Notice how the loss curve drops significantly as we add the most correlated features.
    This visualization demonstrates the **Information Gain** provided by each additional feature in reducing the model's overall uncertainty.
    """)

# --- Page 3: Final Model Evaluation ---
elif page == "3. Final Model Evaluation":
    st.title("✅ Final Evaluation & Classification Results")
    st.markdown("We evaluate our model using the top 10 features, analyzing its ability to generalize on unseen validation data.")

    n_final = min(10, len(sorted_features))
    X_full = df_processed[sorted_features]
    y_full = df_processed['survived']

    _, final_model, X_val, y_val = train_lgbm_get_loss(X_full, y_full, n_final)

    y_pred_prob = final_model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    st.metric(label="Overall Accuracy", value=f"{acc:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=['Pred: Dead (0)', 'Pred: Survived (1)'],
                    yticklabels=['Actual: Dead (0)', 'Actual: Survived (1)'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig_cm)

    with col2:
        st.subheader("Feature Importance (Gain)")
        st.write("Calculated based on the reduction of Cross-Entropy contributed by each feature.")
        importance = final_model.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'feature': X_full.columns[:n_final], 'gain': importance}).sort_values('gain', ascending=False)
        st.bar_chart(imp_df.set_index('feature'))

st.sidebar.markdown("---")
st.sidebar.caption("Project by Sahiyo | 2026 Portfolio")