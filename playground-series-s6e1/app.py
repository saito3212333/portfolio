
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Path Configuration ---

# --- Page Config ---
st.set_page_config(page_title="ML Journey Tracker", layout="wide", page_icon="🚀")

# --- Helper Functions ---
@st.cache_data
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_csv(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

# app.pyが存在するディレクトリの絶対パスを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ファイル読み込み部分をBASE_DIRを使って書き換え
metrics = load_json(os.path.join(BASE_DIR, 'model_metrics.json'))
correlations = load_json(os.path.join(BASE_DIR, 'eda_correlations.json'))
importance_df = load_csv(os.path.join(BASE_DIR, 'feature_importance.csv'))

# --- Sidebar Navigation ---
st.sidebar.title("Navigation 🧭")
st.sidebar.markdown("Explore the ML pipeline phases:")
page = st.sidebar.radio("Phases", ["1. EDA", "2. Baseline", "3. Feature Engineering", "4. Optuna Tuning", "📊 Summary & Takeaways"])

# --- Main Content ---
st.title("🎓 Student Exam Score Predictor (S6E1)")
st.markdown("Tracking the journey from raw data to an optimized XGBoost Regressor.")
st.markdown("---")

if not metrics:
    st.warning("⚠️ Data files not found. Please run the backend script first (generate_metrics.py).")
    st.stop()

# --- Page 1: EDA ---
if page == "1. EDA":
    st.header("Phase 1: Exploratory Data Analysis 🔍")
    st.markdown("Understanding relationships before building models is crucial.")

    if correlations:
        st.subheader("Target Variable Correlations (with Exam Score)")
        if 'exam_score' in correlations:
            target_corr = correlations['exam_score']
            target_corr_df = pd.DataFrame(list(target_corr.items()), columns=['Feature', 'Correlation'])
            target_corr_df = target_corr_df[target_corr_df['Feature'] != 'exam_score'].sort_values(by='Correlation', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Correlation', y='Feature', data=target_corr_df, hue='Feature', palette='coolwarm', legend=False, ax=ax)
            ax.set_title("Feature Correlations with Exam Score")
            st.pyplot(fig)

            # --- Added: Pro-Tip for Reusability ---
            with st.expander("💡 Pro-Tip: How to build this Target-Correlation Chart", expanded=False):
                st.markdown("""
                **Why this is effective:** Massive correlation heatmaps are often too noisy. Isolating the correlation of all features against *just the target variable* provides a clean, prioritized view for Feature Engineering.

                **Reusable Python Snippet:**
                ```python
                import pandas as pd
                import seaborn as sns
                import matplotlib.pyplot as plt

                # 1. Calculate the correlation matrix
                corr_matrix = df.corr()

                # 2. Extract only the target variable's correlations
                target_col = 'your_target_column'
                target_corr = corr_matrix[target_col].reset_index()
                target_corr.columns = ['Feature', 'Correlation']

                # 3. Drop the target itself (correlation = 1.0) and sort
                target_corr = target_corr[target_corr['Feature'] != target_col]
                target_corr = target_corr.sort_values(by='Correlation', ascending=False)

                # 4. Plot
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Correlation', y='Feature', data=target_corr, hue='Feature', palette='coolwarm', legend=False)
                plt.title(f"Feature Correlations with {target_col}")
                plt.show()
                ```
                Keep this snippet handy for your next EDA!
                """)

# --- Page 2: Baseline & Theory ---
elif page == "2. Baseline":
    st.header("Phase 2: Baseline Model 🚀")

    base_mean = metrics['Baseline']['mean_rmse']
    base_std = metrics['Baseline']['std_rmse']

    col1, col2 = st.columns(2)
    col1.metric("Baseline Mean RMSE", f"{base_mean:.4f}")
    col2.metric("Baseline CV Std Dev", f"± {base_std:.4f}")
    st.info("💡 Note: For RMSE (Root Mean Squared Error), **lower is better**.")

    with st.expander("🧠 Theory: How XGBoost Regressor Works", expanded=False):
        st.markdown("""
        **eXtreme Gradient Boosting (XGBoost)** is an ensemble learning method that builds decision trees sequentially.
        Unlike Random Forest which builds trees independently, XGBoost builds each new tree to correct the *residual errors* of the previous trees.
        """)
        st.markdown("**1. Additive Prediction:**")
        st.latex(r"\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)")
        st.markdown("**2. Objective Function (Loss + Regularization):**")
        st.latex(r"\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)")
        st.markdown(r"""
        * $l$: The loss function (e.g., Mean Squared Error for regression).
        * $\Omega$: The regularization term that penalizes model complexity (tree depth, leaf weights) to prevent overfitting.
        """)

# --- Page 3: Feature Engineering ---
elif page == "3. Feature Engineering":
    st.header("Phase 3: Feature Engineering 🛠️")

    base_mean = metrics['Baseline']['mean_rmse']
    fe_mean = metrics['Feature_Engineering']['mean_rmse']
    fe_std = metrics['Feature_Engineering']['std_rmse']
    improvement = base_mean - fe_mean

    col1, col2 = st.columns(2)
    col1.metric("FE Mean RMSE", f"{fe_mean:.4f}", delta=f"-{improvement:.4f} (Improvement)", delta_color="inverse")
    col2.metric("FE CV Std Dev", f"± {fe_std:.4f}")

    st.markdown("""
    **New Features Added:**
    * `effective_study`: Interaction between Study Hours and Attendance.
    * `sleep_efficiency`: Interaction between Sleep Hours and Sleep Quality.
    * `study_intensity`: Interaction between Study Hours and Study Method.
    """)

# --- Page 4: Optuna Tuning ---
elif page == "4. Optuna Tuning":
    st.header("Phase 4: Hyperparameter Optimization 🤖")

    fe_mean = metrics['Feature_Engineering']['mean_rmse']
    tuned_mean = metrics['Optuna_Tuning']['mean_rmse']
    improvement = fe_mean - tuned_mean

    st.metric("Tuned Mean RMSE", f"{tuned_mean:.4f}", delta=f"-{improvement:.4f} (vs FE)", delta_color="inverse")
    st.subheader("Best Parameters")
    st.json(metrics['Optuna_Tuning']['best_params'])

    if importance_df is not None:
        st.subheader("Final Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), hue='Feature', palette='viridis', legend=False, ax=ax)
        st.pyplot(fig)

# --- Summary Report & Takeaways ---
elif page == "📊 Summary & Takeaways":
    st.header("Executive Summary: RMSE Progression")

    progression = {
        'Phase': ['1. Baseline', '2. Feature Engineering', '3. Optuna Tuned'],
        'RMSE': [
            metrics['Baseline']['mean_rmse'],
            metrics['Feature_Engineering']['mean_rmse'],
            metrics['Optuna_Tuning']['mean_rmse']
        ]
    }
    df_prog = pd.DataFrame(progression)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='Phase', y='RMSE', data=df_prog, marker='o', markersize=10, linewidth=2, color='b', ax=ax)

    y_min = df_prog['RMSE'].min()
    y_max = df_prog['RMSE'].max()
    margin = (y_max - y_min) * 0.1 if y_max != y_min else y_min * 0.01
    ax.set_ylim(y_min - margin, y_max + margin * 1.5)

    for i, val in enumerate(df_prog['RMSE']):
        ax.text(i, val + margin * 0.2, f"{val:.4f}", horizontalalignment='center', fontweight='bold', color='black')

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧠 Key Takeaways & Lessons Learned")
    st.info("""
    **1. The Correlation Trap (Non-linearity):**
    Features like `sleep_quality` and `facility_rating` had near-zero Pearson correlation with the target. However, the XGBoost Feature Importance chart revealed they were highly predictive. *Lesson: Never drop features solely based on linear correlation; tree-based models find complex interactions.*

    **2. The "Silent Bug" in Data Pipelines:**
    Defensive programming (`if 'col' in df:`) in the Feature Engineering phase caused the pipeline to silently skip transformations due to a column name typo. The model trained on baseline data without throwing an error. *Lesson: Always validate data shapes (e.g., `assert df.shape[1] > old_shape`) or log transformations after FE.*

    **3. Trusting Cross-Validation:**
    By implementing Stratified K-Fold inside the Optuna objective function, we ensured the hyperparameters generalize well to unseen data, preventing the classic overfitting trap.
    """)
