import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import json
import os

# Set random seed
RANDOM_SEED = 42

def load_and_prep_data(filepath="./data/train.csv"):
    """
    Load S6E1 Student Test Scores dataset.
    Target variable: 'exam_score'
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        print(f"File {filepath} not found. Generating dummy data for testing.")
        # Dummy data matching S6E1 structure
        df = pd.DataFrame({
            'hours_studied': np.random.randint(1, 10, 100),
            'attendance': np.random.randint(60, 100, 100),
            'sleep_hours': np.random.randint(4, 10, 100),
            'previous_scores': np.random.randint(50, 100, 100),
            'tutoring_sessions': np.random.randint(0, 5, 100),
            'physical_activity': np.random.choice(['Yes', 'No'], 100),
            'exam_score': np.random.randint(60, 100, 100)
        })

    # Basic Preprocessing
    # Encode categorical variables if any (S6E1 has some categorical columns)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(['exam_score', 'id'], axis=1, errors='ignore')
    y = df['exam_score']

    return X, y, df

def evaluate_model(X, y, params=None):
    """
    Evaluate using K-Fold CV (Regression).
    Metric: RMSE (Root Mean Squared Error)
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Define baseline parameters
    model_kwargs = {
        'random_state': RANDOM_SEED,
        'objective': 'reg:squarederror'
    }

    # Safely merge Optuna parameters if they exist, overwriting defaults
    if params is not None:
        model_kwargs.update(params)

    # Unpack safely: no hardcoded kwargs here to prevent multiple values error
    model = xgb.XGBRegressor(**model_kwargs)

    scores = cross_validate(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', return_estimator=True)

    mean_rmse = -np.mean(scores['test_score']) # Convert back to positive RMSE
    std_rmse = np.std(scores['test_score'])

    best_estimator = scores['estimator'][0]
    importance = best_estimator.feature_importances_

    return mean_rmse, std_rmse, importance

def apply_feature_engineering(X):
    """
    Create features specific to Student Performance (S6E1)
    using the CORRECT column names.
    """
    X_fe = X.copy()

    # Feature 1: Effective Study (学習時間 × 出席率)
    if 'study_hours' in X_fe.columns and 'class_attendance' in X_fe.columns:
        # ％表記(0-100)を想定して100で割る、またはそのまま掛ける
        X_fe['effective_study'] = X_fe['study_hours'] * (X_fe['class_attendance'] / 100.0)

    # Feature 2: Sleep Efficiency (睡眠時間 × 睡眠の質)
    # 重要度2位のsleep_qualityと、sleep_hoursの相乗効果を狙う
    if 'sleep_hours' in X_fe.columns and 'sleep_quality' in X_fe.columns:
        X_fe['sleep_efficiency'] = X_fe['sleep_hours'] * X_fe['sleep_quality']

    # Feature 3: Study Intensity (学習時間 × 勉強法)
    # 重要度4位のstudy_methodとのインタラクション
    if 'study_hours' in X_fe.columns and 'study_method' in X_fe.columns:
        X_fe['study_intensity'] = X_fe['study_hours'] * X_fe['study_method']

    return X_fe

def optimize_hyperparameters(X, y, n_trials=20):
    """
    Optuna for Regression (Minimizing RMSE).
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'reg:squarederror'
        }
        mean_rmse, _, _ = evaluate_model(X, y, params)
        return mean_rmse # Optuna minimizes this

    study = optuna.create_study(direction='minimize') # Minimize RMSE
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value

def main():
    print("🚀 Starting S6E1 ML Pipeline (Regression)...")
    results_dict = {}

    # 1. Load Data
    X_raw, y, df = load_and_prep_data()

    # Export EDA correlations
    corr_matrix = df.corr().to_dict()
    with open('eda_correlations.json', 'w') as f:
        json.dump(corr_matrix, f)

    # 2. Baseline
    print("📊 Evaluating Baseline...")
    base_mean, base_std, _ = evaluate_model(X_raw, y)
    results_dict['Baseline'] = {'mean_rmse': base_mean, 'std_rmse': base_std}

    # 3. Feature Engineering
    print("🛠 Evaluating Feature Engineering...")
    X_fe = apply_feature_engineering(X_raw)
    fe_mean, fe_std, _ = evaluate_model(X_fe, y)
    results_dict['Feature_Engineering'] = {'mean_rmse': fe_mean, 'std_rmse': fe_std}

    # 4. Optuna Tuning
    print("🤖 Running Optuna Tuning...")
    best_params, best_rmse = optimize_hyperparameters(X_fe, y, n_trials=20)

    tuned_mean, tuned_std, final_importance = evaluate_model(X_fe, y, best_params)
    results_dict['Optuna_Tuning'] = {
        'mean_rmse': tuned_mean,
        'std_rmse': tuned_std,
        'best_params': best_params
    }

    # 5. Export Metrics
    with open('model_metrics.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    # 6. Export Feature Importance
    importance_df = pd.DataFrame({
        'Feature': X_fe.columns,
        'Importance': final_importance
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)

    print("✅ Pipeline completed! Files 'model_metrics.json' and 'feature_importance.csv' are ready.")

if __name__ == "__main__":
    main()