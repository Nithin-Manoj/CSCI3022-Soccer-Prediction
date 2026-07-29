import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

def gd_to_outcome(gd_series):
    """Converts continuous goal differential prediction into discrete match outcome."""
    conditions = [gd_series > 0.33, (gd_series >= -0.33) & (gd_series <= 0.33), gd_series < -0.33]
    choices = ['Win', 'Draw', 'Loss']
    return pd.Series(np.select(conditions, choices, default='Draw'), index=gd_series.index)

def train_and_evaluate_models():
    data_path = os.path.join(PROCESSED_DATA_DIR, 'clean_soccer_data.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Define features and target
    feature_cols = ['rank_gap', 'home_rank', 'away_rank', 'neutral', 'is_friendly', 'form_gap', 'home_form', 'away_form', 'point_gap']
    target_col = 'goal_difference'
    
    # Drop rows with NaNs in features
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    # Time-based Train/Test Split (Train: <= 2019, Test: >= 2020)
    split_date = '2020-01-01'
    train_mask = df_clean['date'] < split_date
    test_mask = df_clean['date'] >= split_date
    
    X_train, y_train = df_clean.loc[train_mask, feature_cols], df_clean.loc[train_mask, target_col]
    X_test, y_test = df_clean.loc[test_mask, feature_cols], df_clean.loc[test_mask, target_col]
    y_test_outcome = df_clean.loc[test_mask, 'home_result']
    
    print(f"Train Set Size: {len(X_train)} matches (1993 - 2019)")
    print(f"Test Set Size:  {len(X_test)} matches (2020 - 2026)")
    
    # Baseline Model: OLS Regression
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    
    # Advanced Model: XGBoost Gradient Boosting Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # HistGradientBoosting as secondary check
    hgb_model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, random_state=42)
    hgb_model.fit(X_train, y_train)
    y_pred_hgb = hgb_model.predict(X_test)
    
    # Compute Metrics
    results = []
    models_preds = {
        'OLS Regression (Baseline)': y_pred_ols,
        'XGBoost (Gradient Boosting)': y_pred_xgb,
        'HistGradientBoosting': y_pred_hgb
    }
    
    for name, preds in models_preds.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Classification outcome conversion
        outcome_preds = gd_to_outcome(pd.Series(preds, index=y_test.index))
        acc = accuracy_score(y_test_outcome, outcome_preds)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R^2': r2,
            'Match Outcome Acc': acc
        })
        
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL EVALUATION & METRICS SUMMARY")
    print("=" * 60)
    print(res_df.to_string(index=False))
    
    # Feature Importance Analysis (XGBoost)
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n--- XGBOOST FEATURE IMPORTANCES ---")
    print(importance_df.to_string(index=False))
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature Importance Plot
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[0], palette='Blues_r', hue='Feature', legend=False)
    axes[0].set_title('Gradient Boosting Feature Importances', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Relative Importance')
    
    # Actual vs Predicted Goal Difference
    axes[1].scatter(y_test, y_pred_xgb, alpha=0.2, color='#2b5c8f', label='Predictions')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal 1:1')
    axes[1].set_title('Actual vs Predicted Goal Difference (XGBoost)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Actual Goal Difference')
    axes[1].set_ylabel('Predicted Goal Difference')
    axes[1].legend()
    
    plt.tight_layout()
    fig_path = os.path.join(REPORTS_DIR, 'model_evaluation.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved evaluation plot to {fig_path}")

def main():
    train_and_evaluate_models()

if __name__ == "__main__":
    main()
