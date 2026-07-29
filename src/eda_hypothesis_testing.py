import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Inter'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def perform_eda(df):
    print("=" * 60)
    print("1. SUMMARY STATISTICS & TARGET DISTRIBUTION")
    print("=" * 60)
    
    metrics = ['home_score', 'away_score', 'total_goals', 'goal_difference', 'rank_gap', 'form_gap']
    summary = df[metrics].describe().T
    summary['skewness'] = df[metrics].skew()
    summary['kurtosis'] = df[metrics].kurtosis()
    print(summary[['mean', 'std', 'min', '50%', 'max', 'skewness']])
    
    # Check Poisson fit for goals
    mean_home_goals = df['home_score'].mean()
    mean_total_goals = df['total_goals'].mean()
    
    print("\nPoisson Distribution Analysis:")
    print(f"Home score mean = {mean_home_goals:.4f}, variance = {df['home_score'].var():.4f}")
    print(f"Total goals mean = {mean_total_goals:.4f}, variance = {df['total_goals'].var():.4f}")
    
    # Figure 1: Distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Goal Difference distribution
    sns.histplot(df['goal_difference'], kde=True, ax=axes[0, 0], color='#2b5c8f', discrete=True)
    axes[0, 0].set_title('Match Goal Difference (Home - Away)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Goal Difference')
    
    # Home vs Away score comparison
    sns.histplot(df['home_score'], label='Home Goals', ax=axes[0, 1], color='#1f77b4', discrete=True, alpha=0.6)
    sns.histplot(df['away_score'], label='Away Goals', ax=axes[0, 1], color='#ff7f0e', discrete=True, alpha=0.6)
    axes[0, 1].set_title('Home vs Away Goals Distribution (Right Skewness)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Goals Scored')
    axes[0, 1].legend()
    
    # Goal Diff by Neutral Venue
    sns.boxplot(x='neutral', y='goal_difference', data=df, ax=axes[1, 0], palette=['#2b5c8f', '#d95f02'], hue='neutral', legend=False)
    axes[1, 0].set_xticklabels(['True Home (0)', 'Neutral (1)'])
    axes[1, 0].set_title('Goal Difference by Venue Neutrality', fontsize=13, fontweight='bold')
    
    # Rank Gap vs Goal Difference
    sns.regplot(x='rank_gap', y='goal_difference', data=df.sample(min(3000, len(df)), random_state=42), 
                ax=axes[1, 1], scatter_kws={'alpha': 0.15, 'color': '#333333'}, line_kws={'color': '#e41a1c', 'linewidth': 2})
    axes[1, 1].set_title('Rank Gap vs Goal Difference', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Rank Gap (Away Rank - Home Rank)')
    axes[1, 1].set_ylabel('Goal Difference')
    
    plt.tight_layout()
    fig_path = os.path.join(REPORTS_DIR, 'eda_distributions.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved EDA plot to {fig_path}")

def perform_hypothesis_testing(df):
    print("\n" + "=" * 60)
    print("2. STATISTICAL HYPOTHESIS TESTING")
    print("=" * 60)
    
    non_neutral = df[df['neutral'] == 0]
    neutral = df[df['neutral'] == 1]
    
    # Test 1: Is Home Advantage goal difference > 0 in non-neutral matches?
    gd_non_neutral = non_neutral['goal_difference']
    t_stat_1, p_val_1 = stats.ttest_1samp(gd_non_neutral, popmean=0, alternative='greater')
    
    print("\n--- HYPOTHESIS TEST 1: Existence of Home Field Advantage ---")
    print("H0: Mean goal difference in home matches <= 0")
    print("H1: Mean goal difference in home matches > 0")
    print(f"Sample size (non-neutral matches): {len(gd_non_neutral)}")
    print(f"Sample Mean Goal Difference: {gd_non_neutral.mean():.4f}")
    print(f"T-statistic: {t_stat_1:.4f}, p-value: {p_val_1:.4e}")
    if p_val_1 < 0.05:
        print("Conclusion: Reject H0! Home team has a statistically significant positive goal advantage.")
    else:
        print("Conclusion: Fail to reject H0.")

    # Test 2: Does Home Field Advantage differ between Competitive vs Friendly matches?
    comp_home = non_neutral[non_neutral['is_friendly'] == 0]['goal_difference']
    friendly_home = non_neutral[non_neutral['is_friendly'] == 1]['goal_difference']
    
    t_stat_2, p_val_2 = stats.ttest_ind(comp_home, friendly_home, equal_var=False)
    
    print("\n--- HYPOTHESIS TEST 2: Competitive vs. Friendly Home Advantage ---")
    print("H0: Mean home goal diff in Competitive matches == Mean home goal diff in Friendly matches")
    print("H1: Mean home goal diffs are unequal between match types")
    print(f"Competitive Home Mean GD: {comp_home.mean():.4f} (n={len(comp_home)})")
    print(f"Friendly Home Mean GD:    {friendly_home.mean():.4f} (n={len(friendly_home)})")
    print(f"Welch's T-statistic: {t_stat_2:.4f}, p-value: {p_val_2:.4e}")
    if p_val_2 < 0.05:
        print("Conclusion: Reject H0! Home advantage is statistically significantly different between competitive & friendly games.")
    else:
        print("Conclusion: Fail to reject H0.")

    # Test 3: Home vs Neutral Goal Difference controlling for Rank Gap (Linear Regression Coefficient Test)
    import statsmodels.api as sm
    
    df_reg = df.dropna(subset=['goal_difference', 'rank_gap', 'neutral', 'is_friendly', 'form_gap']).copy()
    X = df_reg[['rank_gap', 'neutral', 'is_friendly', 'form_gap']]
    X = sm.add_constant(X)
    y = df_reg['goal_difference']
    
    ols_model = sm.OLS(y, X).fit()
    print("\n--- STATISTICAL MODEL (OLS Regression for Controlling Factors) ---")
    print(ols_model.summary().tables[1])
    print(f"Overall Model R-squared: {ols_model.rsquared:.4f}, F-statistic p-value: {ols_model.f_pvalue:.4e}")

def main():
    data_path = os.path.join(PROCESSED_DATA_DIR, 'clean_soccer_data.csv')
    df = pd.read_csv(data_path)
    perform_eda(df)
    perform_hypothesis_testing(df)

if __name__ == "__main__":
    main()
