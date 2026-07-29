import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Common team name replacements to harmonize Kaggle match data & FIFA ranking names
TEAM_NAME_MAP = {
    'USA': 'United States',
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    'Korea DPR': 'North Korea',
    'Ivory Coast': "Côte d'Ivoire",
    'Cape Verde': 'Cabo Verde',
    'DR Congo': 'Congo DR',
    'Congo': 'Congo',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Czech Republic': 'Czechia',
    'FYR Macedonia': 'North Macedonia',
    'Macedonia': 'North Macedonia',
    'Swaziland': 'Eswatini',
    'Turkey': 'Türkiye',
    'East Timor': 'Timor-Leste',
}

def load_and_clean_raw_data():
    matches_path = os.path.join(RAW_DATA_DIR, 'results.csv')
    fifa_path = os.path.join(RAW_DATA_DIR, 'fifa_ranking.csv')
    
    matches = pd.read_csv(matches_path)
    fifa = pd.read_csv(fifa_path)
    
    matches['date'] = pd.to_datetime(matches['date'])
    fifa['rank_date'] = pd.to_datetime(fifa['rank_date'])
    
    # Filter matches after FIFA rankings began (August 1993)
    min_fifa_date = fifa['rank_date'].min()
    matches = matches[matches['date'] >= min_fifa_date].copy()
    
    # Map team names
    matches['home_team'] = matches['home_team'].replace(TEAM_NAME_MAP)
    matches['away_team'] = matches['away_team'].replace(TEAM_NAME_MAP)
    fifa['country_full'] = fifa['country_full'].replace(TEAM_NAME_MAP)
    
    # Target variables
    matches['goal_difference'] = matches['home_score'] - matches['away_score']
    matches['total_goals'] = matches['home_score'] + matches['away_score']
    matches['home_result'] = np.select(
        [matches['goal_difference'] > 0, matches['goal_difference'] == 0, matches['goal_difference'] < 0],
        ['Win', 'Draw', 'Loss'],
        default='Unknown'
    )
    
    # Tournament classification
    matches['is_friendly'] = (matches['tournament'] == 'Friendly').astype(int)
    matches['neutral'] = matches['neutral'].astype(int)
    
    return matches, fifa

def merge_fifa_rankings(matches, fifa):
    print("Performing point-in-time merge for home and away FIFA rankings...")
    
    # Select key columns from FIFA rankings
    fifa_sub = fifa[['rank', 'country_full', 'total_points', 'rank_date']].dropna(subset=['country_full', 'rank']).copy()
    fifa_sub = fifa_sub.rename(columns={'rank_date': 'date'}).sort_values('date')
    
    matches = matches.sort_values('date')
    
    # Merge Home Team Rank
    merged_home = pd.merge_asof(
        matches,
        fifa_sub.rename(columns={
            'country_full': 'home_team',
            'rank': 'home_rank',
            'total_points': 'home_total_points'
        }),
        on='date',
        by='home_team',
        direction='backward'
    )
    
    # Merge Away Team Rank
    merged_all = pd.merge_asof(
        merged_home,
        fifa_sub.rename(columns={
            'country_full': 'away_team',
            'rank': 'away_rank',
            'total_points': 'away_total_points'
        }),
        on='date',
        by='away_team',
        direction='backward'
    )
    
    # Drop rows where FIFA rank was not available (e.g., non-FIFA member matches)
    clean_df = merged_all.dropna(subset=['home_rank', 'away_rank']).copy()
    
    # Feature Engineering
    clean_df['rank_gap'] = clean_df['away_rank'] - clean_df['home_rank'] # Positive = Home team better ranked
    clean_df['point_gap'] = clean_df['home_total_points'].fillna(0) - clean_df['away_total_points'].fillna(0)
    
    print(f"Original matches (post-1993): {len(matches)}")
    print(f"Matches after FIFA rank merge: {len(clean_df)}")
    
    return clean_df

def compute_rolling_form(df, window=5):
    print(f"Computing {window}-game rolling form for teams...")

    # Stack matches into long format to compute historical form chronologically
    home_df = df[['date', 'home_team', 'goal_difference']].rename(columns={'home_team': 'team', 'goal_difference': 'gd'})
    away_df = df[['date', 'away_team', 'goal_difference']].rename(columns={'away_team': 'team', 'goal_difference': 'gd'})
    away_df['gd'] = -away_df['gd'] # Away team's goal difference is inverted
    
    team_games = pd.concat([home_df, away_df], axis=0).sort_values(['team', 'date']).reset_index(drop=True)
    
    # Shift by 1 so form only uses PAST matches (no data leakage)
    team_games['past_gd'] = team_games.groupby('team')['gd'].shift(1)
    team_games['form'] = team_games.groupby('team')['past_gd'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # Merge form back into main dataframe
    home_form = team_games[['date', 'team', 'form']].rename(columns={'team': 'home_team', 'form': 'home_form'})
    away_form = team_games[['date', 'team', 'form']].rename(columns={'team': 'away_team', 'form': 'away_form'})
    
    # Group duplicates if team played twice on same date
    home_form = home_form.groupby(['date', 'home_team']).first().reset_index()
    away_form = away_form.groupby(['date', 'away_team']).first().reset_index()
    
    df = pd.merge(df, home_form, on=['date', 'home_team'], how='left')
    df = pd.merge(df, away_form, on=['date', 'away_team'], how='left')
    
    df['home_form'] = df['home_form'].fillna(0)
    df['away_form'] = df['away_form'].fillna(0)
    df['form_gap'] = df['home_form'] - df['away_form']
    
    return df

def main():
    matches, fifa = load_and_clean_raw_data()
    clean_df = merge_fifa_rankings(matches, fifa)
    final_df = compute_rolling_form(clean_df, window=5)
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'clean_soccer_data.csv')
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved cleaned dataset to {output_path} ({len(final_df)} rows).")
    print("Columns:", final_df.columns.tolist())

if __name__ == "__main__":
    main()
