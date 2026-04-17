"""
IPL Score Prediction Pipeline - ENTERPRISE GRADE
================================================
- Residual framing (Remaining Runs)
- Chasing / Scoreboard Pressure Features
- Bayesian Player Smoothing Priors
- 2026 Roster Default Extraction
- Pace/Spin Matchup Matrix
"""

import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD, CLEAN & CHASING FEATURES
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop super-over innings (inningID > 2) — they corrupt target construction
    # and represent a fundamentally different game state not useful for score prediction
    df = df[df['inningID'].isin([1, 2])].copy()
    df = df.sort_values(['matchID', 'inningID', 'over', 'ball']).reset_index(drop=True)

    df['is_legal'] = (df['isWide'] == 0) & (df['isNoBall'] == 0)
    df['is_wicket_int'] = df['isWicket'].astype(int)
    df['is_boundary_int'] = df['isBoundary'].astype(int)
    df['is_dot'] = ((df['runs'] == 0) & df['is_legal']).astype(int)
    df['is_six'] = ((df['batsmanRuns'] == 6) & df['is_legal']).astype(int)
    df['is_four'] = ((df['batsmanRuns'] == 4) & df['is_legal']).astype(int)

    df['cum_runs'] = df.groupby(['matchID', 'inningID'])['runs'].cumsum()
    df['cum_wickets'] = df.groupby(['matchID', 'inningID'])['is_wicket_int'].cumsum()

    df['legal_ball_num'] = df.groupby(['matchID', 'inningID'])['is_legal'].cumsum() - 1
    df['legal_ball_num'] = df.groupby(['matchID', 'inningID'])['legal_ball_num'].ffill().clip(0, 119).astype(int)
    df['over_int'] = df['over'].astype(int).clip(0, 19)

    final = df.groupby(['matchID', 'inningID'])['cum_runs'].last().rename('final_score').reset_index()
    df = df.merge(final, on=['matchID', 'inningID'])

    df['remaining_runs'] = df['final_score'] - df['cum_runs']
    df['balls_remaining'] = 119 - df['legal_ball_num']

    first_innings = df[df['inningID'] == 1].groupby('matchID')['cum_runs'].last().reset_index()
    first_innings.rename(columns={'cum_runs': 'target_score'}, inplace=True)
    first_innings['target_score'] += 1 

    df = df.merge(first_innings, on='matchID', how='left')
    df['target_score'] = df['target_score'].fillna(0)

    df['is_chasing'] = (df['inningID'] == 2).astype(int)
    df['target_score'] = np.where(df['is_chasing'] == 1, df['target_score'], 0)
    df['runs_required'] = np.where(df['is_chasing'] == 1, df['target_score'] - df['cum_runs'], 0)
    
    safe_balls = df['balls_remaining'].replace(0, 1)
    df['rrr'] = np.where(df['is_chasing'] == 1, (df['runs_required'] / safe_balls) * 6, 0.0)

    return df

# ─────────────────────────────────────────────
# STEP 2: BAYESIAN STATS & RECENT ROSTERS
# ─────────────────────────────────────────────
def compute_player_stats(df: pd.DataFrame):
    global_rpb = df['runs'].sum() / df['is_legal'].sum()
    M_bat = 60 
    ghost_runs = global_rpb * M_bat
    
    batters = df[df['is_legal']].groupby('batterName').agg(
        real_runs=('batsmanRuns', 'sum'),
        real_balls=('is_legal', 'sum')
    ).reset_index()
    batters['smoothed_sr'] = ((batters['real_runs'] + ghost_runs) / (batters['real_balls'] + M_bat)) * 100
    
    bat_team_map = df.groupby('batterName')['battingTeam'].agg(lambda x: x.value_counts().index[0]).reset_index()
    batters = batters.merge(bat_team_map, on='batterName').sort_values('real_balls', ascending=False)

    M_bowl = 120 
    ghost_runs_c = global_rpb * M_bowl

    bowlers = df[df['is_legal']].groupby('bowlerName').agg(
        real_runs_c=('runs', 'sum'),
        real_balls_b=('is_legal', 'sum')
    ).reset_index()
    bowlers['smoothed_econ'] = ((bowlers['real_runs_c'] + ghost_runs_c) / (bowlers['real_balls_b'] + M_bowl)) * 6
    
    bowl_team_map = df.groupby('bowlerName')['bowlingTeam'].agg(lambda x: x.value_counts().index[0]).reset_index()
    bowlers = bowlers.merge(bowl_team_map, on='bowlerName').sort_values('real_balls_b', ascending=False)

    latest_season = df['season'].max()
    recent_df = df[df['season'] == latest_season]

    recent_bat = recent_df.groupby(['battingTeam', 'batterName']).size().reset_index(name='balls')
    recent_rosters_bat = recent_bat.sort_values('balls', ascending=False).groupby('battingTeam')['batterName'].apply(list).to_dict()

    recent_bowl = recent_df.groupby(['bowlingTeam', 'bowlerName']).size().reset_index(name='balls')
    recent_rosters_bowl = recent_bowl.sort_values('balls', ascending=False).groupby('bowlingTeam')['bowlerName'].apply(list).to_dict()

    return batters, bowlers, recent_rosters_bat, recent_rosters_bowl

def compute_matchup_stats(df: pd.DataFrame, mapping_path: str):
    if not os.path.exists(mapping_path):
        print(f"⚠️ {mapping_path} not found. Defaulting all bowlers to 'Pace'.")
        bowler_mapping = pd.DataFrame({'bowlerName': df['bowlerName'].unique(), 'bowler_type': 'Pace'})
    else:
        bowler_mapping = pd.read_csv(mapping_path)
        
    df_temp = df.merge(bowler_mapping, on='bowlerName', how='left')
    df_temp['bowler_type'] = df_temp['bowler_type'].fillna('Pace')
    
    global_p = df_temp[df_temp['bowler_type'] == 'Pace']
    global_s = df_temp[df_temp['bowler_type'] == 'Spin']
    
    global_rpb_pace = global_p['runs'].sum() / max(global_p['is_legal'].sum(), 1)
    global_rpb_spin = global_s['runs'].sum() / max(global_s['is_legal'].sum(), 1)
    
    M_bat = 40 
    ghost_runs_pace = global_rpb_pace * M_bat
    ghost_runs_spin = global_rpb_spin * M_bat
    
    matchups = df_temp[df_temp['is_legal']].groupby(['batterName', 'bowler_type']).agg(
        real_runs=('batsmanRuns', 'sum'),
        real_balls=('is_legal', 'sum')
    ).reset_index()
    
    def smooth_sr(row):
        if row['bowler_type'] == 'Pace':
            return ((row['real_runs'] + ghost_runs_pace) / (row['real_balls'] + M_bat)) * 100
        else:
            return ((row['real_runs'] + ghost_runs_spin) / (row['real_balls'] + M_bat)) * 100
            
    matchups['smoothed_sr'] = matchups.apply(smooth_sr, axis=1)
    
    matchup_matrix = matchups.pivot(index='batterName', columns='bowler_type', values='smoothed_sr').reset_index()
    matchup_matrix = matchup_matrix.rename(columns={'Pace': 'vs_Pace_SR', 'Spin': 'vs_Spin_SR'})
    matchup_matrix['vs_Pace_SR'] = matchup_matrix['vs_Pace_SR'].fillna(global_rpb_pace * 100)
    matchup_matrix['vs_Spin_SR'] = matchup_matrix['vs_Spin_SR'].fillna(global_rpb_spin * 100)
    
    return matchup_matrix, bowler_mapping

def compute_team_strength(df: pd.DataFrame) -> pd.DataFrame:
    match_scores = df.groupby(['matchID', 'inningID', 'battingTeam', 'venue']).agg(total_runs=('runs', 'sum')).reset_index()
    match_scores = match_scores.rename(columns={'battingTeam': 'batting_team'})

    opponent = match_scores.groupby('matchID')['batting_team'].apply(list).reset_index()
    opponent['bowling_team'] = opponent['batting_team'].apply(lambda x: [x[1], x[0]] if len(x) == 2 else [x[0], x[0]])
    opponent = opponent.explode('batting_team').reset_index(drop=True)
    opponent['bowling_team'] = opponent.groupby('matchID')['batting_team'].transform(lambda x: x.iloc[::-1].values)
    
    match_scores = match_scores.merge(opponent[['matchID', 'batting_team', 'bowling_team']], on=['matchID', 'batting_team'], how='left')

    all_matches = sorted(match_scores['matchID'].unique())
    g_mean = match_scores['total_runs'].mean()
    records = []
    
    for mid in all_matches:
        past = match_scores[match_scores['matchID'] < mid]
        bat_str = past.groupby('batting_team')['total_runs'].mean() if len(past) >= 4 else pd.Series(dtype=float)
        bowl_str = past.groupby('bowling_team')['total_runs'].mean() if len(past) >= 4 else pd.Series(dtype=float)

        for _, row in match_scores[match_scores['matchID'] == mid].iterrows():
            records.append({
                'matchID': mid, 'batting_team': row['batting_team'], 'bowling_team': row.get('bowling_team', ''),
                'batting_strength': bat_str.get(row['batting_team'], g_mean),
                'bowling_strength': bowl_str.get(row.get('bowling_team', ''), g_mean),
                'venue': row['venue'],
            })
    return pd.DataFrame(records)

def compute_venue_avg(df: pd.DataFrame) -> pd.DataFrame:
    match_venue = df.groupby(['matchID', 'inningID', 'venue']).agg(total_runs=('runs', 'sum')).reset_index()
    all_matches = sorted(match_venue['matchID'].unique())
    g_mean = match_venue['total_runs'].mean()
    records = []
    
    for mid in all_matches:
        past = match_venue[match_venue['matchID'] < mid]
        v_avg = past.groupby('venue')['total_runs'].mean() if len(past) >= 4 else pd.Series(dtype=float)
        records.append({'matchID': mid, 'venue_avg_score': v_avg.get(match_venue[match_venue['matchID'] == mid]['venue'].iloc[0], g_mean)})
    return pd.DataFrame(records)

def build_expected_curve(df: pd.DataFrame) -> np.ndarray:
    return df[df['is_legal']].groupby('legal_ball_num')['cum_runs'].mean().reindex(range(120)).interpolate().values

def build_full_dataset(raw_df, team_str_df, venue_avg_df, curve, matchup_matrix, bowler_mapping):
    df = raw_df.copy()
    df = df.merge(team_str_df[['matchID', 'batting_team', 'batting_strength', 'bowling_strength']], left_on=['matchID', 'battingTeam'], right_on=['matchID', 'batting_team'], how='left')
    df = df.merge(venue_avg_df, on='matchID', how='left')

    df['batting_strength'] = df['batting_strength'].fillna(df['batting_strength'].median())
    df['bowling_strength'] = df['bowling_strength'].fillna(df['bowling_strength'].median())
    df['venue_avg_score'] = df['venue_avg_score'].fillna(df['venue_avg_score'].median())

    df['expected_score'] = df['legal_ball_num'].map(lambda b: curve[int(b)] if int(b) < len(curve) else curve[-1])
    df['delta_vs_expected'] = df['cum_runs'] - df['expected_score']

    grp = df.sort_values(['matchID', 'inningID', 'legal_ball_num']).groupby(['matchID', 'inningID'])
    df['runs_last6'] = grp['runs'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    df['runs_last12'] = grp['runs'].transform(lambda x: x.rolling(12, min_periods=1).sum())
    df['wkts_last12'] = grp['is_wicket_int'].transform(lambda x: x.rolling(12, min_periods=1).sum())
    df['dots_last6'] = grp['is_dot'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    df['boundaries_last6'] = grp['is_boundary_int'].transform(lambda x: x.rolling(6, min_periods=1).sum())
    df['sixes_last6'] = grp['is_six'].transform(lambda x: x.rolling(6, min_periods=1).sum())

    df['wickets_per10'] = (df['cum_wickets'] / (df['legal_ball_num'] + 1)) * 10
    df['current_rr'] = (df['cum_runs'] / (df['legal_ball_num'] + 1)) * 6

    df['over_int'] = df['legal_ball_num'] // 6
    df['is_powerplay'] = (df['over_int'] < 6).astype(int)
    df['is_middle'] = ((df['over_int'] >= 6) & (df['over_int'] < 15)).astype(int)
    df['is_death'] = (df['over_int'] >= 15).astype(int)
    df['balls_in_phase'] = np.where(df['is_powerplay'] == 1, df['legal_ball_num'], np.where(df['is_death'] == 1, df['legal_ball_num'] - 90, df['legal_ball_num'] - 36)).clip(0)
    df['wickets_in_hand'] = 10 - df['cum_wickets']
    df['rr_acceleration'] = df['runs_last12'] / (df['current_rr'].clip(lower=0.01) * 2)

    total_curve_runs = curve[-1]
    df['expected_6ov'] = df['venue_avg_score'] * (curve[35] / total_curve_runs)
    df['expected_10ov'] = df['venue_avg_score'] * (curve[59] / total_curve_runs)
    df['expected_12ov'] = df['venue_avg_score'] * (curve[71] / total_curve_runs)
    df['expected_15ov'] = df['venue_avg_score'] * (curve[89] / total_curve_runs)
    df['expected_20ov'] = df['venue_avg_score']

    # New Matchup Features
    df = df.merge(matchup_matrix, left_on='batterName', right_on='batterName', how='left')
    df = df.merge(bowler_mapping, on='bowlerName', how='left')
    df['bowler_type'] = df['bowler_type'].fillna('Pace')

    # Calculate live matchup edge
    # bowling_strength is mean runs conceded per innings (~180), convert to equivalent batter SR:
    # (runs / 120 balls) * 100 = SR conceded; then edge = batter SR - SR conceded
    bowl_str_as_sr = (df['bowling_strength'] / 120) * 100
    df['live_matchup_edge'] = np.where(
        df['bowler_type'] == 'Pace',
        df['vs_Pace_SR'].fillna(130) - bowl_str_as_sr,
        df['vs_Spin_SR'].fillna(130) - bowl_str_as_sr
    )

    return df

FEATURE_COLS = [
    'batting_strength', 'bowling_strength', 'venue_avg_score',
    'cum_runs', 'cum_wickets', 'balls_remaining', 'current_rr',
    'wickets_in_hand', 'expected_score', 'delta_vs_expected', 
    'runs_last6', 'runs_last12', 'wkts_last12',
    'dots_last6', 'boundaries_last6', 'sixes_last6',
    'legal_ball_num', 'is_powerplay', 'is_middle', 'is_death',
    'balls_in_phase', 'wickets_per10', 'rr_acceleration',
    'expected_6ov', 'expected_10ov', 'expected_12ov', 'expected_15ov', 'expected_20ov',
    'is_chasing', 'runs_required', 'rrr', 'live_matchup_edge'
]

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'mae', 'n_estimators': 600,
    'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 5,
    'min_child_samples': 40, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'verbose': -1, 'random_state': 42,
}

def train_model(dataset: pd.DataFrame):
    train_df = dataset[dataset['is_legal']].copy().reset_index(drop=True)
    # Sort by matchID to enforce temporal ordering before splitting
    train_df = train_df.sort_values('matchID').reset_index(drop=True)
    X, y, groups = train_df[FEATURE_COLS], train_df['remaining_runs'], train_df['matchID']

    # Use GroupKFold with sorted data so folds are temporally ordered
    # (earlier matches train, later matches validate — no future leakage)
    gkf = GroupKFold(n_splits=5)
    oof_remaining = np.zeros(len(train_df))
    models = []

    print(f"Training on {len(train_df)} balls with {len(FEATURE_COLS)} features...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_remaining[val_idx] = model.predict(X.iloc[val_idx])
        models.append(model)

    print(f"OOF Final Score MAE: {mean_absolute_error(train_df['final_score'], train_df['cum_runs'] + oof_remaining):.2f}")
    return models

if __name__ == '__main__':
    PATH = "IPL_2024_2026_Combined.csv"
    raw = load_and_clean(PATH)
    
    batters, bowlers, recent_bat, recent_bowl = compute_player_stats(raw)
    matchup_matrix, bowler_map = compute_matchup_stats(raw, 'bowler_types.csv')
    
    team_str = compute_team_strength(raw)
    venue_avg = compute_venue_avg(raw)
    curve = build_expected_curve(raw)
    dataset = build_full_dataset(raw, team_str, venue_avg, curve, matchup_matrix, bowler_map)
    models = train_model(dataset)
    
    joblib.dump({
        'models': models, 'team_strength': team_str, 'venue_avg': venue_avg,
        'expected_curve': curve, 'feature_cols': FEATURE_COLS,
        'batters': batters, 'bowlers': bowlers,
        'recent_rosters_bat': recent_bat,      
        'recent_rosters_bowl': recent_bowl,    
        'matchup_matrix': matchup_matrix,
        'bowler_mapping': bowler_map          # needed for live_matchup_edge at inference
    }, 'ipl_predictor_assetsv3.pkl')
    print("✓ Pipeline complete. Saved to ipl_predictor_assetsv3.pkl")
