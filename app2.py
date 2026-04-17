import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. SETUP & LOAD ASSETS ---
st.set_page_config(page_title="IPL Live Predictor", layout="wide")

@st.cache_resource
def load_assets():
    return joblib.load('ipl_predictor_assetsv3.pkl')

try:
    assets = load_assets()
    models = assets['models']
    team_str_df = assets['team_strength']
    venue_avg_df = assets['venue_avg']
    curve = assets['expected_curve']
    TRAINING_FEATURE_ORDER = assets['feature_cols']
    batters_df = assets['batters']
    bowlers_df = assets['bowlers']
    
    recent_rosters_bat = assets.get('recent_rosters_bat', {})
    recent_rosters_bowl = assets.get('recent_rosters_bowl', {})
    matchup_matrix = assets.get('matchup_matrix', None)
    bowler_mapping = assets.get('bowler_mapping', None)  # bowler type lookup for matchup edge
    
except Exception as e:
    st.error(f"⚠️ Assets issue: {e}. Please run pipeline.py again.")
    st.stop()

venue_mapping = team_str_df[['matchID', 'venue']].drop_duplicates().merge(venue_avg_df, on='matchID')
latest_venue_priors = venue_mapping.groupby('venue')['venue_avg_score'].mean().reset_index()

# --- 2. SIDEBAR: MATCH SETUP ---
st.sidebar.title("🏏 Live Match State")
teams = sorted(team_str_df['batting_team'].unique())
venues = sorted(latest_venue_priors['venue'].unique())

bat_team = st.sidebar.selectbox("Batting Team", teams, index=teams.index('Royal Challengers Bengaluru') if 'Royal Challengers Bengaluru' in teams else 0)
bowl_team = st.sidebar.selectbox("Bowling Team", teams, index=teams.index('Chennai Super Kings') if 'Chennai Super Kings' in teams else 1)

if bat_team == bowl_team: st.sidebar.warning("⚠️ Batting and Bowling teams are the same!")
venue = st.sidebar.selectbox("Venue", venues)

# --- 3. SIDEBAR: PLAYING 12 & STRIKER LOGIC ---
with st.sidebar.expander("🛠️ Customize Playing 12 (Impact Sub)", expanded=False):
    t_batters = recent_rosters_bat.get(bat_team, [])
    if not t_batters: t_batters = batters_df[batters_df['battingTeam'] == bat_team]['batterName'].tolist()
    sel_batters = st.multiselect(f"{bat_team} Batters", batters_df['batterName'].sort_values(), default=t_batters[:12] if len(t_batters) >= 12 else t_batters)
    
    t_bowlers = recent_rosters_bowl.get(bowl_team, [])
    if not t_bowlers: t_bowlers = bowlers_df[bowlers_df['bowlingTeam'] == bowl_team]['bowlerName'].tolist()
    sel_bowlers = st.multiselect(f"{bowl_team} Bowlers", bowlers_df['bowlerName'].sort_values(), default=t_bowlers[:12] if len(t_bowlers) >= 12 else t_bowlers)

st.sidebar.markdown("---")
st.sidebar.subheader("🏏 At The Crease")
safe_batters = sel_batters if len(sel_batters) >= 2 else batters_df['batterName'].unique()[:2]
striker = st.sidebar.selectbox("On Strike", safe_batters, index=0)
non_striker = st.sidebar.selectbox("Non-Striker", safe_batters, index=1)
if striker == non_striker: st.sidebar.warning("Striker and Non-Striker cannot be the same person!")

# --- 4. SIDEBAR: MATCH SITUATION & MATCHUPS ---
st.sidebar.markdown("---")
is_chasing_ui = st.sidebar.radio("Innings Phase", ["1st Innings (Setting Target)", "2nd Innings (Chasing)"])
target_score = st.sidebar.number_input("Target Score to Win", min_value=1, max_value=350, value=180) if "2nd" in is_chasing_ui else 0

st.sidebar.markdown("---")
overs_bowled = st.sidebar.number_input("Overs Bowled (e.g., 10.4)", min_value=0.0, max_value=19.5, value=10.0, step=0.1,
                                        help="Use cricket notation: .1 to .5 after decimal (e.g. 10.5 = end of 10th over)")
cum_runs = st.sidebar.number_input("Current Score", min_value=0, max_value=350, value=85)
cum_wickets = st.sidebar.slider("Wickets Lost", 0, 10, 2)

# Validate overs notation
over_decimal = round(overs_bowled - int(overs_bowled), 1)
if over_decimal > 0.5:
    st.sidebar.warning(f"⚠️ Invalid over notation ({overs_bowled}): decimal part should be .1–.5 (balls 1–5 in over). Did you mean {int(overs_bowled)}.{min(int(over_decimal*10),5)}?")

st.sidebar.markdown("---")
st.sidebar.subheader("Opponent Bowling Attack")
c_pace, c_spin = st.sidebar.columns(2)
overs_pace_left = c_pace.number_input("Pace Overs Left", 0, 20, 5)
overs_spin_left = c_spin.number_input("Spin Overs Left", 0, 20, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Momentum (Last 2 Overs)")
runs_last12 = st.sidebar.slider("Runs in last 12 balls", 0, 36, 15)
wkts_last12 = st.sidebar.slider("Wickets in last 12 balls", 0, 4, 0)
dots_last12 = st.sidebar.slider("Dot balls in last 12", 0, 12, 4) 
sixes_last12 = st.sidebar.slider("Sixes in last 12 balls", 0, 6, 0) 
show_debug = st.sidebar.checkbox("Show Model Input Debugger")

# --- 5. FEATURE ENGINEERING ---
over_int = int(overs_bowled)
ball_in_over = min(int(round((overs_bowled - over_int) * 10)), 5)
legal_ball_num = max(1, min((over_int * 6) + ball_in_over, 119))
balls_remaining = 120 - legal_ball_num

# ⚡ WEIGHTED MATCHUP-ADJUSTED BATTING STRENGTH ⚡
if len(sel_batters) > 0:
    if matchup_matrix is not None:
        lineup_m = matchup_matrix[matchup_matrix['batterName'].isin(sel_batters)]
        
        def get_player_sr(player_name, p_ratio, s_ratio):
            p_data = lineup_m[lineup_m['batterName'] == player_name]
            if len(p_data) == 0: return 130.0
            return (p_data['vs_Pace_SR'].values[0] * p_ratio) + (p_data['vs_Spin_SR'].values[0] * s_ratio)

        total_overs_left = max(overs_pace_left + overs_spin_left, 1)
        pace_ratio = overs_pace_left / total_overs_left
        spin_ratio = overs_spin_left / total_overs_left

        striker_sr = get_player_sr(striker, pace_ratio, spin_ratio)
        non_striker_sr = get_player_sr(non_striker, pace_ratio, spin_ratio)
        
        dugout = lineup_m[~lineup_m['batterName'].isin([striker, non_striker])]
        if len(dugout) > 0:
            top5_dugout_pace = dugout.sort_values('vs_Pace_SR', ascending=False).head(5)['vs_Pace_SR'].mean()
            top5_dugout_spin = dugout.sort_values('vs_Spin_SR', ascending=False).head(5)['vs_Spin_SR'].mean()
            dugout_sr = (top5_dugout_pace * pace_ratio) + (top5_dugout_spin * spin_ratio)
        else:
            dugout_sr = 130.0

        if balls_remaining <= 18:
            w_striker, w_non_striker, w_dugout = 0.60, 0.35, 0.05
        elif balls_remaining <= 60:
            w_striker, w_non_striker, w_dugout = 0.45, 0.35, 0.20
        else:
            w_striker, w_non_striker, w_dugout = 0.35, 0.30, 0.35

        adjusted_sr = (striker_sr * w_striker) + (non_striker_sr * w_non_striker) + (dugout_sr * w_dugout)
        bat_strength = (adjusted_sr / 100) * 120 

        # Estimate live matchup edge using correct scale
        # bowling_strength is mean innings runs conceded; convert to SR equivalent
        bowl_str_as_sr = (team_str_df[team_str_df['bowling_team'] == bowl_team]['bowling_strength'].iloc[-1] / 120) * 100
        if matchup_matrix is not None:
            striker_row = matchup_matrix[matchup_matrix['batterName'] == striker]
            if len(striker_row) > 0:
                # Use pace/spin weighted SR for the striker
                striker_type_sr = (striker_row['vs_Pace_SR'].values[0] * pace_ratio +
                                   striker_row['vs_Spin_SR'].values[0] * spin_ratio)
                live_edge = striker_type_sr - bowl_str_as_sr
            else:
                live_edge = 130.0 - bowl_str_as_sr
        else:
            live_edge = 0.0
    else:
        bat_strength = (batters_df[batters_df['batterName'].isin(sel_batters)].sort_values('smoothed_sr', ascending=False).head(7)['smoothed_sr'].mean() / 100) * 120 
        live_edge = 0.0
else:
    bat_strength = team_str_df[team_str_df['batting_team'] == bat_team]['batting_strength'].iloc[-1]
    live_edge = 0.0

if len(sel_bowlers) > 0:
    bowl_strength = bowlers_df[bowlers_df['bowlerName'].isin(sel_bowlers)].sort_values('smoothed_econ', ascending=True).head(6)['smoothed_econ'].mean() * 20 
else:
    bowl_strength = team_str_df[team_str_df['bowling_team'] == bowl_team]['bowling_strength'].iloc[-1]

venue_priors = latest_venue_priors[latest_venue_priors['venue'] == venue]
venue_avg_score = venue_priors['venue_avg_score'].values[0] if len(venue_priors) > 0 else 160.0
current_rr = (cum_runs / legal_ball_num) * 6 if legal_ball_num > 0 else 0.0
expected_score = curve[legal_ball_num] if legal_ball_num < len(curve) else curve[-1]
total_curve_runs = curve[-1]

is_chasing = 1 if "2nd" in is_chasing_ui else 0
runs_required = max(0, target_score - cum_runs) if is_chasing else 0
rrr = (runs_required / max(balls_remaining, 1)) * 6 if is_chasing else 0.0

live_features = {
    'batting_strength': bat_strength, 'bowling_strength': bowl_strength, 'venue_avg_score': venue_avg_score,
    'cum_runs': cum_runs, 'cum_wickets': cum_wickets, 'balls_remaining': balls_remaining, 'current_rr': current_rr,
    'wickets_in_hand': 10 - cum_wickets, 'expected_score': expected_score, 'delta_vs_expected': cum_runs - expected_score,
    'runs_last6': runs_last12 / 2, 'runs_last12': runs_last12, 'wkts_last12': wkts_last12,
    'dots_last6': dots_last12 // 2, 'boundaries_last6': (runs_last12 / 2) // 4, 'sixes_last6': sixes_last12 // 2,
    'legal_ball_num': legal_ball_num, 'is_powerplay': 1 if over_int < 6 else 0,
    'is_middle': 1 if 6 <= over_int < 15 else 0, 'is_death': 1 if over_int >= 15 else 0,
    'balls_in_phase': legal_ball_num if over_int < 6 else (legal_ball_num - 36 if over_int < 15 else legal_ball_num - 90),
    'wickets_per10': (cum_wickets / legal_ball_num) * 10 if legal_ball_num > 0 else 0,
    'rr_acceleration': runs_last12 / (max(current_rr, 0.01) * 2) if current_rr > 0 else 0,
    'expected_6ov': venue_avg_score * (curve[35] / total_curve_runs),
    'expected_10ov': venue_avg_score * (curve[59] / total_curve_runs),
    'expected_12ov': venue_avg_score * (curve[71] / total_curve_runs),
    'expected_15ov': venue_avg_score * (curve[89] / total_curve_runs),
    'expected_20ov': venue_avg_score,
    'is_chasing': is_chasing, 'runs_required': runs_required, 'rrr': rrr, 'live_matchup_edge': live_edge
}

# --- 6. PREDICTION & MATH ---
row = pd.DataFrame([live_features])[TRAINING_FEATURE_ORDER]
if show_debug: st.sidebar.write(row.T)

preds = np.array([m.predict(row)[0] for m in models])
mean_rem = np.mean(preds)
std_rem  = np.std(preds)

predicted_final = int(round(cum_runs + mean_rem))
ci_radius_20 = int(round(1.28 * std_rem))
ci_low  = predicted_final - ci_radius_20
ci_high = predicted_final + ci_radius_20

def get_blended_milestone(target_ball):
    if legal_ball_num >= target_ball: return None, None
    curve_diff = (curve[target_ball] - curve[legal_ball_num]) * (venue_avg_score / total_curve_runs)
    crr_diff = (current_rr / 6) * (target_ball - legal_ball_num)
    proj = int(round(cum_runs + (0.5 * curve_diff) + (0.5 * crr_diff)))
    time_fraction = (target_ball - legal_ball_num) / max(balls_remaining, 1)
    ci_radius = int(round(1.28 * (std_rem * np.sqrt(time_fraction))))
    return proj, ci_radius

p6, ci6 = get_blended_milestone(35)
p10, ci10 = get_blended_milestone(59)
p12, ci12 = get_blended_milestone(71)
p15, ci15 = get_blended_milestone(89)

# --- 7. UI RENDER ---
st.title("⚡ IPL Live Score Predictor")
st.write(f"**{bat_team}** vs **{bowl_team}** | 🏟️ {venue}")

if is_chasing: st.error(f"🎯 **TARGET: {target_score}** | Need **{runs_required}** from **{balls_remaining}** (RRR: {rrr:.2f})")

c1, c2, c3 = st.columns(3)
c1.metric("Current Score", f"{int(cum_runs)}/{int(cum_wickets)}", f"Overs: {overs_bowled}")
c2.metric("Current Run Rate", f"{current_rr:.2f}")

if is_chasing:
    diff = predicted_final - target_score
    c3.metric("Projected Total", f"{predicted_final} (±{ci_radius_20})", f"{abs(diff)} runs {'short' if diff < 0 else 'ahead'}", delta_color="normal" if diff >= 0 else "inverse")
else:
    c3.metric("Projected Total", f"{predicted_final}", f"± {ci_radius_20} runs", delta_color="off")

st.markdown("### 📊 Live Milestones vs Par")
m1, m2, m3, m4, m5 = st.columns(5)

def display_milestone(col, title, proj, ci_rad, exp):
    if proj is None: col.metric(title, "Passed")
    else: col.metric(title, f"{proj} (±{ci_rad})", f"{proj - exp:+.0f} vs Par", delta_color="normal")

display_milestone(m1, "6 Overs", p6, ci6, live_features['expected_6ov'])
display_milestone(m2, "10 Overs", p10, ci10, live_features['expected_10ov'])
display_milestone(m3, "12 Overs", p12, ci12, live_features['expected_12ov'])
display_milestone(m4, "15 Overs", p15, ci15, live_features['expected_15ov'])
display_milestone(m5, "20 Overs (Model)", predicted_final, ci_radius_20, venue_avg_score)

st.markdown("---")
st.markdown("### 📈 Innings Progression & Confidence Cone")
fig = go.Figure()

scaled_curve = [c * (venue_avg_score / total_curve_runs) for c in curve]
fig.add_trace(go.Scatter(x=[i/6 for i in range(120)], y=scaled_curve, mode='lines', name='Par Pace', line=dict(color='gray', dash='dash')))

milestone_balls = [35, 59, 71, 89]
milestones_proj = [p6, p10, p12, p15]
milestones_ci = [ci6, ci10, ci12, ci15]

path_balls, path_scores, path_upper, path_lower = [legal_ball_num], [cum_runs], [cum_runs], [cum_runs]
for b, p, ci in zip(milestone_balls, milestones_proj, milestones_ci):
    if p is not None:
        path_balls.append(b); path_scores.append(p); path_upper.append(p + ci); path_lower.append(p - ci)

path_balls.append(119); path_scores.append(predicted_final); path_upper.append(ci_high); path_lower.append(ci_low)
path_overs = [b/6 for b in path_balls]

fig.add_trace(go.Scatter(x=path_overs + path_overs[::-1], y=path_upper + path_lower[::-1], fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='80% Confidence Band'))
fig.add_trace(go.Scatter(x=path_overs, y=path_scores, mode='lines+markers', name='Projected Path', line=dict(color='#38BDF8', width=3)))
fig.add_trace(go.Scatter(x=[legal_ball_num/6], y=[cum_runs], mode='markers+text', name='Now', text=['Now'], textposition='top left', marker=dict(color='#ef4444', size=14, symbol='star')))
if is_chasing: fig.add_hline(y=target_score, line_dash="dot", line_color="red", annotation_text="Target", annotation_position="top left")

fig.update_layout(xaxis_title="Overs", yaxis_title="Runs", height=450, hovermode="x unified", xaxis=dict(range=[0, 20], dtick=2), yaxis=dict(range=[0, max(250, target_score + 10 if is_chasing else 0, ci_high + 10)]))
st.plotly_chart(fig, use_container_width=True)

# --- PLAYER SCOUTING EXPANDER ---
with st.expander("📊 View Selected Playing 12 Scouting Report"):
    if len(sel_batters) > 0 or len(sel_bowlers) > 0:
        scout_c1, scout_c2 = st.columns(2)
        with scout_c1:
            st.markdown(f"**{bat_team} Selected Batters**")
            lineup_b = batters_df[batters_df['batterName'].isin(sel_batters)].copy()
            lineup_b = lineup_b[['batterName', 'real_runs', 'real_balls', 'smoothed_sr']]
            lineup_b.columns = ['Player', 'Career Runs', 'Balls Faced', 'Bayesian SR']
            lineup_b['Bayesian SR'] = lineup_b['Bayesian SR'].round(1)
            st.dataframe(lineup_b.sort_values('Bayesian SR', ascending=False), hide_index=True)
            
        with scout_c2:
            st.markdown(f"**{bowl_team} Selected Bowlers**")
            lineup_bw = bowlers_df[bowlers_df['bowlerName'].isin(sel_bowlers)].copy()
            lineup_bw = lineup_bw[['bowlerName', 'real_runs_c', 'real_balls_b', 'smoothed_econ']]
            lineup_bw.columns = ['Player', 'Runs Conceded', 'Balls Bowled', 'Bayesian Econ']
            lineup_bw['Bayesian Econ'] = lineup_bw['Bayesian Econ'].round(2)
            st.dataframe(lineup_bw.sort_values('Bayesian Econ', ascending=True), hide_index=True)
    else:
        st.info("Select players in the sidebar to view their stats!")
