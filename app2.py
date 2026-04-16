import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. SETUP & LOAD ASSETS ---
st.set_page_config(page_title="IPL Live Predictor", layout="wide")

@st.cache_resource
def load_assets():
    return joblib.load('ipl_predictor_assets.pkl')

try:
    assets = load_assets()
    models = assets['models']
    team_str_df = assets['team_strength']
    venue_avg_df = assets['venue_avg']
    curve = assets['expected_curve']
    TRAINING_FEATURE_ORDER = assets['feature_cols']
    batters_df = assets['batters']
    bowlers_df = assets['bowlers']
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

# --- 3. SIDEBAR: PLAYING 12 ---
with st.sidebar.expander("🛠️ Customize Playing 12 (Impact Sub)", expanded=False):
    t_batters = batters_df[batters_df['battingTeam'] == bat_team]['batterName'].tolist()
    sel_batters = st.multiselect(f"{bat_team} Batters", batters_df['batterName'].sort_values(), default=t_batters[:12] if len(t_batters) >= 12 else t_batters)
    
    t_bowlers = bowlers_df[bowlers_df['bowlingTeam'] == bowl_team]['bowlerName'].tolist()
    sel_bowlers = st.multiselect(f"{bowl_team} Bowlers", bowlers_df['bowlerName'].sort_values(), default=t_bowlers[:12] if len(t_bowlers) >= 12 else t_bowlers)

# --- 4. SIDEBAR: MATCH SITUATION ---
st.sidebar.markdown("---")
is_chasing_ui = st.sidebar.radio("Innings Phase", ["1st Innings (Setting Target)", "2nd Innings (Chasing)"])
target_score = st.sidebar.number_input("Target Score to Win", min_value=1, max_value=350, value=180) if "2nd" in is_chasing_ui else 0

st.sidebar.markdown("---")
overs_bowled = st.sidebar.number_input("Overs Bowled (e.g., 10.4)", min_value=0.0, max_value=19.5, value=10.0, step=0.1)
cum_runs = st.sidebar.number_input("Current Score", min_value=0, max_value=350, value=85)
cum_wickets = st.sidebar.slider("Wickets Lost", 0, 10, 2)

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

if len(sel_batters) > 0:
    bat_strength = (batters_df[batters_df['batterName'].isin(sel_batters)].sort_values('smoothed_sr', ascending=False).head(7)['smoothed_sr'].mean() / 100) * 120 
else:
    bat_strength = team_str_df[team_str_df['batting_team'] == bat_team]['batting_strength'].iloc[-1]

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
    'is_chasing': is_chasing, 'runs_required': runs_required, 'rrr': rrr
}

# --- 6. PREDICTION & MATH ---
row = pd.DataFrame([live_features])[TRAINING_FEATURE_ORDER]
if show_debug: st.sidebar.write(row.T)

preds = np.array([m.predict(row)[0] for m in models])
mean_rem = np.mean(preds)
std_rem  = np.std(preds)

predicted_final = int(round(cum_runs + mean_rem))
ci_radius_20 = int(round(1.28 * std_rem)) # 80% CI radius for final score
ci_low  = predicted_final - ci_radius_20
ci_high = predicted_final + ci_radius_20

# MILESTONE CI SCALING (Square Root of Time Rule)
def get_blended_milestone(target_ball):
    if legal_ball_num >= target_ball: return None, None
    
    # 1. Calculate Score Projection
    curve_diff = (curve[target_ball] - curve[legal_ball_num]) * (venue_avg_score / total_curve_runs)
    crr_diff = (current_rr / 6) * (target_ball - legal_ball_num)
    proj = int(round(cum_runs + (0.5 * curve_diff) + (0.5 * crr_diff)))
    
    # 2. Calculate Scaled CI Radius
    balls_to_milestone = target_ball - legal_ball_num
    time_fraction = balls_to_milestone / max(balls_remaining, 1)
    scaled_std = std_rem * np.sqrt(time_fraction)
    ci_radius = int(round(1.28 * scaled_std))
    
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

# --- MILESTONE CARDS WITH CI ---
st.markdown("### 📊 Live Milestones vs Par")
m1, m2, m3, m4, m5 = st.columns(5)

def display_milestone(col, title, proj, ci_rad, exp):
    if proj is None:
        col.metric(title, "Passed")
    else:
        delta = proj - exp
        col.metric(title, f"{proj} (±{ci_rad})", f"{delta:+.0f} vs Par", delta_color="normal")

display_milestone(m1, "6 Overs", p6, ci6, live_features['expected_6ov'])
display_milestone(m2, "10 Overs", p10, ci10, live_features['expected_10ov'])
display_milestone(m3, "12 Overs", p12, ci12, live_features['expected_12ov'])
display_milestone(m4, "15 Overs", p15, ci15, live_features['expected_15ov'])
display_milestone(m5, "20 Overs (Model)", predicted_final, ci_radius_20, venue_avg_score)

st.markdown("---")

# --- CHART RENDERING (WITH CI BAND) ---
st.markdown("### 📈 Innings Progression & Confidence Cone")
fig = go.Figure()

# 1. Par Pace
scaled_curve = [c * (venue_avg_score / total_curve_runs) for c in curve]
fig.add_trace(go.Scatter(x=[i/6 for i in range(120)], y=scaled_curve, mode='lines', name='Par Pace', line=dict(color='gray', dash='dash')))

# 2. Build Path Data
milestone_balls = [35, 59, 71, 89]
milestones_proj = [p6, p10, p12, p15]
milestones_ci = [ci6, ci10, ci12, ci15]

path_balls = [legal_ball_num]
path_scores = [cum_runs]
path_upper = [cum_runs]
path_lower = [cum_runs]

for b, p, ci in zip(milestone_balls, milestones_proj, milestones_ci):
    if p is not None:
        path_balls.append(b)
        path_scores.append(p)
        path_upper.append(p + ci)
        path_lower.append(p - ci)

path_balls.append(119)
path_scores.append(predicted_final)
path_upper.append(ci_high)
path_lower.append(ci_low)

path_overs = [b/6 for b in path_balls]

# 3. Draw Confidence Cone (Shaded Area)
fig.add_trace(go.Scatter(
    x=path_overs + path_overs[::-1], 
    y=path_upper + path_lower[::-1], 
    fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', 
    line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='80% Confidence Band'
))

# 4. Draw Projected Line
fig.add_trace(go.Scatter(x=path_overs, y=path_scores, mode='lines+markers', name='Projected Path', line=dict(color='#38BDF8', width=3)))

# 5. Draw "Now" Marker
fig.add_trace(go.Scatter(x=[legal_ball_num/6], y=[cum_runs], mode='markers+text', name='Now', text=['Now'], textposition='top left', marker=dict(color='#ef4444', size=14, symbol='star')))

if is_chasing:
    fig.add_hline(y=target_score, line_dash="dot", line_color="red", annotation_text="Target", annotation_position="top left")

fig.update_layout(xaxis_title="Overs", yaxis_title="Runs", height=450, hovermode="x unified", xaxis=dict(range=[0, 20], dtick=2), yaxis=dict(range=[0, max(250, target_score + 10 if is_chasing else 0, ci_high + 10)]))
st.plotly_chart(fig, use_container_width=True)
