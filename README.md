# IPL Predictive Analytics Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Dashboard-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Wrangling-150458.svg)

## Overview
This project is an enterprise-grade Machine Learning pipeline and real-time dashboard designed to predict Indian Premier League (IPL) match scores ball-by-ball. 

Unlike basic models that attempt to predict absolute final scores using static franchise names, this engine utilizes **Residual Framing**, **Bayesian Player Shrinkage**, and **Scoreboard Pressure** to generate highly accurate, dynamic projections based on the actual 12 players on the field.

---

##  Key Features & Methodology

### 1. Residual Prediction Framing
Predicting a T20 final score from the first over is highly volatile. Instead of predicting the `final_score` directly, this model predicts `remaining_runs`. By predicting the delta (what happens next), the mathematical bounds of the problem are strictly confined. The final projection is simply: `Current Score + Predicted Remaining Runs`.

### 2. Bayesian Lineup Strength (The "Mega Auction" Fix)
Franchise rosters change drastically. A model relying on "Chennai Super Kings" as a static string fails when the core team changes. This system calculates real-time strength using the specific **Playing 12**.
To handle debutants or players with tiny sample sizes (the "Cold Start" problem), the model uses **Bayesian Shrinkage**. Every player is assigned a "ghost history" of the league average. As they face more deliveries, their true stats overwrite the ghost stats.

### 3. Chasing Dynamics (Scoreboard Pressure)
Batting first and batting second are entirely different sports. The model isolates 2nd innings data and injects `is_chasing`, `runs_required`, and `rrr` (Required Run Rate). This forces the model to understand how scoreboard pressure alters batting aggression and wicket probability.

### 4. Expanding Windows (Preventing Data Leakage)
When calculating historical venue averages or player strengths for Match 45, the model *only* uses data from Matches 1 through 44. It never looks into the future, perfectly simulating a real-time production environment.

---

## Architecture

### Data Ingestion (`data_clean.py`)
Ingests raw ball-by-ball JSON files sourced from Cricsheet. Flattens deeply nested structures into a continuous tabular format, extracting critical micro-events (runs, extras, wicket types, boundary flags) while separating legal deliveries from illegal ones for precise momentum tracking.

### The Machine Learning Engine (`pipeline.py`)
The core predictor is a Gradient Boosting Machine utilizing the `LightGBM` framework.
* Natively handles non-linear interactions (e.g., the shifting value of `wickets_in_hand` based on `balls_remaining`).
* Uses `GroupKFold` cross-validation split by `matchID` to guarantee that deliveries from the same match are never split between training and testing sets.
* Trains 5 separate models to generate an ensemble average and an **80% Confidence Interval** band.

### Real-Time Dashboard (`app.py`)
A Streamlit-powered interactive UI visualizing the model's predictions in real-time.
* **Custom Playing 12 Selector:** Users select exact batters and bowlers (including Impact Subs). The app dynamically recalculates the Lineup Average Strike Rate and Economy.
* **Milestone Blending (Broadcast Style):** Projects the score at 6, 10, 12, and 15 overs by blending a naive Current Run Rate (CRR) extrapolation with a Venue-Adjusted Historical Pace curve at a 50/50 ratio.
* **Live Progression Chart:** A dynamic Plotly visualization plotting the "Par Pace" for the specific stadium against the model's live projected path.

---

## Installation & Usage

### Prerequisites
* Python 3.9+
* Required libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `streamlit`, `plotly`, `joblib`

```bash
pip install pandas numpy scikit-learn lightgbm streamlit plotly joblib
