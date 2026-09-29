import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

from core.thermal_model_v2 import calculate_temperature_decay_with_uncertainty, calculate_viscosity
from core.wave_equation import calculate_float_risk, calculate_production
from core.optimizer import optimize_spm

st.set_page_config(page_title="Adaptive Baghewala Twin", layout="wide")

# --- Try to load the trained ML Models ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'core', 'models')
risk_model_path = os.path.join(MODEL_DIR, 'rf_risk_model.pkl')
failure_model_path = os.path.join(MODEL_DIR, 'rf_failure_model.pkl')

try:
    rf_risk_model = joblib.load(risk_model_path)
    rf_failure_model = joblib.load(failure_model_path)
    using_ml = True
except FileNotFoundError:
    using_ml = False

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #FF4B4B; color: white; }
</style>
""", unsafe_allow_html=True)

if 'current_day' not in st.session_state:
    st.session_state.current_day = 0
if 'drift_detected' not in st.session_state:
    st.session_state.drift_detected = False
if 'calibrated_multiplier' not in st.session_state:
    st.session_state.calibrated_multiplier = 1.0

st.title("🧠 Adaptive Baghewala Twin (BAT)")
if using_ml:
    st.success("✅ Live ML Inference Active: Powered by Scikit-Learn Random Forest.")
else:
    st.warning("⚠️ Physics Surrogate Mode: Run `core/ml_trainer.py` to train the Random Forest and activate Live ML Inference.")

st.sidebar.header("⏱️ Time Machine Simulator")
st.sidebar.markdown(f"**Current Day:** {st.session_state.current_day}")

if st.sidebar.button("Advance 7 Days (Simulate Field)"):
    st.session_state.current_day += 7
if st.sidebar.button("Inject Geological Disturbance"):
    st.session_state.calibrated_multiplier = 3.0
if st.sidebar.button("Recalibrate AI Twin"):
    st.session_state.calibrated_multiplier = 1.0
    st.session_state.drift_detected = False
if st.sidebar.button("Reset Simulator"):
    st.session_state.current_day = 0
    st.session_state.drift_detected = False
    st.session_state.calibrated_multiplier = 1.0

days = np.arange(0, 180, 1)
T_res = 47.0

# 1. Base Physics (Always needed for thermal decay)
pred_temp, actual_temp, temp_uncertainty = calculate_temperature_decay_with_uncertainty(
    days, steam_vol=3000, soak_time=7, error_multiplier=st.session_state.calibrated_multiplier
)

# Drift Logic
error = abs(actual_temp[st.session_state.current_day] - pred_temp[st.session_state.current_day])
if error > temp_uncertainty[st.session_state.current_day]:
    st.session_state.drift_detected = True

if st.session_state.drift_detected:
    st.error(f"🚨 MODEL DRIFT DETECTED AT DAY {st.session_state.current_day}: Actual temperature deviated outside confidence interval. Recalibration required.")

pred_visc = calculate_viscosity(pred_temp)
spm = optimize_spm(pred_visc) 
stroke = np.full_like(days, 3.0)

# 2. Inference: ML vs Physics
if using_ml:
    # Build feature DataFrame for the ML model
    X_inference = pd.DataFrame({
        'Temperature_C': pred_temp,
        'Viscosity_cP': pred_visc,
        'SPM': spm,
        'Stroke_Length_m': stroke
    })
    # Actual ML Inference
    pred_float = rf_risk_model.predict(X_inference)
    failure_risk = rf_failure_model.predict_proba(X_inference)[:, 1] * 100 # Probability of failure
else:
    # Fallback to physics math
    pred_float = calculate_float_risk(pred_visc, spm)
    failure_risk = np.where(pred_float > 80, 80, pred_float / 2)

# Visualization
col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=days, y=pred_temp + temp_uncertainty, line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=days, y=pred_temp - temp_uncertainty, fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', line=dict(width=0), name="Confidence Interval"))
    fig1.add_trace(go.Scatter(x=days, y=pred_temp, name="Predicted Temp", line=dict(color='#00BFFF', width=2, dash='dash')))
    fig1.add_trace(go.Scatter(x=days[:st.session_state.current_day+1], y=actual_temp[:st.session_state.current_day+1], name="Actual Field Temp", line=dict(color='#FF4B4B', width=3)))
    fig1.update_layout(title="M1: Thermal State Estimation & Drift Tracking", xaxis_title="Days", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=days, y=spm, name="AI Recommended SPM", line=dict(color='#00FF00', width=2)))
    fig2.add_trace(go.Scatter(x=days, y=pred_float, name="Float Risk Prediction", line=dict(color='yellow', width=2, dash='dot')))
    fig2.add_trace(go.Scatter(x=days, y=failure_risk, fill='tozeroy', name="Critical Failure Risk (%)", line=dict(color='red', width=2), fillcolor='rgba(255, 0, 0, 0.2)'))
    fig2.update_layout(title="M6 & ML Inference: Pump Control & Risk Profile", xaxis_title="Days", yaxis_title="Risk (%) / SPM")
    st.plotly_chart(fig2, use_container_width=True)
