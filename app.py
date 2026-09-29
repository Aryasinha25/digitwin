import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.thermal_model_v2 import calculate_temperature_decay_with_uncertainty, calculate_viscosity
from core.wave_equation import calculate_float_risk, calculate_production
from core.optimizer import optimize_spm

st.set_page_config(page_title="Adaptive Baghewala Twin", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #FF4B4B; color: white; }
    .metric-box { background-color: #262730; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

if 'current_day' not in st.session_state:
    st.session_state.current_day = 0
if 'drift_detected' not in st.session_state:
    st.session_state.drift_detected = False
if 'calibrated_multiplier' not in st.session_state:
    st.session_state.calibrated_multiplier = 1.0

st.title("🧠 Adaptive Baghewala Twin (BAT)")
st.markdown("### Closed-Loop, Uncertainty-Aware Digital Twin for CSS & SRP")

st.sidebar.header("⏱️ Time Machine Simulator")
st.sidebar.markdown(f"**Current Day:** {st.session_state.current_day}")

if st.sidebar.button("Advance 7 Days (Simulate Field)"):
    st.session_state.current_day += 7

if st.sidebar.button("Inject Geological Disturbance (Drift)"):
    st.session_state.calibrated_multiplier = 3.0 # Forces actual temp to plummet faster than expected
    st.sidebar.warning("Disturbance injected! The well is cooling rapidly.")

if st.sidebar.button("Recalibrate AI Twin"):
    st.session_state.calibrated_multiplier = 1.0
    st.session_state.drift_detected = False
    st.sidebar.success("Model recalibrated successfully.")

if st.sidebar.button("Reset Simulator"):
    st.session_state.current_day = 0
    st.session_state.drift_detected = False
    st.session_state.calibrated_multiplier = 1.0

days = np.arange(0, 180, 1)
T_res = 47.0

# 1. Physics Engine with Uncertainty
pred_temp, actual_temp, temp_uncertainty = calculate_temperature_decay_with_uncertainty(
    days, steam_vol=3000, soak_time=7, error_multiplier=st.session_state.calibrated_multiplier
)

# Detect Drift
error = abs(actual_temp[st.session_state.current_day] - pred_temp[st.session_state.current_day])
if error > temp_uncertainty[st.session_state.current_day]:
    st.session_state.drift_detected = True

if st.session_state.drift_detected:
    st.error(f"🚨 MODEL DRIFT DETECTED AT DAY {st.session_state.current_day}: Actual temperature deviated outside confidence interval. Recalibration required.")

# Viscosities
pred_visc = calculate_viscosity(pred_temp)
actual_visc = calculate_viscosity(actual_temp)

# Optimizer
spm = optimize_spm(pred_visc) 
pred_float = calculate_float_risk(pred_visc, spm)
actual_float = calculate_float_risk(actual_visc, spm) # What happens to the rod in reality

# 2. Visualizations (Uncertainty & Drift)
col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    # Confidence Band
    fig1.add_trace(go.Scatter(x=days, y=pred_temp + temp_uncertainty, line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=days, y=pred_temp - temp_uncertainty, fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', line=dict(width=0), name="Confidence Interval"))
    
    # Predicted vs Actual
    fig1.add_trace(go.Scatter(x=days, y=pred_temp, name="Predicted Temp", line=dict(color='#00BFFF', width=2, dash='dash')))
    
    # Only show actual up to current day
    fig1.add_trace(go.Scatter(x=days[:st.session_state.current_day+1], y=actual_temp[:st.session_state.current_day+1], name="Actual Field Temp", line=dict(color='#FF4B4B', width=3)))
    
    fig1.update_layout(title="M1: Thermal State Estimation & Drift Tracking", xaxis_title="Days", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=days, y=spm, name="AI Recommended SPM", line=dict(color='#00FF00', width=2)))
    fig2.add_trace(go.Scatter(x=days, y=pred_float, name="Predicted Float Risk", line=dict(color='yellow', width=2, dash='dot')))
    
    # Actual float risk based on hidden reality
    fig2.add_trace(go.Scatter(x=days[:st.session_state.current_day+1], y=actual_float[:st.session_state.current_day+1], name="Actual Float Risk", line=dict(color='red', width=3)))
    
    fig2.update_layout(title="M6 & M2: Uncertainty-Aware SPM Control", xaxis_title="Days", yaxis_title="SPM / Risk (%)")
    st.plotly_chart(fig2, use_container_width=True)
