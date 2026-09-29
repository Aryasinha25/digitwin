import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.thermal_model_v2 import calculate_temperature_decay_with_uncertainty, calculate_viscosity
from core.wave_equation import calculate_float_risk, calculate_production
from core.optimizer import optimize_spm
from core.model_loader import load_models

st.set_page_config(page_title="Adaptive Baghewala Twin", layout="wide")

# --- Formally Load Models ---
model_state = load_models()
using_ml = (model_state["mode"] == "ml")

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stButton>button { width: 100%; font-weight: bold; background-color: #FF4B4B; color: white; }
    .health-box { background-color: #1e1e1e; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; }
    .status-ready { color: #00FF00; }
    .status-failed { color: #FF0000; }
    .status-degraded { color: #FFA500; }
</style>
""", unsafe_allow_html=True)

if 'current_day' not in st.session_state:
    st.session_state.current_day = 0
if 'drift_detected' not in st.session_state:
    st.session_state.drift_detected = False
if 'calibrated_multiplier' not in st.session_state:
    st.session_state.calibrated_multiplier = 1.0

st.title("Adaptive Baghewala Twin (BAT)")

if using_ml:
    st.success("MODEL MODE: ML Inference Active")
else:
    st.warning(f"MODEL MODE: Physics Simulation Fallback (ML inference unavailable; physics simulation is being used. Reason: {model_state['reason']})")

st.sidebar.header("Time Machine Simulator")
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

# --- MODEL HEALTH DASHBOARD ---
st.sidebar.markdown("---")
st.sidebar.markdown("### MODEL HEALTH")
health_html = '<div class="health-box">'
for k, v in model_state['health'].items():
    color_class = "status-ready" if v == "READY" else "status-failed" if v == "FAILED" else "status-degraded"
    health_html += f"<div>{k.replace('_', ' ').title()}: <span class='{color_class}'>{v}</span></div>"
health_html += "</div>"
st.sidebar.markdown(health_html, unsafe_allow_html=True)

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
    X_inference = pd.DataFrame({'Temperature_C': pred_temp, 'Viscosity_cP': pred_visc, 'SPM': spm, 'Stroke_Length_m': stroke})
    rf_risk_model = model_state["models"]["risk_model"]
    rf_failure_model = model_state["models"]["failure_model"]
    
    pred_float = rf_risk_model.predict(X_inference)
    failure_risk = rf_failure_model.predict_proba(X_inference)[:, 1] * 100 
else:
    # Fallback to physics math
    pred_float = calculate_float_risk(pred_visc, spm)
    failure_risk = np.where(pred_float > 80, 80, pred_float / 2)

tab1, tab2, tab3, tab4 = st.tabs(["WELL ANALYTICS", "FIELD COMMAND", "ECONOMIC OPTIMIZER", "LIVE SCADA STREAM"])

with tab1:
    st.subheader("Single Well AI Diagnostics")
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

with tab2:
    st.subheader("Baghewala Field Command Center")
    st.markdown("Monitor all 42 heavy oil wells in the sector simultaneously. The AI highlights high-risk wells requiring immediate maintenance.")
    
    # Dummy grid of wells
    field_cols = st.columns(4)
    wells = [("BGH-01", "Healthy", "green"), ("BGH-02", "Critical Risk", "red"), ("BGH-03", "Warning", "orange"), ("BGH-04", "Healthy", "green"),
             ("BGH-05", "Healthy", "green"), ("BGH-06", "Offline", "gray"), ("BGH-07", "Critical Risk", "red"), ("BGH-08", "Warning", "orange")]
    
    for i, (well, status, color) in enumerate(wells):
        with field_cols[i % 4]:
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px; border: 2px solid {color};">
                <h3 style="margin:0;">{well}</h3>
                <p style="color:{color}; font-weight:bold; margin:0;">{status}</p>
                <small>SPM: {np.random.randint(4, 9)} | Temp: {np.random.randint(60, 180)}°C</small>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.subheader("Net Present Value (NPV) & Economic Optimizer")
    
    # Dummy economic calculation based on our simulation
    oil_price = 70.0  # $ per bbl
    steam_cost_per_day = 500.0
    rod_break_penalty = 50000.0 # Cost to fix a snapped rod
    
    # Calculate daily production loosely based on physics
    daily_prod = (actual_temp / T_res) * 50 * (spm / 6.0) 
    cumulative_revenue = np.cumsum(daily_prod * oil_price)
    
    # Calculate expected penalty cost based on AI failure risk
    expected_penalty = (failure_risk / 100) * rod_break_penalty
    cumulative_cost = np.cumsum(np.full_like(days, steam_cost_per_day) + expected_penalty)
    
    npv = cumulative_revenue - cumulative_cost
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=days, y=cumulative_revenue, name="Cumulative Revenue ($)", line=dict(color='#00FF00', width=2)))
    fig3.add_trace(go.Scatter(x=days, y=cumulative_cost, name="Cumulative Cost + Expected Penalty ($)", line=dict(color='#FF4B4B', width=2)))
    fig3.add_trace(go.Scatter(x=days, y=npv, name="Net Present Value (NPV)", fill='tozeroy', line=dict(color='#00BFFF', width=3)))
    
    fig3.update_layout(title="Economic Viability: Balancing Production vs. Failure Risk", xaxis_title="Days", yaxis_title="Dollars ($)")
    st.plotly_chart(fig3, use_container_width=True)
    
with tab4:
    st.subheader("Live SCADA Data Stream")
    st.markdown("Raw data pipeline feeding into the machine learning inference engine.")
    
    if using_ml:
        st.dataframe(X_inference.head(st.session_state.current_day + 10).style.highlight_max(axis=0, color='red'), use_container_width=True)
    else:
        st.info("Physics Fallback Active. SCADA stream simulation requires ML Inference Mode.")
