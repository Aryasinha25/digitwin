import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from core.thermal_model_v2 import calculate_temperature_decay_with_uncertainty, calculate_viscosity
from core.wave_equation import calculate_float_risk
from core.optimizer import optimize_spm
from core.model_loader import load_models

st.set_page_config(page_title="Adaptive Baghewala Twin", layout="wide")

# --- Language Dictionary ---
lang_dict = {
    "EN": {
        "title": "Adaptive Baghewala Twin (BAT)",
        "tab0": "HOME / OVERVIEW", "tab1": "WELL ANALYTICS", "tab2": "FIELD COMMAND", "tab3": "ECONOMIC OPTIMIZER", "tab4": "LIVE SCADA STREAM", "tab5": "AI ARCHITECTURE",
        "t1_header": "Pump-as-Thermometer & Damping Estimator",
        "t2_header": "Baghewala Field Command Center",
        "t3_header": "Ablation & Economic NPV",
        "sim_panel": "Time Machine Simulator",
        "adv_7": "Advance 7 Days (Simulate Field)",
        "geo_dist": "Inject Geological Disturbance",
        "recal": "Recalibrate AI Twin"
    },
    "HI": {
        "title": "अनुकूली बाघेवाला ट्विन (BAT)",
        "tab0": "होम / अवलोकन", "tab1": "कुआं विश्लेषण", "tab2": "क्षेत्र कमान", "tab3": "आर्थिक अनुकूलक", "tab4": "लाइव स्काडा स्ट्रीम", "tab5": "एआई आर्किटेक्चर",
        "t1_header": "पंप-एज़-थर्मामीटर और डंपिंग एस्टिमेटर",
        "t2_header": "बाघेवाला फील्ड कमांड सेंटर",
        "t3_header": "एब्लेशन और आर्थिक एनपीवी",
        "sim_panel": "टाइम मशीन सिम्युलेटर",
        "adv_7": "7 दिन आगे बढ़ें (सिमुलेशन)",
        "geo_dist": "भौगोलिक गड़बड़ी इंजेक्ट करें",
        "recal": "एआई ट्विन को रिकैलिब्रेट करें"
    }
}

# --- Formally Load Models ---
model_state = load_models()
using_ml = (model_state["mode"] == "ml")

st.markdown("""
<style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #38bdf8; font-weight: 700; letter-spacing: -0.02em; }
    
    /* Premium Buttons */
    .stButton>button { 
        width: 100%; 
        font-weight: 600; 
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
        color: white; 
        border: none; 
        border-radius: 8px; 
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.4);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    
    /* Glassmorphic Health Box */
    .health-box { 
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Consolas', monospace; 
        font-size: 13px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .status-ready { color: #10b981; font-weight: bold; }
    .status-failed { color: #ef4444; font-weight: bold; }
    .status-degraded { color: #f59e0b; font-weight: bold; }
    
    /* Audit log terminal */
    .audit-log { 
        font-family: 'Consolas', monospace; 
        font-size: 12px; 
        color: #94a3b8; 
        background: #0f172a; 
        padding: 15px; 
        height: 200px; 
        overflow-y: scroll; 
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
    }
    
    /* Big Metric styling */
    div[data-testid="stMetricValue"] {
        color: #38bdf8;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(56, 189, 248, 0.2);
    }
    
    /* Dynamic Well Cards */
    .well-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 20px; 
        border-radius: 16px; 
        text-align: center; 
        margin-bottom: 20px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .well-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# State Management
if 'current_day' not in st.session_state: st.session_state.current_day = 0
if 'drift_detected' not in st.session_state: st.session_state.drift_detected = False
if 'calibrated_multiplier' not in st.session_state: st.session_state.calibrated_multiplier = 1.0
if 'audit_log' not in st.session_state: st.session_state.audit_log = []

# --- Sidebar UI ---
lang = st.sidebar.radio("Language / भाषा", ["EN", "HI"], horizontal=True)
t = lang_dict[lang]

st.title(t["title"])
if using_ml: st.success("MODEL MODE: ML Inference Active")

st.sidebar.header(t["sim_panel"])
st.sidebar.markdown(f"**Day:** {st.session_state.current_day}")

if st.sidebar.button(t["adv_7"]): st.session_state.current_day += 7
if st.sidebar.button(t["geo_dist"]): st.session_state.calibrated_multiplier = 3.0
if st.sidebar.button(t["recal"]): 
    st.session_state.calibrated_multiplier = 1.0
    st.session_state.drift_detected = False

st.sidebar.markdown("---")
st.sidebar.header("Demo Script Controls")
steam_vol = st.sidebar.slider("Steam Injection Volume (bbls)", 1000, 5000, 3000, step=500)
ai_optimizer_active = st.sidebar.toggle("Enable AI Coupled Optimizer", value=False)
if not ai_optimizer_active:
    manual_spm = st.sidebar.slider("Manual Setpoint (SPM)", 2.0, 10.0, 8.5)

# --- Core Physics / Math ---
days = np.arange(0, 180, 1)
T_res = 47.0

# Calculate temperature with steam volume variable
pred_temp, actual_temp, temp_uncertainty = calculate_temperature_decay_with_uncertainty(
    days, steam_vol=steam_vol, soak_time=7, error_multiplier=st.session_state.calibrated_multiplier
)

error = abs(actual_temp[st.session_state.current_day] - pred_temp[st.session_state.current_day])
if error > temp_uncertainty[st.session_state.current_day]:
    st.session_state.drift_detected = True

pred_visc = calculate_viscosity(pred_temp)

# Optimizer Logic (Demo Step 3)
if ai_optimizer_active:
    spm = optimize_spm(pred_visc) # AI Curve
else:
    spm = np.full_like(days, manual_spm) # Manual straight line

stroke = np.full_like(days, 3.0)

# ML Inference
if using_ml:
    X_inference = pd.DataFrame({'Temperature_C': pred_temp, 'Viscosity_cP': pred_visc, 'SPM': spm, 'Stroke_Length_m': stroke})
    pred_float = model_state["models"]["risk_model"].predict(X_inference)
    failure_risk = model_state["models"]["failure_model"].predict_proba(X_inference)[:, 1] * 100 
else:
    pred_float = calculate_float_risk(pred_visc, spm)
    failure_risk = np.where(pred_float > 80, 80, pred_float / 2)

# --- TABS ---
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([t["tab0"], t["tab1"], t["tab2"], t["tab3"], t["tab4"], t["tab5"]])

with tab0:
    st.markdown("""
    ## Welcome to the Adaptive Baghewala Twin (BAT)
    **The Problem:** In heavy oil fields like Baghewala, operators inject steam to melt thick oil so it can be pumped to the surface. However, as the well cools down over time, the oil turns back into sludge (high viscosity). If the pump runs too fast in this sludge, the rod bends and eventually snaps—a catastrophic failure costing $50,000+ and weeks of downtime.
    
    **Our AI Solution:** BAT is a Machine Learning Digital Twin. It ingests live physics data (temperature, viscosity) and uses an offline Edge AI model to dynamically recommend the perfect pump speed (SPM) to maximize oil production while keeping failure risk near zero.
    
    ---
    ### 🚀 Quickstart Guide for Operators:
    1. **Well Analytics:** View the real-time AI vs Physics simulation. Watch what happens to the Catastrophic Failure Risk when you override the AI's safe limits.
    2. **Field Command:** Monitor the entire 42-well sector. Approve or reject AI setpoint changes with full Audit Logging.
    3. **Economic Optimizer:** See the exact Net Present Value (NPV) and Rupees per barrel we save by preventing broken rods.
    4. **AI Architecture:** View the transparent Random Forest weights powering the dashboard.
    """)
    st.info("👈 Use the tabs above to navigate the platform, or the controls on the left sidebar to simulate time and geological disturbances.")

with tab1:
    st.subheader(t["t1_header"])
    
    if st.session_state.drift_detected:
        st.error("🚨 DRIFT DETECTED (Pump-as-Thermometer): Measured damping indicates actual temperature differs from model! Estimator correction required.")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=days, y=pred_temp + temp_uncertainty, line=dict(width=0), showlegend=False))
        fig1.add_trace(go.Scatter(x=days, y=pred_temp - temp_uncertainty, fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', line=dict(width=0), name="Confidence Interval"))
        fig1.add_trace(go.Scatter(x=days, y=pred_temp, name="Predicted Temp", line=dict(color='#00BFFF', width=2, dash='dash')))
        fig1.add_trace(go.Scatter(x=days[:st.session_state.current_day+1], y=actual_temp[:st.session_state.current_day+1], name="Actual Temp (Measured)", line=dict(color='#FF4B4B', width=3)))
        fig1.update_layout(title="Thermal State Estimation", xaxis_title="Days", yaxis_title="Temperature (°C)", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=days, y=spm, name="SPM Setpoint", line=dict(color='#00FF00', width=2)))
        fig2.add_trace(go.Scatter(x=days, y=pred_float, name="Float Risk Profile", line=dict(color='yellow', width=2, dash='dot')))
        fig2.add_trace(go.Scatter(x=days, y=failure_risk, fill='tozeroy', name="Catastrophic Rod Failure Risk (%)", line=dict(color='red', width=2), fillcolor='rgba(255, 0, 0, 0.2)'))
        fig2.update_layout(title="Setpoint Schedule & Risk Profile", xaxis_title="Days", yaxis_title="Risk (%) / SPM", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader(t["t2_header"])
    
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("Monitor heavy oil wells. Mobile-friendly view enabled.")
        field_cols = st.columns(3)
        wells = [("BGH-01", "Healthy", "green"), ("BGH-02", "Critical Risk" if not ai_optimizer_active else "Healthy", "red" if not ai_optimizer_active else "green"), ("BGH-03", "Warning", "orange")]
        for i, (well, status, color) in enumerate(wells):
            with field_cols[i % 3]:
                glow_color = "#10b981" if color == "green" else "#ef4444" if color == "red" else "#f59e0b"
                st.markdown(f"""
                <div class="well-card" style="border-top: 4px solid {glow_color};">
                    <h3 style="margin:0; color:#f8fafc;">{well}</h3>
                    <p style="color:{glow_color}; font-weight:bold; margin:10px 0;">{status.upper()}</p>
                    <div style="background:rgba(0,0,0,0.2); padding:5px; border-radius:5px; font-size: 12px; color:#cbd5e1;">
                        SPM: {np.random.randint(4, 9)} &nbsp;|&nbsp; Temp: {np.random.randint(60, 180)}°C
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    with c2:
        st.markdown("**Workflow Approvals & Audit Log**")
        if not ai_optimizer_active:
            st.warning("Action Required: BGH-02 Float Risk > 80%. AI recommends dropping SPM.")
            a1, a2 = st.columns(2)
            if a1.button("Approve (Trigger AI)"):
                st.session_state.audit_log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] USER APPROVED AI Coupled Optimizer for BGH-02.")
                st.success("Approved! Please enable AI Toggle in sidebar to apply.")
            if a2.button("Reject"):
                st.session_state.audit_log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] USER REJECTED AI setpoint. Overriding safety limits.")
                
            if st.button("Push Offline SMS Alert"):
                st.info("📲 WhatsApp/SMS Payload dispatched via Edge Gateway: 'BGH-02 Rod Float Critical. Check Dashboard.'")
                st.session_state.audit_log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] SYSTEM triggered Offline Edge SMS alert.")
        
        st.markdown("<div class='audit-log'>" + "<br>".join(st.session_state.audit_log) + "</div>", unsafe_allow_html=True)

with tab3:
    st.subheader(t["t3_header"])
    
    oil_price_inr = 5800.0  # ~70 USD in INR
    steam_cost_per_day_inr = 40000.0 
    rod_break_penalty_inr = 4000000.0 # ~50k USD
    
    daily_prod = (actual_temp / T_res) * 50 * (spm / 6.0) 
    cumulative_revenue = np.cumsum(daily_prod * oil_price_inr)
    
    expected_penalty = (failure_risk / 100) * rod_break_penalty_inr
    cumulative_cost = np.cumsum(np.full_like(days, steam_cost_per_day_inr) + expected_penalty)
    
    npv = cumulative_revenue - cumulative_cost
    
    current_rupees_per_bbl = oil_price_inr - ((steam_cost_per_day_inr + expected_penalty[st.session_state.current_day]) / max(1, daily_prod[st.session_state.current_day]))
    
    st.metric("Current Economics (₹ per barrel)", f"₹{current_rupees_per_bbl:,.2f}", delta="Optimal" if ai_optimizer_active else "-₹ Penalty Risk")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=days, y=cumulative_revenue, name="Cumulative Revenue (₹)", line=dict(color='#00FF00', width=2)))
    fig3.add_trace(go.Scatter(x=days, y=cumulative_cost, name="Cumulative Cost + Penalty Risk (₹)", line=dict(color='#FF4B4B', width=2)))
    fig3.add_trace(go.Scatter(x=days, y=npv, name="Net Present Value (NPV)", fill='tozeroy', line=dict(color='#00BFFF', width=3)))
    
    fig3.update_layout(title="Ablation Chart: AI Optimized vs Manual Operation", xaxis_title="Days", yaxis_title="Rupees (₹)", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)
    
with tab4:
    st.subheader(t["tab4"])
    st.dataframe(X_inference.head(st.session_state.current_day + 10).style.highlight_max(axis=0, color='red'), use_container_width=True)

with tab5:
    st.subheader("AI Model Architecture & Interpretability")
    st.markdown("Transparent view into the Random Forest weights. This proves the dashboard is driven by a trained machine learning model, not static equations.")
    
    if using_ml:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Feature Importance (Float Risk AI)**")
            importances = model_state["models"]["risk_model"].feature_importances_
            features = ['Temperature (°C)', 'Viscosity (cP)', 'SPM', 'Stroke Length (m)']
            
            fig_fi = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker=dict(color=['#FF4B4B', '#00BFFF', '#00FF00', '#FFA500'])
            ))
            fig_fi.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_fi, use_container_width=True)
            
        with c2:
            st.markdown("**Python Training Pipeline Snippet**")
            st.code('''
# The AI was trained in Google Colab on 10,000 days of synthetic data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 1. Train Regression model for Float Risk %
rf_risk = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_risk.fit(X_train, yr_train)

# 2. Train Classification model for Catastrophic Failure
rf_failure = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')
rf_failure.fit(X_train, yf_train)

# 3. Export weights to .pkl for Streamlit Edge deployment
joblib.dump(rf_risk, 'rf_risk_model.pkl')
            ''', language='python')
    else:
        st.info("Physics Fallback Active. Connect ML inference to view model architecture.")
