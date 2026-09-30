with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Imports
code = code.replace(
    "from core.wave_equation import calculate_float_risk",
    "from core.wave_equation import calculate_float_risk, generate_dynacard"
)
code = code.replace(
    "from core.optimizer import optimize_spm",
    "from core.optimizer import optimize_spm, optimize_css_cycle_steam"
)

# 2. Sidebar controls
old_sidebar = """st.sidebar.markdown("---")
st.sidebar.header("Demo Script Controls")
steam_vol = st.sidebar.slider("Steam Injection Volume (bbls)", 1000, 5000, 3000, step=500)
ai_optimizer_active = st.sidebar.toggle("Enable AI Coupled Optimizer", value=False)
if not ai_optimizer_active:
    manual_spm = st.sidebar.slider("Manual Setpoint (SPM)", 2.0, 10.0, 8.5)"""

new_sidebar = """st.sidebar.markdown("---")
st.sidebar.header("Demo Script Controls")
control_policy = st.sidebar.radio("Optimization Strategy", ["Baseline (Manual)", "SRP-Only AI", "Fully Coupled AI"])

if control_policy != "Fully Coupled AI":
    steam_vol = st.sidebar.slider("Steam Injection Volume (bbls)", 1000, 5000, 3000, step=500)
else:
    steam_vol = optimize_css_cycle_steam(t_res=47.0)
    st.sidebar.info(f"Coupled AI Selected Steam: {steam_vol:.0f} bbls")

if control_policy == "Baseline (Manual)":
    manual_spm = st.sidebar.slider("Manual Setpoint (SPM)", 2.0, 10.0, 8.5)
else:
    manual_spm = 8.5 # Fallback
    
initial_bias = st.sidebar.slider("Initial Sensor Bias (°C)", 0.0, 50.0, 0.0, step=5.0)"""
code = code.replace(old_sidebar, new_sidebar)

# 3. Core Physics
old_physics = """days = np.arange(0, 180, 1)
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
    spm = np.full_like(days, manual_spm) # Manual straight line"""

new_physics = """days = np.arange(0, 180, 1)
T_res = 47.0

# Calculate temperature with steam volume variable
pred_temp_base, actual_temp, temp_uncertainty = calculate_temperature_decay_with_uncertainty(
    days, steam_vol=steam_vol, soak_time=7, error_multiplier=st.session_state.calibrated_multiplier
)

# Apply initial bias (EnKF Robustness demonstration)
bias_decay = np.exp(-days * 0.1) # Converges over ~30 days
pred_temp = pred_temp_base + (initial_bias * bias_decay)

error = abs(actual_temp[st.session_state.current_day] - pred_temp[st.session_state.current_day])
if error > temp_uncertainty[st.session_state.current_day] and initial_bias == 0:
    st.session_state.drift_detected = True

pred_visc = calculate_viscosity(pred_temp)

# Optimizer Logic (Demo Step 3)
if control_policy != "Baseline (Manual)":
    spm = optimize_spm(pred_visc) # AI Curve
else:
    spm = np.full_like(days, manual_spm) # Manual straight line"""
code = code.replace(old_physics, new_physics)

# 4. Tab 2 modifications
code = code.replace("not ai_optimizer_active", "control_policy == 'Baseline (Manual)'")

# 5. Tab 1 Dynacards
tab1_end = "st.plotly_chart(fig2, use_container_width=True)"
tab1_add = """st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Pump-as-Thermometer (Live Dynamometer Card)")
    st.markdown("Physics-informed visual approximation. The area and shape of the surface load card distorts mathematically as downhole fluid viscosity (resistance) rises.")
    
    c_dyna, c_metrics = st.columns([2, 1])
    with c_dyna:
        d_pos, d_load = generate_dynacard(pred_visc[st.session_state.current_day], spm[st.session_state.current_day])
        fig_dyna = go.Figure()
        fig_dyna.add_trace(go.Scatter(x=d_pos, y=d_load, mode='lines', line=dict(color='#38bdf8', width=3), fill='toself', fillcolor='rgba(56, 189, 248, 0.2)'))
        fig_dyna.update_layout(title=f"Surface Dynacard @ Day {st.session_state.current_day}", xaxis_title="Position (Normalized)", yaxis_title="Load (lbs)", template="plotly_dark", height=400)
        st.plotly_chart(fig_dyna, use_container_width=True)
        
    with c_metrics:
        st.metric("Viscosity (cP)", f"{pred_visc[st.session_state.current_day]:.0f}")
        st.metric("SPM", f"{spm[st.session_state.current_day]:.1f}")
        st.metric("Float Risk", f"{failure_risk[st.session_state.current_day]:.1f}%")"""
code = code.replace(tab1_end, tab1_add)

# 6. Tab 3 Ablation Chart
old_tab3 = r'''oil_price_inr = 5800.0  # ~70 USD in INR
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
    st.plotly_chart(fig3, use_container_width=True)'''

new_tab3 = r'''# 3-Way Ablation Study
    oil_price_inr = 5800.0  # ~70 USD in INR
    steam_cost_per_day_inr = 40000.0 
    rod_break_penalty_inr = 4000000.0 # ~50k USD
    
    def simulate_policy_economics(policy_name, man_steam, man_spm):
        if policy_name == "Fully Coupled AI":
            s_vol = optimize_css_cycle_steam(T_res)
        else:
            s_vol = man_steam
            
        p_t, a_t, _ = calculate_temperature_decay_with_uncertainty(days, steam_vol=s_vol, soak_time=7, error_multiplier=1.0)
        p_v = calculate_viscosity(a_t)
        
        if policy_name == "Baseline (Manual)":
            s_spm = np.full_like(days, man_spm)
        else:
            s_spm = optimize_spm(p_v)
            
        p_f = calculate_float_risk(p_v, s_spm)
        f_r = np.where(p_f > 80, 80, p_f / 2)
        
        d_p = (a_t / T_res) * 50 * (s_spm / 6.0)
        c_rev = np.cumsum(d_p * oil_price_inr)
        
        steam_op_cost = np.full_like(days, steam_cost_per_day_inr)
        e_pen = (f_r / 100) * rod_break_penalty_inr
        c_cost = np.cumsum(steam_op_cost + e_pen)
        
        return c_rev - c_cost, np.sum(d_p), np.max(f_r)

    # Use 3000 as fallback if control_policy == Fully Coupled and we want to show hypothetical alternatives
    base_steam_for_chart = steam_vol if control_policy != "Fully Coupled AI" else 3000
    
    npv_base, prod_base, risk_base = simulate_policy_economics("Baseline (Manual)", base_steam_for_chart, manual_spm)
    npv_srp, prod_srp, risk_srp = simulate_policy_economics("SRP-Only AI", base_steam_for_chart, manual_spm)
    npv_cpl, prod_cpl, risk_cpl = simulate_policy_economics("Fully Coupled AI", base_steam_for_chart, manual_spm)
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Baseline NPV", f"₹{npv_base[-1]:,.0f}", help=f"Max Float Risk: {risk_base:.1f}%")
    col_b.metric("SRP-Only NPV", f"₹{npv_srp[-1]:,.0f}", delta=f"₹{npv_srp[-1] - npv_base[-1]:,.0f} vs Base", help=f"Max Float Risk: {risk_srp:.1f}%")
    col_c.metric("Fully Coupled NPV", f"₹{npv_cpl[-1]:,.0f}", delta=f"₹{npv_cpl[-1] - npv_srp[-1]:,.0f} vs SRP", help=f"Max Float Risk: {risk_cpl:.1f}%")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=days, y=npv_base, name="Baseline (Manual)", line=dict(color='#a3a3a3', width=2, dash='dash')))
    fig3.add_trace(go.Scatter(x=days, y=npv_srp, name="SRP-Only AI", line=dict(color='#3b82f6', width=2)))
    fig3.add_trace(go.Scatter(x=days, y=npv_cpl, name="Fully Coupled AI", line=dict(color='#10b981', width=3)))
    
    fig3.update_layout(title="Ablation Chart: NPV Trajectories (The Value of Coupling)", xaxis_title="Days", yaxis_title="Net Present Value (₹)", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)'''

code = code.replace(old_tab3, new_tab3)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Patch successful!")
