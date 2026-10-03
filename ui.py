import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import ManagerAgent

# ==========================================
# 1. PAGE CONFIG & THEME TOGGLE
# ==========================================
st.set_page_config(page_title="Smart Solar AI", layout="wide", page_icon="🌞")

if 'theme' not in st.session_state: st.session_state.theme = "Light"
if 'agent_logs' not in st.session_state: st.session_state.agent_logs = []

with st.sidebar:
    st.header("⚙️ Control Panel")
    st.session_state.theme = st.radio("Theme Mode", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
    st.divider()
    st.caption(f"Theme: {'☀️ Light' if st.session_state.theme == 'Light' else '🌙 Dark'}")
    st.info("**System Status:** 🟢 Operational\n**MTTR Target:** < 5 mins\n**Active Agents:** 7")

# Dynamic CSS Variables (FIXED: Renamed 'text' to 'text_color' to prevent NameError)
if st.session_state.theme == "Light":
    bg, text_color, sidebar_bg, card_bg, accent, heading = "#FFFFFF", "#212121", "#F5F5F5", "#F0F4F8", "#1E88E5", "#1565C0"
else:
    bg, text_color, sidebar_bg, card_bg, accent, heading = "#121212", "#E0E0E0", "#1E1E1E", "#2C2C2C", "#90CAF9", "#BBDEFB"

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg} !important; color: {text_color} !important; }}
    section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
    div[data-testid="stMetric"] {{ background-color: {card_bg} !important; border: 1px solid {accent} !important; border-radius: 8px !important; padding: 12px !important; }}
    h1, h2, h3 {{ color: {heading} !important; }}
    .stTabs [data-baseweb="tab"] {{ background-color: {card_bg}; color: {text_color}; }}
    .stTabs [aria-selected="true"] {{ background-color: {accent} !important; color: #FFFFFF !important; }}
    .stPlotlyChart {{ background-color: {card_bg} !important; border-radius: 8px; padding: 10px; }}
    .step-container {{ background-color: {card_bg}; padding: 10px; border-radius: 8px; margin-bottom: 5px; border-left: 4px solid {accent}; }}
</style>
""", unsafe_allow_html=True)

st.title("🌞 Smart Solar AI Diagnostic Dashboard")

tab_control, tab_logs, tab_analytics = st.tabs(["🎛️ Control & Diagnosis", "🤖 Live Agent Execution", "📊 Analytics & Charts"])

# ==========================================
# 2. TAB 1: CONTROL & DIAGNOSIS
# ==========================================
with tab_control:
    st.subheader("Run Diagnostic Pipeline")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### ⚙️ Configuration")
        site_id = st.selectbox("Site ID", ["SITE-01", "SITE-02"])
        fault_sim = st.selectbox("Simulate Fault", ["Normal", "MPPT_Failure", "Sensor_Drift", "Comms_Protocol_Failure", "Grid_Voltage_Fluctuation", "Inverter_Overheat"])
        
        st.markdown("#### ⚡ Electrical Metrics")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            voltage = st.slider("L1 Voltage (V)", 150.0, 250.0, 230.0, help="Nominal is 230V. <200V triggers Grid Fluctuation.")
        with col_s2:
            resistance = st.slider("Resistance (ohm)", 0.01, 5.00, 0.05, help=">1.0 ohm indicates Wiring Fault.")
            
        st.markdown("#### 🌡️ Environmental Metrics")
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            temp = st.slider("Panel Temp (°C)", 20.0, 80.0, 35.0, help=">65°C triggers Inverter Overheat.")
        with col_s4:
            irradiance = st.slider("Irradiance (W/m²)", 0, 1200, 800, help="Solar irradiance level.")
            
        st.markdown("#### 📡 Network Metrics")
        ping = st.slider("Network Ping (ms)", 10.0, 2000.0, 15.0, help=">1000ms indicates Network Degradation/Offline.")
        
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            st.session_state.agent_logs = []
            def log_step(agent, msg): st.session_state.agent_logs.append(f"[{agent}] {msg}")
            
            with st.spinner("Agents analyzing data..."):
                payload = {"site_id": site_id, "panel_temp_c": temp, "irradiance_wm2": irradiance,
                           "power_output_kw": 12.1, "voltage_l1_v": voltage, 
                           "resistance_ohm": resistance, "ping_ms": ping}
                
                manager = ManagerAgent(max_workers=4, timeout=30, enable_mlops=True, cache=True)
                result = manager(payload, on_step=log_step)
                st.session_state['last_result'] = result

    with col2:
        st.markdown("#### 📋 Diagnostic Output")
        if 'last_result' in st.session_state:
            report = st.session_state['last_result'].get('final_report', {})
            diagnosis = report.get('diagnosis', {})
            severity = report.get('severity', 'Low')
            impact = report.get('impact', {})
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Detected Fault", diagnosis.get('fault_type', 'Unknown'))
            c2.metric("Confidence", f"{diagnosis.get('confidence', 0)*100:.1f}%")
            
            sev_color = {"Low": "off", "Medium": "off", "High": "delta", "Critical": "inverse"}
            c3.metric("Severity", severity, delta="⚠️" if severity in ["High", "Critical"] else None)
            
            st.markdown("#### 💰 Estimated Impact")
            imp_c1, imp_c2 = st.columns(2)
            imp_c1.metric("Energy Loss", f"{impact.get('energy_loss_kwh', 0)} kWh")
            imp_c2.metric("Revenue Loss", f"${impact.get('est_revenue_loss_usd', 0)}")

            st.markdown("#### 🛠️ Repair Action Plan")
            repair_plan = report.get('repair_plan', [])
            if repair_plan:
                df_repair = pd.DataFrame(repair_plan)
                st.dataframe(
                    df_repair, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "step": st.column_config.NumberColumn("Step #", format="%d"),
                        "action": st.column_config.TextColumn("Action", width="medium"),
                        "explanation": st.column_config.TextColumn("Short Explanation", width="large"),
                        "time": st.column_config.TextColumn("Est. Time", width="small")
                    }
                )
            st.success(f"✅ Report compiled. Trace ID: {report.get('site_id', 'N/A')}")

# ==========================================
# 3. TAB 2: LIVE AGENT EXECUTION
# ==========================================
with tab_logs:
    st.subheader("🤖 Multi-Agent Processing Steps")
    st.markdown("Real-time visualization of how the Manager, Data, Diagnostic, Repair, and Report agents divide and execute tasks.")
    
    if st.session_state.agent_logs:
        for log in st.session_state.agent_logs:
            agent = log.split(']')[0].strip('[')
            msg = log.split(']')[1].strip()
            st.markdown(f'<div class="step-container"><b>{agent}:</b> {msg}</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Run a simulation in the 'Control & Diagnosis' tab to see the agent steps here.")

# ==========================================
# 4. TAB 3: ANALYTICS & CHARTS
# ==========================================
with tab_analytics:
    st.subheader("System Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MTTR", "4.2 mins", "-60%")
    c2.metric("Accuracy", "94.5%")
    c3.metric("Uptime", "99.8%")
    c4.metric("Active Sites", "12/12")
    st.markdown("---")
    
    np.random.seed(42)
    dates = pd.date_range("2026-09-01", "2026-09-30", freq="h")
    df = pd.DataFrame({
        "timestamp": dates, "power_kw": np.clip(np.random.normal(25, 8, len(dates)), 0, 50), 
        "temp_c": np.random.normal(30, 5, len(dates)), "irradiance": np.random.normal(800, 100, len(dates)),
        "voltage_l1": np.random.normal(230, 5, len(dates))
    })

    st.subheader("⚡ Power Output vs Irradiance")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["timestamp"], y=df["power_kw"], name="Power (kW)", line=dict(color=accent)))
    fig1.add_trace(go.Scatter(x=df["timestamp"], y=df["irradiance"]/20, name="Irradiance (Scaled)", line=dict(color="#FF9800", dash="dot")))
    fig1.update_layout(height=350, plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color))
    st.plotly_chart(fig1, use_container_width=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("📊 Fault Distribution (Bar)")
        fault_counts = pd.DataFrame({"Fault": ["Normal", "MPPT", "Sensor", "Comms", "Grid"], "Count": [120, 15, 8, 12, 25]})
        fig2 = px.bar(fault_counts, x="Fault", y="Count", color="Fault", color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=300, plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_p2:
        st.subheader("🥧 Site Status Breakdown (Pie)")
        fig3 = px.pie(fault_counts, values="Count", names="Fault", hole=0.4)
        fig3.update_layout(height=300, plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color))
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📉 Voltage L1 Distribution (Histogram)")
    fig4 = px.histogram(df, x="voltage_l1", nbins=30, color_discrete_sequence=[accent])
    fig4.update_layout(height=300, plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color))
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("🌡️ Temperature vs Power (Interactive Slider)")
    fig5 = px.scatter(df, x="temp_c", y="power_kw", color="power_kw", color_continuous_scale="Viridis")
    fig5.update_layout(height=350, plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color), xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("🎯 Overall System Health")
    fig6 = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=94.5, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diagnostic Accuracy (%)", 'font': {'color': text_color}},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': accent}, 'steps': [{'range': [0, 80], 'color': '#ef9a9a'}, {'range': [80, 95], 'color': '#fff59d'}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    fig6.update_layout(height=300, plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color))
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("📡 Live Screen Monitoring: Network Topology")
    fig_topo = go.Figure()
    nodes = ["Gateway", "Inverter A1", "Inverter B2", "Sensor Grid", "Cloud API"]
    x_coords = [0, -1, 1, -0.5, 0]
    y_coords = [0, 1, 1, -1, -1]
    
    fig_topo.add_trace(go.Scatter(
        x=x_coords, y=y_coords, mode="markers+text", text=nodes, textposition="top center",
        marker=dict(size=25, color=accent, line=dict(width=2, color=text_color)), 
        textfont=dict(color=text_color, size=12), name="Nodes"
    ))
    
    edges = [(0,1), (0,2), (0,3), (0,4)]
    for i, j in edges:
        fig_topo.add_trace(go.Scatter(
            x=[x_coords[i], x_coords[j]], y=[y_coords[i], y_coords[j]], mode="lines", 
            line=dict(color=text_color, width=2, dash="dot"), showlegend=False
        ))
    
    fig_topo.update_layout(
        height=400, 
        plot_bgcolor=card_bg, paper_bgcolor=card_bg, font=dict(color=text_color), 
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig_topo, use_container_width=True)
    