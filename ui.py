from agents.manager_agent import ManagerAgent
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ⚙️ PAGE CONFIG & LIGHT MODE ENFORCEMENT ────────────────────────
st.set_page_config(page_title="Smart Solar AI", layout="wide", page_icon="🌞")

st.markdown("""
<style>
    :root {
        --primary-color: #1E88E5;
        --background-color: #FFFFFF;
        --secondary-background-color: #F5F5F5;
        --text-color: #212121;
    }
    .stApp { background-color: var(--background-color) !important; color: var(--text-color) !important; }
    section[data-testid="stSidebar"] { background-color: var(--secondary-background-color) !important; }
    div[data-testid="stMetric"] { background-color: #F0F4F8 !important; border: 1px solid #E0E0E0 !important; border-radius: 8px !important; padding: 12px !important; }
    h1, h2, h3 { color: #1565C0 !important; }
    .stTabs [data-baseweb="tab"] { background-color: var(--secondary-background-color); color: var(--text-color); }
    .stTabs [aria-selected="true"] { background-color: var(--primary-color) !important; color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌞 Smart Solar AI Diagnostic Dashboard")

with st.sidebar:
    st.header("⚙️ Settings")
    st.text_input("OpenAI API Key", type="password",
                  help="For advanced LLM report generation")
    st.caption("Theme: ☀️ Light Mode Enforced")
    st.divider()
    st.info("**System Status:** 🟢 Operational\n**MTTR Target:** < 5 mins\n**Active Agents:** 5")

tab1, tab2 = st.tabs(["🔍 Run Diagnosis", "📊 System Health & Charts"])

# --- TAB 1: DIAGNOSIS ---
with tab1:
    st.subheader("Run Diagnostic Pipeline")
    col1, col2 = st.columns([1, 2])

    with col1:
        site_id = st.selectbox("Site ID", ["SITE-01", "SITE-02"])
        voltage = st.slider("Simulated L1 Voltage (V)", 180.0, 240.0, 230.0)
        ping = st.slider("Simulated Ping (ms)", 10.0, 2000.0, 15.0)

        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Agents analyzing data..."):
                payload = {
                    "site_id": site_id,
                    "panel_temp_c": 32.4,
                    "irradiance_wm2": 800,
                    "power_output_kw": 12.1,
                    "voltage_l1_v": voltage,
                    "resistance_ohm": 2.50 if voltage < 200 else 0.05,
                    "ping_ms": ping
                }
                manager = ManagerAgent()
                result = manager(payload)
                st.session_state['last_result'] = result

    with col2:
        if 'last_result' in st.session_state:
            report = st.session_state['last_result'].get('final_report', {})
            diagnosis = report.get('diagnosis', {})

            c1, c2 = st.columns(2)
            c1.metric("Detected Fault", diagnosis.get('fault_type', 'Unknown'))
            c2.metric("Confidence",
                      f"{diagnosis.get('confidence', 0)*100:.1f}%")

            st.subheader("🛠️ Repair Steps")
            for step in report.get('repair_plan', []):
                st.markdown(f"- {step}")
            st.success(
                f"✅ Report compiled. Trace ID: {report.get('timestamp', 'N/A')}")

# --- TAB 2: CHARTS (SIZE HANDLED) ---
with tab2:
    st.subheader("System Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MTTR", "4.2 mins", "-60%")
    c2.metric("Accuracy", "94.5%")
    c3.metric("Uptime", "99.8%")
    st.markdown("---")

    np.random.seed(42)
    dates = pd.date_range("2026-09-01", "2026-09-30", freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "power_kw": np.clip(np.random.normal(25, 8, len(dates)), 0, 50),
        "temp_c": np.random.normal(30, 5, len(dates))
    })

    st.subheader("⚡ Power Output Trend (Plotly)")
    fig1 = px.line(df, x="timestamp", y="power_kw",
                   title="Hourly Power Output (kW)")
    fig1.update_layout(
        height=400, margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font=dict(color="#212121")
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🌡️ Temperature vs Power (Matplotlib)")
    fig2, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(df["temp_c"], df["power_kw"], alpha=0.5, color="#1E88E5")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Temp vs Power Output")
    ax.grid(True, alpha=0.3)
    fig2.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

