# Smart Solar Ai,  Multi-Agent Technical Issue Solver

## 📌 Description
A Python-based multi-agentic application that detects and solves solar technical issues using **Sequence**, **Parallel**, and **Magantic** patterns. It integrates data handling, machine learning, MLOps, and multi-agent frameworks to diagnose faults such as wiring/grounding faults, grid voltage fluctuation, monitoring system offline, and loose or corroded wiring.

## 🚀 Features
- Multi-agent architecture with manager and specialist agents
- Hybrid pattern support: Sequence + Parallel + Magantic
- Real-time solar data ingestion (CSV, sensors, APIs)
- ML-based fault detection and prediction
- Automated repair recommendations and reporting
- MLOps-ready with model retraining and monitoring


## 🧠 Multi-Agentic Patterns
| Pattern | Description | Use Case |
|---------|-------------|----------|
| Sequence | Step-by-step execution | Ordered diagnosis and repair |
| Parallel | Simultaneous task execution | Multi-source data checks |
| Magantic | Manager delegates to specialist agents | Complex or unknown faults |

## Technical Issue
| # | Technical Issue | Solution |
| --- | --- |--- |
| 1 | Loose or corroded wiring | Tighten connections and replace damaged cables. |
| 2 | Monitoring system offline | Check internet, power, reset gateway, update firmware. |
| 3 | Wiring/grounding fault | Tighten connections, test ground, repair damaged cables. |
| 4 | Grid voltage fluctuation | Use voltage stabilizer, check grid code, adjust inverter settings. |

##  Multi-Agent Pattern
| # | Pattern | Structure | Best For | Solar Technical Issue | Agentic Flow | Result |
|----|---------|-----------|----------|----------------------|--------------|--------|
| 1 | Sequence | Step-by-step | Ordered diagnosis and repair | Monitoring system offline | Detect → collect logs → diagnose → fix → verify | System back online |
| 2 | Parallel | Simultaneous tasks | Multi-source data checks | Grid voltage fluctuation | Voltage sensor + grid API + inverter logs run together | Fast fluctuation profile |
| 3 | Magantic | Manager + specialist agents | Complex or unknown faults | Wiring/grounding fault | Manager splits tasks to safety, electrical, and vision agents | Safe repair plan |
|4 | Sequence + Parallel + Magantic | Hybrid multi-agent flow | End-to-end solar troubleshooting | Loose or corroded wiring | Parallel inspect + Magantic decide + Sequence repair | Complete resolution |
| 5 | Parallel + Magantic | Fast detection with expert control | Real-time grid issues | Grid voltage fluctuation | Parallel sensors feed Magantic manager for action | Quick voltage correction |
| 6 | Sequence + Magantic | Managed stepwise recovery | Offline monitoring recovery | Monitoring system offline | Magantic plans, Sequence executes reboot and checks | Stable monitoring restored |

## Project Structure

```
solar-multi-agent/
│
├── app.py                      # Main CLI entry point & core agent/pattern logic
├── ui.py                       # Interactive Streamlit UI (Light/Dark mode)
├── config                     # Configuration file (theme, server, LLM, app settings)
├── requirements.txt            # Python dependencies
├── README.md                   # Project structure and documentation
│
├── agents/                     # Multi-Agent System definitions
│   ├── __init__.py
│   ├── manager_agent.py        # Central orchestrator routing tasks based on fault type
│   ├── data_agent.py           # Ingests and preprocesses raw sensor data
│   ├── diagnostic_agent.py     # Uses ML model to classify the specific fault
│   ├── repair_agent.py         # Generates step-by-step repair recommendations
│   └── report_agent.py         # Compiles the final diagnostic report with impact scoring
│
├── patterns/                   # Agent orchestration patterns
│   ├── __init__.py
│   ├── sequence.py             # Strict step-by-step execution (e.g., reboot sequences)
│   ├── parallel.py             # Concurrent task execution (e.g., multi-source data checks)
│   └── magnetic.py             # Manager delegates to specialist agents dynamically
│
├── data/                       # Sample and historical sensor data (CSV, 20 rows each)
│   ├── grid_voltage.csv        # 3-phase grid voltage and frequency
│   ├── inverter_logs.csv       # Inverter efficiency and error codes
│   ├── wiring_faults.csv       # Historical wiring/grounding fault logs
│   ├── battery_data.csv        # Battery State of Charge (SoC) and health
│   └── monitoring_status.csv   # Network ping, packet loss, and connection status
│
├── ml_models/                  # Directory for saved model artifacts (generated at runtime)
│   ├── solar_fault_model.pkl   # Trained Scikit-learn model
│   └── label_encoder.pkl       # Maps numeric predictions to fault names
│
└── utils/                      # Utility and helper functions
    ├── __init__.py
    ├── logger.py               # Configures rotating file and console logging
    └── helpers.py              # Config loading, timestamp formatting, path helpers
```

## Setup
1)  Create a virtual environment and install dependencies:
- pip install -r requirements.txt
2) Operation Performing:
- python run app.py
3) Run for Streamlit ui:
- streamlit run ui.py
---
Deploying
https://smart-solar-ai-mlops-dev.streamlit.app/

## Technology Stack & Tools

| Category | Technology / Tool | Purpose |
| --- | --- | --- |
| **Language** | Python 3.10+ | Core application logic and agent orchestration. |
| **Data Processing** | Pandas, NumPy | Data cleaning, transformation, and statistical analysis. |
| **Visualization** | Plotly, Matplotlib | Interactive, responsive charts (line, bar, pie, histogram, gauge, topology) in the UI. |
| **Machine Learning** | Scikit-learn | Fault classification models (Random Forest) with label encoding. |
| **MLOps** | MLflow, Joblib | Model tracking, artifact saving (`solar_fault_model.pkl`), and deployment monitoring. |
| **Multi-Agent Framework** | Custom Python classes | Orchestrating Sequence, Parallel, and Magnetic agent patterns with `ThreadPoolExecutor`. |
| **Frontend / UI** | Streamlit | Rapid development of the internal dashboard with dynamic Light/Dark CSS theming. |
| **Configuration** | TOML | Secure, readable storage of API keys, model paths, and site configs. |
| **Logging** | Python `logging` | Audit trails of agent actions, errors, and ML predictions. |
