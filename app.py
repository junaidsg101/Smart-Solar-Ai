import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# LOGGING & PATHS
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("SolarAI")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'solar_fault_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# ==========================================
# 1. ORCHESTRATION PATTERNS
# ==========================================
class SequencePattern:
    def execute(self, tasks: List[Callable], initial_input: Any, on_step=None) -> List[Any]:
        current_input = initial_input
        results = []
        for task in tasks:
            name = getattr(task, '__name__', task.__class__.__name__)
            if on_step: on_step("Pattern", f"Sequence executing: {name}")
            current_input = task(current_input)
            results.append(current_input)
        return results

class ParallelPattern:
    def execute(self, tasks: List[Callable], input_data: Any, max_workers=4, on_step=None) -> Dict[str, Any]:
        results = {}
        if on_step: on_step("Pattern", f"Parallel executing {len(tasks)} tasks with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task in tasks:
                name = getattr(task, '__name__', task.__class__.__name__)
                futures[executor.submit(task, input_data)] = name
            for future in as_completed(futures):
                name = futures[future]
                try: 
                    results[name] = future.result()
                except Exception as e: 
                    results[name] = f"Error: {e}"
        return results

# ==========================================
# 2. ML PREDICTOR (With Auto-Fallback)
# ==========================================
class SolarFaultPredictor:
    def __init__(self):
        self.model = None
        self.le = None
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            try:
                self.model = joblib.load(MODEL_PATH)
                self.le = joblib.load(ENCODER_PATH)
                return
            except Exception: 
                pass
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame({
            'panel_temp_c': [25, 45, 30, 25, 35, 40, 32, 28, 50, 60], 
            'irradiance': [500, 800, 600, 0, 900, 850, 700, 650, 800, 800],
            'power_kw': [20, 5, 25, 0, 10, 8, 22, 24, 10, 5], 
            'voltage_v': [230, 190, 231, 230, 220, 225, 230, 228, 230, 230],
            'resistance': [0.05, 2.5, 0.06, 0.05, 1.8, 0.05, 2.1, 0.06, 0.05, 0.05], 
            'ping_ms': [15, 15, 18, 2000, 15, 20, 1200, 45, 15, 15]
        })
        y = [0, 1, 0, 3, 4, 5, 6, 7, 8, 9] 
        self.model.fit(X, y)
        self.le = LabelEncoder()
        self.le.fit(["Normal", "Wiring_Fault", "Grid_Voltage_Fluctuation", "Monitoring_Offline", 
                     "Loose_Corroded_Wiring", "MPPT_Failure", "Sensor_Drift", "Comms_Protocol_Failure",
                     "Inverter_Overheat", "Thermal_Runaway"])

    def predict(self, data: dict) -> dict:
        features = pd.DataFrame([{
            'panel_temp_c': data.get('panel_temp_c', 25), 'irradiance': data.get('irradiance_wm2', 500),
            'power_kw': data.get('power_output_kw', 20), 'voltage_v': data.get('voltage_l1_v', 230),
            'resistance': data.get('resistance_ohm', 0.05), 'ping_ms': data.get('ping_ms', 15)
        }])
        pred_idx = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        fault = self.le.inverse_transform([pred_idx])[0] if self.le else "Unknown"
        return {"fault_type": fault, "confidence": float(max(proba))}

# ==========================================
# 3. AGENTS (Task Division & Execution)
# ==========================================
class DiagnosticAgent:
    def __init__(self): self.predictor = SolarFaultPredictor()
    def __call__(self, data, on_step=None):
        if on_step: on_step("Diagnostic Agent", "Running ML inference on preprocessed data...")
        data['diagnosis'] = self.predictor.predict(data)
        return data

class RepairAgent:
    def __call__(self, data, on_step=None):
        if on_step: on_step("Repair Agent", "Mapping fault to structured repair checklist...")
        fault = data.get('diagnosis', {}).get('fault_type', 'Unknown')
        
        steps = {
            "Wiring_Fault": [
                {"step": 1, "action": "Turn off DC disconnect", "explanation": "Isolate circuit to prevent arc flash.", "time": "2 mins"},
                {"step": 2, "action": "Inspect junction box", "explanation": "Look for physical damage or burn marks.", "time": "5 mins"},
                {"step": 3, "action": "Test continuity", "explanation": "Use multimeter to check wire integrity.", "time": "10 mins"},
                {"step": 4, "action": "Replace damaged wiring", "explanation": "Swap out degraded cables with UV-rated PV wire.", "time": "30 mins"}
            ],
            "Grid_Voltage_Fluctuation": [
                {"step": 1, "action": "Check grid connection", "explanation": "Verify main breaker and utility feed.", "time": "5 mins"},
                {"step": 2, "action": "Verify inverter profile", "explanation": "Ensure voltage ride-through settings are correct.", "time": "10 mins"},
                {"step": 3, "action": "Monitor voltage 24h", "explanation": "Log data to confirm if issue is transient.", "time": "24 hrs"},
                {"step": 4, "action": "Contact utility provider", "explanation": "Report sustained grid anomalies.", "time": "1 hr"}
            ],
            "Monitoring_Offline": [
                {"step": 1, "action": "Check ethernet/Wi-Fi", "explanation": "Verify physical network cables and signal.", "time": "5 mins"},
                {"step": 2, "action": "Reboot gateway", "explanation": "Power cycle the communication hub.", "time": "5 mins"},
                {"step": 3, "action": "Verify network creds", "explanation": "Ensure API keys and passwords haven't expired.", "time": "10 mins"},
                {"step": 4, "action": "Ping gateway", "explanation": "Confirm network latency is < 100ms.", "time": "2 mins"}
            ],
            "Loose_Corroded_Wiring": [
                {"step": 1, "action": "Inspect MC4 connectors", "explanation": "Check for green oxidation or loose clicks.", "time": "10 mins"},
                {"step": 2, "action": "Clean corrosion", "explanation": "Use contact cleaner and wire brush.", "time": "15 mins"},
                {"step": 3, "action": "Tighten connections", "explanation": "Torque to manufacturer specifications.", "time": "10 mins"},
                {"step": 4, "action": "Apply dielectric grease", "explanation": "Prevent future moisture ingress.", "time": "5 mins"}
            ],
            "MPPT_Failure": [
                {"step": 1, "action": "Reset MPPT tracker", "explanation": "Clear error via local HMI or remote API.", "time": "2 mins"},
                {"step": 2, "action": "Check firmware updates", "explanation": "Update inverter to latest stable release.", "time": "20 mins"},
                {"step": 3, "action": "Verify irradiance sensor", "explanation": "Ensure pyranometer is clean and aligned.", "time": "10 mins"},
                {"step": 4, "action": "Replace MPPT controller", "explanation": "Hardware replacement if software fails.", "time": "2 hrs"}
            ],
            "Sensor_Drift": [
                {"step": 1, "action": "Compare with reference", "explanation": "Check reading against a calibrated handheld meter.", "time": "10 mins"},
                {"step": 2, "action": "Run Kalman filter", "explanation": "Apply software smoothing to raw data.", "time": "5 mins"},
                {"step": 3, "action": "Recalibrate via software", "explanation": "Apply offset correction in the SCADA system.", "time": "15 mins"},
                {"step": 4, "action": "Replace faulty unit", "explanation": "Install new sensor if drift exceeds 5%.", "time": "30 mins"}
            ],
            "Comms_Protocol_Failure": [
                {"step": 1, "action": "Reset Modbus/CAN gateway", "explanation": "Clear protocol stack buffer.", "time": "5 mins"},
                {"step": 2, "action": "Inspect physical wiring", "explanation": "Check RS485/CAN bus termination resistors.", "time": "15 mins"},
                {"step": 3, "action": "Update protocol stack", "explanation": "Flash latest firmware to communication module.", "time": "20 mins"},
                {"step": 4, "action": "Verify packet capture", "explanation": "Analyze traffic for collisions or CRC errors.", "time": "15 mins"}
            ],
            "Inverter_Overheat": [
                {"step": 1, "action": "Check cooling fans", "explanation": "Ensure internal fans are spinning freely.", "time": "5 mins"},
                {"step": 2, "action": "Reduce load temporarily", "explanation": "Curtail output to lower thermal stress.", "time": "2 mins"},
                {"step": 3, "action": "Clean heat sinks", "explanation": "Remove dust and debris from fins.", "time": "20 mins"},
                {"step": 4, "action": "Verify ambient sensors", "explanation": "Ensure temp sensors aren't in direct sunlight.", "time": "10 mins"}
            ]
        }
        data['repair_plan'] = steps.get(fault, [{"step": 1, "action": "Escalate to senior tech", "explanation": "Manual review required for unknown faults.", "time": "N/A"}])
        return data

class ReportAgent:
    def __call__(self, data, on_step=None):
        if on_step: on_step("Report Agent", "Compiling final JSON report...")
        data['final_report'] = {
            "site_id": data.get('site_id', 'UNKNOWN'), 
            "diagnosis": data.get('diagnosis', {}), 
            "repair_plan": data.get('repair_plan', []),
            "severity": data.get('severity', 'Low'),
            "impact": data.get('impact', {})
        }
        return data

class ManagerAgent:
    def __init__(self, max_workers=4, timeout=30, enable_mlops=True, cache=True):
        self.max_workers = max_workers
        self.timeout = timeout
        self.enable_mlops = enable_mlops
        self.cache = cache
        self.diag = DiagnosticAgent()
        self.repair = RepairAgent()
        self.report = ReportAgent()

    def __call__(self, data, on_step=None):
        if on_step: on_step("Manager Agent", f"Received task for {data.get('site_id')}. Config: Workers={self.max_workers}, MLOps={self.enable_mlops}")
        
        # 1. Diagnose
        data = self.diag(data, on_step)
        fault = data.get('diagnosis', {}).get('fault_type', 'Normal')
        confidence = data.get('diagnosis', {}).get('confidence', 0.0)

        if confidence < 0.7 and fault != "Normal":
            if on_step: on_step("Manager Agent", f"Low confidence ({confidence:.2f}). Triggering Deep Diagnostics...")
            data['deep_diagnostics'] = True

        severity_map = {"Normal": "Low", "Sensor_Drift": "Medium", "Loose_Corroded_Wiring": "Medium", 
                        "MPPT_Failure": "High", "Comms_Protocol_Failure": "High", "Inverter_Overheat": "High",
                        "Grid_Voltage_Fluctuation": "Critical", "Monitoring_Offline": "Critical", "Wiring_Fault": "Critical"}
        data['severity'] = severity_map.get(fault, "Low")
        
        power = data.get('power_output_kw', 0)
        data['impact'] = {
            "energy_loss_kwh": round(power * 4, 2),
            "est_revenue_loss_usd": round(power * 4 * 0.15, 2)
        }

        # 2. Route via Patterns
        if fault in ["Monitoring_Offline", "Comms_Protocol_Failure"]:
            seq = SequencePattern()
            data = seq.execute([self._reboot, self._verify, lambda d: self.repair(d, on_step)], data, on_step)[-1]
        elif fault == "Grid_Voltage_Fluctuation":
            par = ParallelPattern()
            data['parallel_checks'] = par.execute([self._check_inv, self._check_grid], data, self.max_workers, on_step)
            data = self.repair(data, on_step)
        elif fault == "MPPT_Failure":
            par = ParallelPattern()
            data['mlops_checks'] = par.execute([self._check_mppt_logs, self._check_irradiance], data, self.max_workers, on_step)
            data = self.repair(data, on_step)
        elif fault == "Sensor_Drift":
            par = ParallelPattern()
            data['drift_checks'] = par.execute([self._run_kalman, self._check_ref], data, self.max_workers, on_step)
            data = self.repair(data, on_step)
        elif fault == "Inverter_Overheat":
            seq = SequencePattern()
            data = seq.execute([self._check_cooling, lambda d: self.repair(d, on_step)], data, on_step)[-1]
        else:
            data = self.repair(data, on_step)
            
        # 3. Report
        return self.report(data, on_step)

    def _reboot(self, d): return d
    def _verify(self, d): return d
    def _check_inv(self, d): return "Inverter OK"
    def _check_grid(self, d): return "Grid OK"
    def _check_mppt_logs(self, d): return "MPPT Logs Analyzed"
    def _check_irradiance(self, d): return "Irradiance Verified"
    def _run_kalman(self, d): return "Kalman Filter Applied"
    def _check_ref(self, d): return "Reference Checked"
    def _check_cooling(self, d): return "Cooling System Checked"

# ==========================================
# 4. CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="SITE-01")
    args = parser.parse_args()
    
    payload = {"site_id": args.site, "panel_temp_c": 35.0, "irradiance_wm2": 850,
               "power_output_kw": 10.0, "voltage_l1_v": 230.0, "resistance_ohm": 1.8, "ping_ms": 2000}
               
    manager = ManagerAgent(max_workers=4, timeout=30, enable_mlops=True, cache=True)
    result = manager(payload)
    print("\n--- FINAL DIAGNOSTIC REPORT ---")
    print(json.dumps(result.get('final_report', {}), indent=4))
    