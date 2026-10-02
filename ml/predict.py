import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from utils.helpers import get_project_root


class SolarFaultPredictor:
    def __init__(self, model_path: str = None):
        self.model = None
        self.le = None
        if model_path is None:
            model_path = os.path.join(
                get_project_root(), "ml", "models", "solar_fault_model.pkl")
        self.model_path = model_path
        self.encoder_path = os.path.join(
            get_project_root(), "ml", "models", "label_encoder.pkl")
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model = joblib.load(self.model_path)
                self.le = joblib.load(self.encoder_path)
                return
            except Exception:
                pass

        # Fallback dummy model if files are missing
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame({
            'panel_temp_c': [25, 45, 30, 25, 35], 'irradiance': [500, 800, 600, 0, 900],
            'power_kw': [20, 5, 25, 0, 10], 'voltage_v': [230, 190, 231, 230, 220],
            'resistance': [0.05, 2.5, 0.06, 0.05, 1.8], 'ping_ms': [15, 15, 18, 2000, 15]
        })
        y = ["Normal", "Wiring_Fault", "Grid_Voltage_Fluctuation",
             "Monitoring_Offline", "Loose_Corroded_Wiring"]
        self.le = LabelEncoder()
        self.model.fit(X, self.le.fit_transform(y))

    def predict(self, data: dict) -> dict:
        features = pd.DataFrame([{
            'panel_temp_c': data.get('panel_temp_c', 25), 'irradiance': data.get('irradiance_wm2', 500),
            'power_kw': data.get('power_output_kw', 20), 'voltage_v': data.get('voltage_l1_v', 230),
            'resistance': data.get('resistance_ohm', 0.05), 'ping_ms': data.get('ping_ms', 15)
        }])
        pred_idx = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        fault = self.le.inverse_transform([pred_idx])[0]
        return {"fault_type": fault, "confidence": float(max(proba)), "all_probabilities": dict(zip(self.le.classes_, proba))}

