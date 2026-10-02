# -*- coding: utf-8 -*-
"""
AI/ML layer for Smart Solar AI:
  - forecast_next_month(): consumption forecasting (scikit-learn LinearRegression)
  - detect_anomalies(): anomaly detection on daily consumption (IsolationForest)
  - predict_hourly_generation(): predicted solar generation curve.
    Uses a tiny TensorFlow/Keras model when TensorFlow is available in the
    environment; falls back to a physics-based irradiance model otherwise,
    so the app never hard-fails just because TF isn't installed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

try:
    import tensorflow as tf  # noqa: F401
    _HAS_TF = True
except Exception:
    _HAS_TF = False


def forecast_next_month(df: pd.DataFrame, months_ahead: int = 1) -> float:
    """
    df must have columns: date (datetime-like), consumption_kwh (daily).
    Aggregates to monthly totals and fits a simple linear trend + seasonal
    average to project the next month's usage.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M")
    monthly = d.groupby("month")["consumption_kwh"].sum().reset_index()
    monthly["idx"] = np.arange(len(monthly))

    X = monthly[["idx"]].values
    y = monthly["consumption_kwh"].values
    if len(monthly) < 2:
        return float(y[-1]) if len(y) else 0.0

    model = LinearRegression().fit(X, y)
    next_idx = np.array([[monthly["idx"].max() + months_ahead]])
    pred = float(model.predict(next_idx)[0])
    return round(max(pred, 0.0), 1)


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.06) -> list[dict]:
    """
    Runs IsolationForest over daily consumption + day-of-year seasonality
    features to flag unusual consumption days. Returns a list of
    {date, severity, note} dicts, most recent first, capped at 5.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) < 10:
        return []

    d["doy_sin"] = np.sin(2 * np.pi * d["date"].dt.dayofyear / 365.0)
    d["doy_cos"] = np.cos(2 * np.pi * d["date"].dt.dayofyear / 365.0)
    features = d[["consumption_kwh", "doy_sin", "doy_cos"]].values

    clf = IsolationForest(contamination=contamination, random_state=42)
    d["anomaly"] = clf.fit_predict(features)
    d["score"] = clf.decision_function(features)

    anomalies = d[d["anomaly"] == -1].sort_values("score")
    results = []
    for _, row in anomalies.head(5).iterrows():
        severity = "High" if row["score"] < -0.15 else ("Medium" if row["score"] < -0.05 else "Low")
        direction = "spike" if row["consumption_kwh"] > d["consumption_kwh"].mean() else "dip"
        results.append({
            "date": row["date"].strftime("%b %d"),
            "severity": severity,
            "note": f"Unusual consumption {direction} vs. seasonal pattern",
        })
    return results


def _fallback_hourly_curve(sun_hours: float) -> np.ndarray:
    """Physics-inspired half-sine irradiance curve scaled by peak sun hours."""
    hours = np.arange(0, 24)
    # Sun window widens/narrows with sun_hours; peak at solar noon (hour 13).
    window = np.clip(sun_hours * 1.6, 6, 14)
    start = 13 - window / 2
    end = 13 + window / 2
    curve = np.zeros_like(hours, dtype=float)
    mask = (hours >= start) & (hours <= end)
    curve[mask] = np.sin(np.pi * (hours[mask] - start) / (end - start))
    peak_kw = sun_hours / 5.0  # normalized peak output factor
    return curve * peak_kw


def predict_hourly_generation(sun_hours: float, solar_kw: float = 1.0) -> pd.DataFrame:
    """
    Returns a DataFrame with columns hour, predicted_kwh for a representative day.
    Tries a tiny Keras model trained on synthetic irradiance data when
    TensorFlow is present; otherwise uses the physics-based fallback.
    """
    hours = np.arange(0, 24)

    if _HAS_TF:
        try:
            rng = np.random.default_rng(42)
            n = 400
            train_hours = rng.uniform(0, 24, n)
            train_sun = rng.uniform(3, 7, n)
            base = _fallback_hourly_curve_vec(train_hours, train_sun)
            noise = rng.normal(0, 0.03, n)
            X = np.column_stack([train_hours / 24.0, train_sun / 7.0])
            y = base + noise

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation="relu", input_shape=(2,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=15, verbose=0)

            Xp = np.column_stack([hours / 24.0, np.full(24, sun_hours) / 7.0])
            curve = model.predict(Xp, verbose=0).flatten()
            curve = np.clip(curve, 0, None)
        except Exception:
            curve = _fallback_hourly_curve(sun_hours)
    else:
        curve = _fallback_hourly_curve(sun_hours)

    total = curve.sum() or 1.0
    scaled = curve / total * (solar_kw * sun_hours)  # normalize to expected daily kWh
    return pd.DataFrame({"hour": hours, "predicted_kwh": np.round(scaled, 3)})


def _fallback_hourly_curve_vec(hours: np.ndarray, sun_hours: np.ndarray) -> np.ndarray:
    """Vectorized version of the fallback curve, used to build TF training data."""
    window = np.clip(sun_hours * 1.6, 6, 14)
    start = 13 - window / 2
    end = 13 + window / 2
    out = np.zeros_like(hours, dtype=float)
    mask = (hours >= start) & (hours <= end)
    frac = np.zeros_like(hours, dtype=float)
    frac[mask] = (hours[mask] - start[mask]) / (end[mask] - start[mask])
    out[mask] = np.sin(np.pi * frac[mask])
    peak_kw = sun_hours / 5.0
    return out * peak_kw
