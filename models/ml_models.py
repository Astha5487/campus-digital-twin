"""
ML Models for Smart Campus Digital Twin
- Time-series forecasting (occupancy prediction)
- Study space recommendation engine
- Anomaly detection
- Crowd flow forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta


# ─── Occupancy Forecaster ─────────────────────────────────────────────────────

class OccupancyForecaster:
    """Random-Forest based short-term occupancy forecaster."""

    def __init__(self):
        self.models = {}
        self.scalers = {}

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    FEATURE_COLS = [
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    def train(self, df: pd.DataFrame):
        df = self._make_features(df)
        for building in df["building"].unique():
            bdf = df[df["building"] == building].dropna(subset=["occupancy_rate"])
            if len(bdf) < 50:
                continue
            X = bdf[self.FEATURE_COLS]
            y = bdf["occupancy_rate"]
            sc = StandardScaler()
            X_scaled = sc.fit_transform(X)
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_scaled, y)
            self.models[building] = model
            self.scalers[building] = sc

    def predict_next_hours(self, building: str, hours: int = 24) -> pd.DataFrame:
        if building not in self.models:
            return pd.DataFrame()

        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_times = [now + timedelta(hours=i) for i in range(1, hours + 1)]
        rows = []
        for t in future_times:
            rows.append({
                "timestamp": t,
                "hour": t.hour,
                "day_of_week": t.weekday(),
                "month": t.month,
                "is_weekend": int(t.weekday() >= 5),
                "hour_sin": np.sin(2 * np.pi * t.hour / 24),
                "hour_cos": np.cos(2 * np.pi * t.hour / 24),
                "dow_sin": np.sin(2 * np.pi * t.weekday() / 7),
                "dow_cos": np.cos(2 * np.pi * t.weekday() / 7),
            })
        future_df = pd.DataFrame(rows)
        X = future_df[self.FEATURE_COLS]
        X_scaled = self.scalers[building].transform(X)
        preds = self.models[building].predict(X_scaled)
        future_df["predicted_occupancy"] = np.clip(preds, 0, 1)
        future_df["building"] = building
        return future_df[["timestamp", "building", "predicted_occupancy"]]

    def predict_all_buildings(self, hours: int = 12) -> pd.DataFrame:
        frames = []
        for b in self.models:
            frames.append(self.predict_next_hours(b, hours))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─── Study Space Recommender ──────────────────────────────────────────────────

class StudySpaceRecommender:
    """Content-based + collaborative filtering recommender for study spaces."""

    PREFERENCE_WEIGHTS = {
        "quiet": {"noise_low": 2.0, "capacity": 0.3, "features_whiteboard": 0.5},
        "group_study": {"capacity": 2.0, "noise_medium": 1.0, "features_projector": 1.5},
        "focused_work": {"noise_low": 2.5, "features_ac": 1.0, "occupancy_low": 2.0},
        "quick_access": {"occupancy_low": 2.5, "capacity": 0.5},
    }

    def recommend(
        self,
        rooms_df: pd.DataFrame,
        preference: str = "quiet",
        top_n: int = 5,
        min_available_seats: int = 1,
    ) -> pd.DataFrame:
        df = rooms_df.copy()
        df = df[df["available"] == True]
        df = df[(df["capacity"] - df["occupied"]) >= min_available_seats]

        if df.empty:
            return pd.DataFrame()

        # Feature engineering
        df["available_seats"] = df["capacity"] - df["occupied"]
        df["noise_low"] = (df["noise_level"] == "Low").astype(float)
        df["noise_medium"] = (df["noise_level"] == "Medium").astype(float)
        df["occupancy_low"] = 1 - df["occupancy_rate"]
        df["features_projector"] = df["features"].str.contains("Projector").astype(float)
        df["features_whiteboard"] = df["features"].str.contains("Whiteboard").astype(float)
        df["features_ac"] = df["features"].str.contains("AC").astype(float)

        weights = self.PREFERENCE_WEIGHTS.get(preference, self.PREFERENCE_WEIGHTS["quiet"])
        df["score"] = 0.0
        for feature, weight in weights.items():
            if feature in df.columns:
                df["score"] += df[feature] * weight
        df["score"] = df["score"] / df["score"].max()

        # Blend with comfort score
        df["final_score"] = 0.6 * df["score"] + 0.4 * df["comfort_score"]
        return df.sort_values("final_score", ascending=False).head(top_n)[
            ["building", "room", "capacity", "occupied", "available_seats",
             "noise_level", "features", "comfort_score", "final_score"]
        ].reset_index(drop=True)


# ─── Anomaly Detector ─────────────────────────────────────────────────────────

class CampusAnomalyDetector:
    """Isolation Forest based anomaly detection for sensor readings."""

    FEATURE_COLS = ["occupancy_rate", "energy_kwh", "temperature_c", "humidity_pct", "co2_ppm", "noise_db"]

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}

    def train(self, df: pd.DataFrame):
        for building in df["building"].unique():
            bdf = df[df["building"] == building].dropna(subset=self.FEATURE_COLS)
            if len(bdf) < 30:
                continue
            X = bdf[self.FEATURE_COLS]
            sc = StandardScaler()
            X_scaled = sc.fit_transform(X)
            iso = IsolationForest(contamination=0.05, random_state=42)
            iso.fit(X_scaled)
            self.models[building] = iso
            self.scalers[building] = sc

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for building in df["building"].unique():
            if building not in self.models:
                continue
            bdf = df[df["building"] == building].copy()
            valid = bdf.dropna(subset=self.FEATURE_COLS)
            if valid.empty:
                continue
            X = valid[self.FEATURE_COLS]
            X_scaled = self.scalers[building].transform(X)
            scores = self.models[building].decision_function(X_scaled)
            preds = self.models[building].predict(X_scaled)
            valid = valid.copy()
            valid["anomaly_score"] = -scores  # Higher = more anomalous
            valid["is_anomaly"] = (preds == -1)
            results.append(valid)

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def get_anomaly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        detected = self.detect(df)
        if detected.empty or "is_anomaly" not in detected.columns:
            return pd.DataFrame()
        anomalies = detected[detected["is_anomaly"] == True]
        if anomalies.empty:
            return pd.DataFrame()

        summary = []
        for _, row in anomalies.iterrows():
            reasons = []
            if row.get("occupancy_rate", 0) > 0.95:
                reasons.append("Over-capacity")
            if row.get("co2_ppm", 0) > 1200:
                reasons.append("High CO₂")
            if row.get("energy_kwh", 0) > 50:
                reasons.append("Energy spike")
            if row.get("temperature_c", 20) > 30:
                reasons.append("High temperature")
            if not reasons:
                reasons.append("Statistical outlier")

            summary.append({
                "timestamp": row.get("timestamp"),
                "building": row.get("building"),
                "anomaly_score": round(row.get("anomaly_score", 0), 3),
                "reason": ", ".join(reasons),
                "occupancy_rate": row.get("occupancy_rate"),
                "energy_kwh": row.get("energy_kwh"),
                "co2_ppm": row.get("co2_ppm"),
            })
        return pd.DataFrame(summary).sort_values("anomaly_score", ascending=False)


# ─── Crowd Flow Predictor ─────────────────────────────────────────────────────

class CrowdFlowPredictor:
    """Predict crowd movement hotspots across campus."""

    TRANSITION_MATRIX = {
        "Library": {"Student Center": 0.3, "Cafeteria": 0.4, "Engineering Block A": 0.2, "Admin Block": 0.1},
        "Engineering Block A": {"Library": 0.3, "Cafeteria": 0.4, "Student Center": 0.2, "Engineering Block B": 0.1},
        "Engineering Block B": {"Library": 0.25, "Cafeteria": 0.45, "Engineering Block A": 0.2, "Science Complex": 0.1},
        "Science Complex": {"Library": 0.3, "Cafeteria": 0.35, "Engineering Block A": 0.25, "Admin Block": 0.1},
        "Student Center": {"Cafeteria": 0.5, "Library": 0.2, "Sports Complex": 0.2, "Admin Block": 0.1},
        "Cafeteria": {"Library": 0.3, "Student Center": 0.3, "Engineering Block A": 0.2, "Science Complex": 0.2},
        "Admin Block": {"Library": 0.3, "Cafeteria": 0.3, "Student Center": 0.3, "Engineering Block A": 0.1},
        "Sports Complex": {"Student Center": 0.5, "Cafeteria": 0.3, "Hostel Block A": 0.2},
        "Hostel Block A": {"Cafeteria": 0.4, "Library": 0.3, "Student Center": 0.2, "Sports Complex": 0.1},
        "Hostel Block B": {"Cafeteria": 0.4, "Library": 0.3, "Student Center": 0.2, "Sports Complex": 0.1},
    }

    def predict_flow(self, current_occupancies: dict, steps: int = 3) -> list:
        """Simulate crowd movement over N time steps."""
        populations = {b: occ for b, occ in current_occupancies.items()}
        snapshots = [dict(populations)]

        for _ in range(steps):
            next_pop = {b: 0.0 for b in populations}
            for building, pop in populations.items():
                transitions = self.TRANSITION_MATRIX.get(building, {})
                stay_fraction = 0.6
                next_pop[building] = next_pop.get(building, 0) + pop * stay_fraction
                total_leave = pop * (1 - stay_fraction)
                if transitions:
                    total_prob = sum(transitions.values())
                    for dest, prob in transitions.items():
                        if dest in next_pop:
                            next_pop[dest] += total_leave * (prob / total_prob)
            populations = next_pop
            snapshots.append(dict(populations))

        return snapshots


# ─── Energy Optimizer ─────────────────────────────────────────────────────────

class EnergyOptimizer:
    """Recommend energy saving actions based on occupancy vs energy usage."""

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        latest = df.sort_values("timestamp").groupby("building").last().reset_index()
        recommendations = []
        for _, row in latest.iterrows():
            occ = row.get("occupancy_rate", 0.5)
            energy = row.get("energy_kwh", 0)
            temp = row.get("temperature_c", 24)

            actions = []
            savings_kwh = 0.0
            priority = "Low"

            if occ < 0.15 and energy > 5:
                actions.append("Switch to standby mode — very low occupancy")
                savings_kwh += energy * 0.4
                priority = "High"
            if occ < 0.3 and temp < 24:
                actions.append("Raise AC setpoint by 2°C (unused cooling)")
                savings_kwh += 1.5
            if occ < 0.5 and energy > 10:
                actions.append("Dim non-essential lighting to 60%")
                savings_kwh += energy * 0.1
                priority = "Medium" if priority == "Low" else priority
            if not actions:
                actions.append("No immediate action required")

            recommendations.append({
                "building": row["building"],
                "current_occupancy": round(occ, 2),
                "current_energy_kwh": round(energy, 2),
                "action": "; ".join(actions),
                "estimated_savings_kwh": round(savings_kwh, 2),
                "priority": priority,
            })

        return pd.DataFrame(recommendations).sort_values(
            "estimated_savings_kwh", ascending=False
        )


# ─── Model Training Pipeline ──────────────────────────────────────────────────

def train_all_models(historical_df: pd.DataFrame):
    print("Training occupancy forecaster...")
    forecaster = OccupancyForecaster()
    forecaster.train(historical_df)

    print("Training anomaly detector...")
    anomaly_detector = CampusAnomalyDetector()
    anomaly_detector.train(historical_df)

    return forecaster, anomaly_detector
