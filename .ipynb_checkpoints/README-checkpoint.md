# 🏛️ Smart Campus Digital Twin

An AI-powered digital twin platform for university campuses — predicts room occupancy, recommends study spaces, forecasts crowd movement, optimizes energy use, and detects infrastructure anomalies in real time.

## ✨ Features

| Module | Description | AI Technique |
|---|---|---|
| 🏠 **Live Dashboard** | Campus-wide KPIs, live room status, alerts | Aggregation |
| 📡 **Real-Time Monitor** | Per-building environmental sensors (temp, CO₂, humidity, noise) | Streaming analytics |
| 🔮 **Occupancy Forecast** | 6–48 hour occupancy predictions per building | Random Forest time-series regression |
| 📚 **Study Space Finder** | Personalized room recommendations (quiet/group/focus/quick) | Content-based recommender |
| 🌊 **Crowd Flow Simulator** | Predicts how crowds move between buildings over time | Markov transition model |
| ⚡ **Energy Optimizer** | Actionable energy-saving recommendations + CO₂ savings | Rule-based + occupancy correlation |
| 🚨 **Anomaly Detection** | Flags abnormal sensor readings (spikes, overcrowding) | Isolation Forest |
| 🏗️ **Infrastructure Health** | Building health scores, maintenance scheduling | Composite scoring |
| 📅 **Events & Impact** | Upcoming events and predicted attendance impact | — |
| 📊 **Analytics & Reports** | Occupancy/energy/sensor trends, underutilization detection | Statistical analysis |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📁 Project Structure

```
smart_campus_twin/
├── app.py                  # Main Streamlit app (all pages)
├── requirements.txt
├── config/
│   └── settings.py         # Thresholds, campus config
├── data/
│   └── generator.py         # Synthetic IoT sensor data generator
├── models/
│   └── ml_models.py          # Forecaster, recommender, anomaly detector, etc.
└── utils/
    └── charts.py             # Plotly chart builders + theme
```

## 🧠 AI Components

1. **Occupancy Forecaster** — Random Forest Regressor with cyclic time features (hour/day sin-cos encoding) trained per building on 30 days of synthetic IoT data.
2. **Study Space Recommender** — Weighted scoring engine combining noise level, available seats, amenities, and comfort score across 4 study-style profiles.
3. **Anomaly Detector** — Isolation Forest per building over occupancy, energy, temperature, humidity, CO₂, and noise — flags overcrowding, energy spikes, and air-quality issues.
4. **Crowd Flow Predictor** — Markov-chain transition model simulating people movement between 10 campus buildings over multiple time steps.
5. **Energy Optimizer** — Correlates occupancy with energy draw to recommend standby/dimming/HVAC actions and estimates kWh + CO₂ savings.

## 🔧 Customization

- Edit `data/generator.py` → `BUILDINGS` / `ROOMS` dicts to model your real campus layout.
- Replace synthetic generators with real IoT feeds (same DataFrame schema) for production use.
- Adjust thresholds in `config/settings.py`.

## 📊 Data Note

All data is **synthetically generated** with realistic daily/weekly patterns (class hours, meal times, hostel occupancy, weekend effects) so the dashboard is fully functional out of the box — swap in live sensor feeds for deployment.
