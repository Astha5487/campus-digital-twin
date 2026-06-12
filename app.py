"""
🏛️ Smart Campus Digital Twin
Main application entry point
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

from config.settings import PAGE_CONFIG, CAMPUS_NAME, AUTO_REFRESH_SECONDS
from data.generator import (
    generate_historical_data, generate_room_data,
    generate_events, generate_infrastructure_health,
    BUILDINGS, CAMPUS_COORDS,
)
from models.ml_models import (
    OccupancyForecaster, StudySpaceRecommender,
    CampusAnomalyDetector, CrowdFlowPredictor, EnergyOptimizer,
    train_all_models,
)
from utils.charts import (
    COLORS, occupancy_badge, format_pct, get_status_color,
    building_bar_chart, occupancy_heatmap, energy_timeline,
    forecast_chart, environmental_radar, crowd_flow_sankey,
    anomaly_scatter,
)

st.set_page_config(**PAGE_CONFIG)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0E1117; }

    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #141824 100%);
        border: 1px solid #2d3546;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(27,79,219,0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    .metric-label { font-size: 0.8rem; color: #8892A4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .metric-delta { font-size: 0.85rem; margin-top: 6px; }

    .alert-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .alert-high { background: rgba(231,76,60,0.12); border-color: #E74C3C; }
    .alert-medium { background: rgba(255,193,7,0.12); border-color: #FFC107; }
    .alert-low { background: rgba(46,204,113,0.12); border-color: #2ECC71; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FAFAFA;
        padding: 12px 0 8px;
        border-bottom: 1px solid #2d3546;
        margin-bottom: 16px;
    }
    .tag {
        display: inline-block;
        background: rgba(27,79,219,0.2);
        border: 1px solid rgba(27,79,219,0.4);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #7BA7FF;
        margin: 2px;
    }
    .sidebar-logo {
        text-align: center;
        padding: 12px 0 24px;
        border-bottom: 1px solid #2d3546;
        margin-bottom: 16px;
    }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #2ECC71;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    .stSelectbox > div > div { background-color: #1A1F2E; border: 1px solid #2d3546; }
    .stSlider > div { color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    .room-card {
        background: #1A1F2E;
        border: 1px solid #2d3546;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
    }
    .progress-bar-bg {
        background: #2d3546;
        border-radius: 4px;
        height: 6px;
        margin-top: 6px;
    }
    h1, h2, h3 { color: #FAFAFA !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data + Model Cache ─────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner="⚙️ Loading historical sensor data...")
def load_historical():
    return generate_historical_data(days=30)


@st.cache_data(ttl=30, show_spinner=False)
def load_live():
    return generate_room_data()


@st.cache_data(ttl=300, show_spinner=False)
def load_events():
    return generate_events()


@st.cache_data(ttl=600, show_spinner=False)
def load_infra():
    return generate_infrastructure_health()


@st.cache_resource(show_spinner="🤖 Training AI models...")
def get_models():
    hist = generate_historical_data(days=30)
    forecaster, anomaly_det = train_all_models(hist)
    recommender = StudySpaceRecommender()
    crowd_pred = CrowdFlowPredictor()
    energy_opt = EnergyOptimizer()
    return forecaster, anomaly_det, recommender, crowd_pred, energy_opt


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem">🏛️</div>
        <div style="font-size:1.1rem;font-weight:700;color:#FAFAFA;margin-top:6px">Smart Campus</div>
        <div style="font-size:0.75rem;color:#8892A4">Digital Twin Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        [
            "🏠 Live Dashboard",
            "📡 Real-Time Monitor",
            "🔮 Occupancy Forecast",
            "📚 Study Space Finder",
            "🌊 Crowd Flow Simulator",
            "⚡ Energy Optimizer",
            "🚨 Anomaly Detection",
            "🏗️ Infrastructure Health",
            "📅 Events & Impact",
            "📊 Analytics & Reports",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<span class="live-dot"></span><span style="font-size:0.8rem;color:#2ECC71">Live Data Connected</span>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.75rem;color:#8892A4;margin-top:6px">Last sync: {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#8892A4">QUICK STATS</div>', unsafe_allow_html=True)
    hist_data = load_historical()
    live_rooms = load_live()
    avg_occ = live_rooms["occupancy_rate"].mean()
    available_rooms = live_rooms["available"].sum()
    st.metric("Campus Avg Occupancy", f"{avg_occ:.0%}")
    st.metric("Available Rooms", f"{available_rooms}")

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(AUTO_REFRESH_SECONDS)
        st.rerun()


# ── Load everything ────────────────────────────────────────────────────────────
historical = load_historical()
rooms = load_live()
events = load_events()
infra = load_infra()
forecaster, anomaly_det, recommender, crowd_pred, energy_opt = get_models()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Live Dashboard":
    st.markdown(f"# 🏛️ {CAMPUS_NAME}")
    st.markdown(f'<span class="live-dot"></span><span style="color:#8892A4;font-size:0.9rem">Live campus intelligence • {datetime.now().strftime("%A, %d %b %Y %H:%M")}</span>', unsafe_allow_html=True)
    st.markdown("")

    # KPI Row
    total_cap = sum(b["capacity"] for b in BUILDINGS.values())
    total_occupied = int(rooms["occupancy_rate"].mean() * total_cap)
    alerts_count = int(historical["alert"].notna().tail(100).sum())
    anomalies_recent = 3
    energy_today = historical.tail(240)["energy_kwh"].sum()

    kpi_cols = st.columns(5)
    kpis = [
        ("🏫", f"{len(BUILDINGS)}", "Buildings Live", COLORS["primary"]),
        ("👥", f"{total_occupied:,}", "People On Campus", COLORS["secondary"]),
        ("📊", f"{avg_occ:.0%}", "Avg Occupancy", get_status_color(avg_occ)),
        ("⚡", f"{energy_today:.0f} kWh", "Energy Today", COLORS["accent"]),
        ("🚨", f"{alerts_count}", "Alerts (24h)", COLORS["warning"]),
    ]
    for col, (icon, val, label, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(building_bar_chart(historical), use_container_width=True)
    with col2:
        recent_24h = historical[pd.to_datetime(historical["timestamp"]) >= pd.Timestamp.now() - pd.Timedelta(hours=48)]
        st.plotly_chart(occupancy_heatmap(recent_24h), use_container_width=True)

    # Live room status
    st.markdown('<div class="section-header">📍 Live Room Status</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for idx, (_, row) in enumerate(rooms.head(8).iterrows()):
        with cols[idx % 4]:
            occ_color = get_status_color(row["occupancy_rate"])
            bar_w = int(row["occupancy_rate"] * 100)
            st.markdown(f"""
            <div class="room-card">
                <div style="font-weight:600;font-size:0.9rem;color:#FAFAFA">{row['room']}</div>
                <div style="font-size:0.75rem;color:#8892A4;margin-top:2px">{row['building']}</div>
                <div style="color:{occ_color};font-size:1.3rem;font-weight:700;margin:6px 0">{row['occupancy_rate']:.0%}</div>
                <div class="progress-bar-bg">
                    <div style="width:{bar_w}%;height:6px;background:{occ_color};border-radius:4px"></div>
                </div>
                <div style="font-size:0.75rem;color:#8892A4;margin-top:4px">{row['occupied']}/{row['capacity']} seats</div>
            </div>
            """, unsafe_allow_html=True)

    # Alerts strip
    recent_alerts = historical[historical["alert"].notna()].tail(5)
    if not recent_alerts.empty:
        st.markdown('<div class="section-header">🔔 Recent Alerts</div>', unsafe_allow_html=True)
        for _, a in recent_alerts.iterrows():
            alert_class = "alert-high" if a["alert"] == "HIGH_OCCUPANCY" else "alert-medium"
            st.markdown(f"""
            <div class="alert-card {alert_class}">
                <b>{a['building']}</b> — {a['alert'].replace('_', ' ')} &nbsp;
                <span style="color:#8892A4;font-size:0.8rem">{pd.to_datetime(a['timestamp']).strftime('%d %b %H:%M')}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REAL-TIME MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Real-Time Monitor":
    st.markdown("## 📡 Real-Time Campus Monitor")
    selected_building = st.selectbox("Select Building", sorted(BUILDINGS.keys()))

    bdata = historical[historical["building"] == selected_building].sort_values("timestamp")

    col1, col2, col3, col4 = st.columns(4)
    latest = bdata.iloc[-1]
    col1.metric("Temperature", f"{latest['temperature_c']:.1f}°C", delta=f"{latest['temperature_c']-23:.1f}°C from comfort")
    col2.metric("Humidity", f"{latest['humidity_pct']:.0f}%")
    col3.metric("CO₂", f"{latest['co2_ppm']:.0f} ppm", delta="⚠️ High" if latest["co2_ppm"] > 1000 else "✅ Normal")
    col4.metric("Noise", f"{latest['noise_db']:.0f} dB")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(energy_timeline(historical, selected_building), use_container_width=True)
    with col2:
        st.plotly_chart(environmental_radar(historical, selected_building), use_container_width=True)

    # Environmental trends
    st.markdown('<div class="section-header">📈 Environmental Trends (Last 48h)</div>', unsafe_allow_html=True)
    recent_bdata = bdata.tail(48)

    import plotly.graph_objects as go
    fig = go.Figure()
    for col_name, color, yaxis in [
        ("temperature_c", "#FF6B35", "y"),
        ("humidity_pct", "#1B4FDB", "y2"),
        ("co2_ppm", "#00C49A", "y3"),
    ]:
        fig.add_trace(go.Scatter(
            x=recent_bdata["timestamp"], y=recent_bdata[col_name],
            name=col_name.replace("_", " ").title(),
            line=dict(color=color, width=2), yaxis=yaxis,
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Temp °C", gridcolor="#2d3546"),
        yaxis2=dict(title="Humidity %", overlaying="y", side="right"),
        yaxis3=dict(title="CO₂ ppm", overlaying="y", side="right", anchor="free", position=0.85),
        margin=dict(l=10, r=60, t=10, b=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OCCUPANCY FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Occupancy Forecast":
    st.markdown("## 🔮 AI Occupancy Forecast")
    st.markdown("*Random Forest model trained on 30 days of IoT sensor data*")

    col1, col2 = st.columns([2, 1])
    with col1:
        sel_building = st.selectbox("Select Building", sorted(forecaster.models.keys()))
    with col2:
        forecast_hours = st.slider("Forecast horizon (hours)", 6, 48, 24)

    forecast_df = forecaster.predict_next_hours(sel_building, forecast_hours)

    if not forecast_df.empty:
        # Key predictions
        peak_row = forecast_df.loc[forecast_df["predicted_occupancy"].idxmax()]
        low_row = forecast_df.loc[forecast_df["predicted_occupancy"].idxmin()]
        avg_pred = forecast_df["predicted_occupancy"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Peak Predicted", f"{peak_row['predicted_occupancy']:.0%}", f"at {pd.to_datetime(peak_row['timestamp']).strftime('%H:%M')}")
        c2.metric("Lowest Expected", f"{low_row['predicted_occupancy']:.0%}", f"at {pd.to_datetime(low_row['timestamp']).strftime('%H:%M')}")
        c3.metric("Average Forecast", f"{avg_pred:.0%}")

        st.plotly_chart(forecast_chart(forecast_df, sel_building), use_container_width=True)

        # Best time to visit
        good_times = forecast_df[forecast_df["predicted_occupancy"] < 0.4]
        if not good_times.empty:
            st.success(f"✅ Best times to visit (occupancy < 40%): " +
                      ", ".join(pd.to_datetime(good_times["timestamp"]).dt.strftime("%H:%M").tolist()[:5]))

        # Warn periods
        busy_times = forecast_df[forecast_df["predicted_occupancy"] > 0.8]
        if not busy_times.empty:
            st.warning(f"⚠️ Expect crowds (> 80%) at: " +
                      ", ".join(pd.to_datetime(busy_times["timestamp"]).dt.strftime("%H:%M").tolist()[:5]))

        with st.expander("📊 View Forecast Data Table"):
            display_df = forecast_df.copy()
            display_df["predicted_occupancy"] = display_df["predicted_occupancy"].apply(format_pct)
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%A %d %b, %H:%M")
            st.dataframe(display_df, use_container_width=True)

    # Multi-building forecast comparison
    st.markdown('<div class="section-header">🏫 All-Buildings 6h Forecast</div>', unsafe_allow_html=True)
    all_forecast = forecaster.predict_all_buildings(hours=6)
    if not all_forecast.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        for bld in all_forecast["building"].unique():
            bdf = all_forecast[all_forecast["building"] == bld]
            fig.add_trace(go.Scatter(
                x=bdf["timestamp"], y=bdf["predicted_occupancy"] * 100,
                name=bld, mode="lines",
            ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=-0.3),
            yaxis=dict(title="Predicted Occupancy %", gridcolor="#2d3546"),
            xaxis=dict(gridcolor="#2d3546"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDY SPACE FINDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Study Space Finder":
    st.markdown("## 📚 AI Study Space Finder")
    st.markdown("Get personalized room recommendations based on your study preferences.")

    col1, col2, col3 = st.columns(3)
    with col1:
        preference = st.selectbox("Study Style", {
            "quiet": "🤫 Quiet Solo Study",
            "group_study": "👥 Group Study",
            "focused_work": "🎯 Deep Focused Work",
            "quick_access": "⚡ Quick Access",
        })
    with col2:
        min_seats = st.number_input("Min available seats", 1, 50, 1)
    with col3:
        top_n = st.slider("How many options", 3, 10, 5)

    pref_map = {
        "🤫 Quiet Solo Study": "quiet",
        "👥 Group Study": "group_study",
        "🎯 Deep Focused Work": "focused_work",
        "⚡ Quick Access": "quick_access",
    }

    if st.button("🔍 Find Best Spaces", type="primary"):
        recs = recommender.recommend(rooms, pref_map.get(preference, "quiet"), top_n, min_seats)
        if recs.empty:
            st.warning("No rooms match your criteria right now. Try reducing seat requirements.")
        else:
            st.markdown(f"### 🎯 Top {len(recs)} Recommendations")
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                avail_pct = 1 - (row["occupied"] / row["capacity"])
                bar_color = get_status_color(1 - avail_pct)
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"][rank - 1]
                st.markdown(f"""
                <div class="room-card" style="border-left: 3px solid {COLORS['primary']}; margin-bottom: 10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="font-size:1.2rem">{medal}</span>
                            <b style="font-size:1rem;color:#FAFAFA;margin-left:8px">{row['room']}</b>
                            <span style="color:#8892A4;font-size:0.85rem"> — {row['building']}</span>
                        </div>
                        <div style="color:{COLORS['secondary']};font-weight:700;font-family:monospace">
                            Score: {row['final_score']:.2f}
                        </div>
                    </div>
                    <div style="margin-top:8px;display:flex;gap:12px;flex-wrap:wrap">
                        <span class="tag">💺 {row['available_seats']} seats free</span>
                        <span class="tag">🔊 {row['noise_level']} noise</span>
                        <span class="tag">✨ {row['features']}</span>
                        <span class="tag">😊 Comfort: {row['comfort_score']:.0%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # All rooms table
    st.markdown('<div class="section-header">📋 All Rooms — Live Status</div>', unsafe_allow_html=True)
    filter_building = st.multiselect("Filter by building", sorted(rooms["building"].unique()), default=[])
    display_rooms = rooms if not filter_building else rooms[rooms["building"].isin(filter_building)]
    display_rooms = display_rooms.copy()
    display_rooms["Status"] = display_rooms["occupancy_rate"].apply(occupancy_badge)
    display_rooms["Available Seats"] = display_rooms["capacity"] - display_rooms["occupied"]
    cols_show = ["building", "room", "capacity", "Available Seats", "Status", "noise_level", "features", "comfort_score"]
    st.dataframe(
        display_rooms[cols_show].rename(columns={"building": "Building", "room": "Room",
            "noise_level": "Noise", "features": "Features", "comfort_score": "Comfort"}),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CROWD FLOW SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌊 Crowd Flow Simulator":
    st.markdown("## 🌊 Crowd Flow Simulator")
    st.markdown("Simulate and forecast crowd movement across campus buildings.")

    latest_occ = historical.sort_values("timestamp").groupby("building").last().reset_index()
    current = {row["building"]: row["occupied"] for _, row in latest_occ.iterrows()}

    col1, col2 = st.columns([2, 1])
    with col1:
        steps = st.slider("Simulation steps (hours ahead)", 1, 6, 3)
    with col2:
        show_sankey = st.checkbox("Show flow diagram", value=True)

    snapshots = crowd_pred.predict_flow(current, steps)

    if show_sankey and len(snapshots) >= 2:
        st.plotly_chart(crowd_flow_sankey(snapshots[0], snapshots[1]), use_container_width=True)

    # Step-by-step heatmap
    st.markdown('<div class="section-header">📊 Predicted Crowd Distribution</div>', unsafe_allow_html=True)
    snapshot_df = pd.DataFrame(snapshots, index=[f"Now" if i == 0 else f"+{i}h" for i in range(len(snapshots))])
    snapshot_df = snapshot_df.round(0).astype(int)

    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=snapshot_df.values,
        x=snapshot_df.columns.tolist(),
        y=snapshot_df.index.tolist(),
        colorscale="Blues",
        hoverongaps=False,
        colorbar=dict(title="People"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis_tickangle=-45,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hotspot warnings
    st.markdown('<div class="section-header">🔥 Predicted Hotspots</div>', unsafe_allow_html=True)
    final_snap = snapshots[-1]
    sorted_buildings = sorted(final_snap.items(), key=lambda x: x[1], reverse=True)
    cols = st.columns(3)
    for idx, (bld, pop) in enumerate(sorted_buildings[:6]):
        cap = BUILDINGS[bld]["capacity"]
        rate = pop / cap
        with cols[idx % 3]:
            color = get_status_color(rate)
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {color}">
                <div style="font-weight:600;color:#FAFAFA;font-size:0.9rem">{bld}</div>
                <div class="metric-value" style="color:{color}">{rate:.0%}</div>
                <div class="metric-label">{int(pop):,} / {cap:,} people</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ENERGY OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Energy Optimizer":
    st.markdown("## ⚡ AI Energy Optimizer")
    st.markdown("AI-driven energy efficiency recommendations based on real-time occupancy data.")

    recs = energy_opt.analyze(historical)

    total_savings = recs["estimated_savings_kwh"].sum()
    high_prio = len(recs[recs["priority"] == "High"])
    st.info(f"💡 **{total_savings:.1f} kWh** potential savings identified across **{high_prio}** high-priority buildings")

    # Priority breakdown
    c1, c2, c3 = st.columns(3)
    for col, prio, color in [(c1, "High", COLORS["danger"]), (c2, "Medium", COLORS["warning"]), (c3, "Low", COLORS["success"])]:
        cnt = len(recs[recs["priority"] == prio])
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{cnt}</div>
            <div class="metric-label">{prio} Priority</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in recs.iterrows():
        prio_class = {"High": "alert-high", "Medium": "alert-medium", "Low": "alert-low"}.get(row["priority"], "alert-low")
        savings_co2 = row["estimated_savings_kwh"] * 0.82  # kg CO₂ per kWh
        st.markdown(f"""
        <div class="alert-card {prio_class}">
            <div style="display:flex;justify-content:space-between">
                <div>
                    <b>{row['building']}</b>
                    <span style="margin-left:8px;font-size:0.8rem;color:#8892A4">
                        Occupancy: {row['current_occupancy']:.0%} | Energy: {row['current_energy_kwh']} kWh
                    </span>
                </div>
                <div style="text-align:right">
                    <span style="color:{COLORS['secondary']};font-weight:600">
                        💾 {row['estimated_savings_kwh']:.1f} kWh saved
                    </span>
                    <span style="color:#8892A4;font-size:0.8rem;margin-left:8px">
                        🌱 {savings_co2:.1f} kg CO₂
                    </span>
                </div>
            </div>
            <div style="margin-top:6px;font-size:0.9rem">💡 {row['action']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Energy trend chart
    st.markdown('<div class="section-header">⚡ Campus Energy Trends</div>', unsafe_allow_html=True)
    energy_trend = historical.groupby(pd.to_datetime(historical["timestamp"]).dt.date)["energy_kwh"].sum().reset_index()
    energy_trend.columns = ["date", "total_kwh"]
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=energy_trend["date"], y=energy_trend["total_kwh"],
        marker_color=COLORS["accent"], opacity=0.8,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        yaxis=dict(title="Total kWh", gridcolor="#2d3546"),
        xaxis=dict(gridcolor="#2d3546"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.markdown("## 🚨 Anomaly Detection")
    st.markdown("*Isolation Forest model monitors sensor readings for unusual patterns*")

    sample = historical.tail(2000)
    anomaly_results = anomaly_det.detect(sample)

    if not anomaly_results.empty:
        total_anomalies = anomaly_results["is_anomaly"].sum()
        anomaly_rate = total_anomalies / len(anomaly_results)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Anomalies Detected", f"{total_anomalies}")
        c2.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
        c3.metric("Buildings Monitored", f"{anomaly_results['building'].nunique()}")

        st.plotly_chart(anomaly_scatter(anomaly_results), use_container_width=True)

        # Anomaly summary
        summary = anomaly_det.get_anomaly_summary(sample)
        if not summary.empty:
            st.markdown('<div class="section-header">📋 Anomaly Details</div>', unsafe_allow_html=True)
            for _, row in summary.head(10).iterrows():
                st.markdown(f"""
                <div class="alert-card alert-high">
                    <div style="display:flex;justify-content:space-between">
                        <b>{row['building']}</b>
                        <span style="color:#E74C3C;font-family:monospace">Score: {row['anomaly_score']:.3f}</span>
                    </div>
                    <div style="font-size:0.85rem;color:#8892A4;margin-top:4px">{row['reason']}</div>
                    <div style="font-size:0.8rem;color:#8892A4;margin-top:2px">
                        Occupancy: {row.get('occupancy_rate', 0):.0%} |
                        Energy: {row.get('energy_kwh', 0):.1f} kWh |
                        CO₂: {row.get('co2_ppm', 0):.0f} ppm
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Per-building anomaly rates
        st.markdown('<div class="section-header">🏫 Anomaly Rates by Building</div>', unsafe_allow_html=True)
        bld_anomalies = anomaly_results.groupby("building").agg(
            total=("is_anomaly", "count"),
            anomalies=("is_anomaly", "sum"),
        ).reset_index()
        bld_anomalies["rate"] = bld_anomalies["anomalies"] / bld_anomalies["total"]
        bld_anomalies = bld_anomalies.sort_values("rate", ascending=False)
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=bld_anomalies["building"], y=bld_anomalies["rate"] * 100,
            marker_color=[get_status_color(r) for r in bld_anomalies["rate"]],
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            yaxis=dict(title="Anomaly Rate %", gridcolor="#2d3546"),
            xaxis=dict(tickangle=-30, gridcolor="#2d3546"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INFRASTRUCTURE HEALTH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Infrastructure Health":
    st.markdown("## 🏗️ Infrastructure Health Monitor")

    avg_health = infra["health_score"].mean()
    maint_due = infra["maintenance_due"].sum()
    critical = len(infra[infra["health_score"] < 0.4])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campus Health Score", f"{avg_health:.0%}")
    c2.metric("Maintenance Due", f"{maint_due}")
    c3.metric("Critical Issues", f"{critical}")
    c4.metric("Avg Building Age", f"{infra['age_years'].mean():.0f} yrs")

    # Health score chart
    infra_sorted = infra.sort_values("health_score")
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=infra_sorted["health_score"] * 100,
        y=infra_sorted["building"],
        orientation="h",
        marker=dict(color=[get_status_color(1 - s) for s in infra_sorted["health_score"]]),
        text=[f"{s:.0%}" for s in infra_sorted["health_score"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Building Health Scores",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(range=[0, 115], gridcolor="#2d3546"),
        yaxis=dict(gridcolor="#2d3546"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown('<div class="section-header">📋 Infrastructure Detail</div>', unsafe_allow_html=True)
    display_infra = infra.copy()
    display_infra["health_score"] = display_infra["health_score"].apply(lambda x: f"{x:.0%}")
    display_infra["utilization_avg"] = display_infra["utilization_avg"].apply(lambda x: f"{x:.0%}")
    display_infra["energy_efficiency"] = display_infra["energy_efficiency"].apply(lambda x: f"{x:.0%}")
    display_infra["maintenance_due"] = display_infra["maintenance_due"].map({True: "⚠️ Due", False: "✅ OK"})
    st.dataframe(display_infra, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVENTS & IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Events & Impact":
    st.markdown("## 📅 Campus Events & Occupancy Impact")

    upcoming = events[events["start_time"] >= datetime.now()].head(10)
    past = events[events["start_time"] < datetime.now()].tail(5)

    st.markdown('<div class="section-header">🎯 Upcoming Events</div>', unsafe_allow_html=True)
    for _, ev in upcoming.iterrows():
        fill_rate = ev["registered"] / ev["expected_attendance"]
        color = get_status_color(fill_rate)
        st.markdown(f"""
        <div class="room-card" style="border-left: 3px solid {COLORS['primary']}; margin-bottom: 8px">
            <div style="display:flex;justify-content:space-between;align-items:start">
                <div>
                    <b style="color:#FAFAFA;font-size:1rem">{ev['event']}</b>
                    <div style="color:#8892A4;font-size:0.85rem;margin-top:2px">
                        📍 {ev['venue']} &nbsp;|&nbsp;
                        🕐 {pd.to_datetime(ev['start_time']).strftime('%d %b, %H:%M')} ({ev['duration_hrs']}h)
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="color:{color};font-weight:700">{fill_rate:.0%} filled</div>
                    <div style="color:#8892A4;font-size:0.8rem">{ev['registered']:,} / {ev['expected_attendance']:,} registered</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Impact visualization
    st.markdown('<div class="section-header">📊 Expected Attendance Distribution</div>', unsafe_allow_html=True)
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=upcoming["event"][:8],
        y=upcoming["expected_attendance"][:8],
        marker_color=COLORS["primary"],
        text=upcoming["registered"][:8],
        textposition="outside",
        name="Expected",
    ))
    fig.add_trace(go.Bar(
        x=upcoming["event"][:8],
        y=upcoming["registered"][:8],
        marker_color=COLORS["secondary"],
        name="Registered",
    ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(tickangle=-30, gridcolor="#2d3546"),
        yaxis=dict(gridcolor="#2d3546"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS & REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics & Reports":
    st.markdown("## 📊 Campus Analytics & Reports")

    tab1, tab2, tab3 = st.tabs(["📈 Occupancy Analytics", "⚡ Energy Analytics", "🔬 Sensor Analytics"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Top 5 most used buildings
            top_bldgs = historical.groupby("building")["occupancy_rate"].mean().sort_values(ascending=False)
            import plotly.express as px
            fig = px.pie(
                names=top_bldgs.index, values=top_bldgs.values,
                title="Average Occupancy Share",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(occupancy_heatmap(historical), use_container_width=True)

        # Underutilized detection
        underutilized = historical.groupby("building")["occupancy_rate"].mean()
        underutilized = underutilized[underutilized < 0.2].sort_values()
        if not underutilized.empty:
            st.warning(f"⚠️ Potentially **underutilized buildings**: {', '.join(underutilized.index.tolist())}")

    with tab2:
        energy_by_bld = historical.groupby("building")["energy_kwh"].sum().sort_values(ascending=False).reset_index()
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=energy_by_bld["building"], y=energy_by_bld["energy_kwh"],
            marker_color=COLORS["accent"],
        ))
        fig.update_layout(
            title="Total Energy Consumption by Building (30 days)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            xaxis=dict(tickangle=-30, gridcolor="#2d3546"),
            yaxis=dict(title="Total kWh", gridcolor="#2d3546"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Efficiency score
        efficiency = historical.groupby("building").apply(
            lambda x: (x["occupancy_rate"] / (x["energy_kwh"] / x["capacity"])).mean()
        ).sort_values(ascending=False).reset_index()
        efficiency.columns = ["building", "efficiency_score"]
        st.markdown("#### 🌱 Energy Efficiency Score (Occupancy per kWh)")
        st.dataframe(efficiency, use_container_width=True, hide_index=True)

    with tab3:
        sensor_stats = historical.groupby("building").agg(
            avg_temp=("temperature_c", "mean"),
            avg_humidity=("humidity_pct", "mean"),
            avg_co2=("co2_ppm", "mean"),
            avg_noise=("noise_db", "mean"),
            alert_count=("alert", lambda x: x.notna().sum()),
        ).round(1).reset_index()
        st.dataframe(sensor_stats, use_container_width=True, hide_index=True)

        # CO2 alerts
        high_co2 = sensor_stats[sensor_stats["avg_co2"] > 800].sort_values("avg_co2", ascending=False)
        if not high_co2.empty:
            st.warning(f"🫁 Buildings with elevated CO₂ (>800 ppm): {', '.join(high_co2['building'].tolist())}")


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#8892A4;font-size:0.8rem">'
    '🏛️ Smart Campus Digital Twin &nbsp;|&nbsp; Built with Streamlit + scikit-learn &nbsp;|&nbsp; '
    f'v1.0.0 &nbsp;|&nbsp; Data refreshed: {datetime.now().strftime("%H:%M:%S")}'
    '</div>',
    unsafe_allow_html=True
)
