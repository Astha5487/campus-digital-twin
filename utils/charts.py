"""Utility helpers for Smart Campus Digital Twin"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# Color palette
COLORS = {
    "primary": "#1B4FDB",
    "secondary": "#00C49A",
    "accent": "#FF6B35",
    "warning": "#FFC107",
    "danger": "#E74C3C",
    "success": "#2ECC71",
    "bg": "#0E1117",
    "card": "#1A1F2E",
    "text": "#FAFAFA",
    "muted": "#8892A4",
}

BUILDING_COLORS = px.colors.qualitative.Bold


def get_status_color(rate: float) -> str:
    if rate < 0.4:
        return COLORS["success"]
    elif rate < 0.7:
        return COLORS["warning"]
    elif rate < 0.9:
        return COLORS["accent"]
    return COLORS["danger"]


def occupancy_badge(rate: float) -> str:
    if rate < 0.4:
        return "🟢 Available"
    elif rate < 0.7:
        return "🟡 Moderate"
    elif rate < 0.9:
        return "🟠 Busy"
    return "🔴 Full"


def format_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ─── Chart Builders ───────────────────────────────────────────────────────────

def occupancy_heatmap(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()

    pivot = df.pivot_table(
        values="occupancy_rate",
        index="day",
        columns="hour",
        aggfunc="mean",
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#0a2463"],
                [0.4, "#1B4FDB"],
                [0.7, "#FFC107"],
                [1.0, "#E74C3C"],
            ],
            zmin=0, zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Occupancy"),
        )
    )
    fig.update_layout(
        title="Weekly Occupancy Heatmap",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis_title="Hour of Day",
        yaxis_title="",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def building_bar_chart(df: pd.DataFrame) -> go.Figure:
    latest = df.sort_values("timestamp").groupby("building").last().reset_index()
    latest = latest.sort_values("occupancy_rate", ascending=True)
    colors = [get_status_color(r) for r in latest["occupancy_rate"]]

    fig = go.Figure(
        go.Bar(
            x=latest["occupancy_rate"] * 100,
            y=latest["building"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{r:.0%}" for r in latest["occupancy_rate"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Current Occupancy by Building",
        xaxis_title="Occupancy %",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(range=[0, 115], gridcolor="#2d3546"),
        yaxis=dict(gridcolor="#2d3546"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


def energy_timeline(df: pd.DataFrame, building: str) -> go.Figure:
    bdf = df[df["building"] == building].sort_values("timestamp").tail(168)  # 7 days
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=bdf["timestamp"], y=bdf["energy_kwh"],
            name="Energy (kWh)", fill="tozeroy",
            line=dict(color=COLORS["accent"], width=2),
            fillcolor="rgba(255,107,53,0.15)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bdf["timestamp"], y=bdf["occupancy_rate"] * 100,
            name="Occupancy %", line=dict(color=COLORS["secondary"], width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Energy vs Occupancy — {building}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(title_text="Energy kWh", secondary_y=False, gridcolor="#2d3546")
    fig.update_yaxes(title_text="Occupancy %", secondary_y=True, gridcolor="#2d3546")
    return fig


def forecast_chart(forecast_df: pd.DataFrame, building: str) -> go.Figure:
    fdf = forecast_df[forecast_df["building"] == building]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fdf["timestamp"],
            y=fdf["predicted_occupancy"] * 100,
            name="Predicted Occupancy %",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(27,79,219,0.15)",
        )
    )
    # Add threshold lines
    for threshold, label, color in [
        (80, "Busy Threshold", COLORS["warning"]),
        (90, "Capacity Alert", COLORS["danger"]),
    ]:
        fig.add_hline(
            y=threshold, line_dash="dash", line_color=color,
            annotation_text=label, annotation_position="bottom right",
        )
    fig.update_layout(
        title=f"24-Hour Occupancy Forecast — {building}",
        xaxis_title="Time",
        yaxis_title="Predicted Occupancy %",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        yaxis=dict(range=[0, 110], gridcolor="#2d3546"),
        xaxis=dict(gridcolor="#2d3546"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def environmental_radar(df: pd.DataFrame, building: str) -> go.Figure:
    row = df[df["building"] == building].sort_values("timestamp").tail(1)
    if row.empty:
        return go.Figure()
    row = row.iloc[0]

    categories = ["Temperature", "Humidity", "CO₂ Level", "Noise", "Occupancy"]
    # Normalize to 0-100
    values = [
        min((row["temperature_c"] - 15) / 20 * 100, 100),
        row["humidity_pct"],
        min((row["co2_ppm"] - 400) / 1600 * 100, 100),
        min(row["noise_db"] / 80 * 100, 100),
        row["occupancy_rate"] * 100,
    ]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(27,79,219,0.25)",
            line=dict(color=COLORS["primary"], width=2),
            name=building,
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2d3546"),
            angularaxis=dict(gridcolor="#2d3546"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def crowd_flow_sankey(flow_snapshot: dict, next_snapshot: dict) -> go.Figure:
    buildings = list(flow_snapshot.keys())
    label = buildings + buildings
    n = len(buildings)
    sources, targets, values = [], [], []
    for i, src in enumerate(buildings):
        for j, tgt in enumerate(buildings):
            if src != tgt:
                move = flow_snapshot.get(src, 0) * 0.1
                if move > 1:
                    sources.append(i)
                    targets.append(n + j)
                    values.append(round(move, 0))

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=label,
                color=[COLORS["primary"]] * n + [COLORS["secondary"]] * n,
            ),
            link=dict(source=sources, target=targets, value=values, color="rgba(27,79,219,0.3)"),
        )
    )
    fig.update_layout(
        title="Crowd Flow Between Buildings",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500,
    )
    return fig


def anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty or "is_anomaly" not in df.columns:
        return go.Figure()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    normal = df[df["is_anomaly"] == False]
    anomaly = df[df["is_anomaly"] == True]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal["energy_kwh"],
        mode="markers", name="Normal",
        marker=dict(color=COLORS["secondary"], size=4, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=anomaly["timestamp"], y=anomaly["energy_kwh"],
        mode="markers", name="Anomaly",
        marker=dict(color=COLORS["danger"], size=10, symbol="x", opacity=0.9),
    ))
    fig.update_layout(
        title="Anomaly Detection — Energy Consumption",
        xaxis_title="Time", yaxis_title="Energy (kWh)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#2d3546"),
        yaxis=dict(gridcolor="#2d3546"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
