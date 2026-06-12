"""
Synthetic IoT data generator for Smart Campus Digital Twin
Generates realistic occupancy, energy, environmental, and event data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

BUILDINGS = {
    "Library": {"capacity": 800, "floors": 4, "type": "study"},
    "Engineering Block A": {"capacity": 600, "floors": 5, "type": "academic"},
    "Engineering Block B": {"capacity": 500, "floors": 4, "type": "academic"},
    "Science Complex": {"capacity": 700, "floors": 6, "type": "academic"},
    "Student Center": {"capacity": 1200, "floors": 3, "type": "social"},
    "Admin Block": {"capacity": 300, "floors": 3, "type": "admin"},
    "Sports Complex": {"capacity": 2000, "floors": 2, "type": "sports"},
    "Cafeteria": {"capacity": 500, "floors": 1, "type": "food"},
    "Hostel Block A": {"capacity": 400, "floors": 8, "type": "residential"},
    "Hostel Block B": {"capacity": 350, "floors": 7, "type": "residential"},
}

ROOMS = {
    "Library": ["Reading Hall L1", "Digital Lab L2", "Group Study L3", "Silent Zone L4", "Rare Books L4"],
    "Engineering Block A": ["Lab EA101", "Lab EA102", "Classroom EA201", "Seminar EA202", "Project Room EA301"],
    "Engineering Block B": ["Lab EB101", "Workshop EB102", "Classroom EB201", "Conference EB301"],
    "Science Complex": ["Chem Lab SC101", "Bio Lab SC102", "Physics Lab SC201", "Research SC301", "Auditorium SC001"],
    "Student Center": ["Activity Hall SC1", "Game Zone SC2", "Music Room SC3", "Counseling SC4"],
    "Admin Block": ["Reception AB1", "Registrar AB2", "Finance AB3", "Exam Cell AB4"],
    "Sports Complex": ["Indoor Arena SP1", "Gym SP2", "Swimming Pool SP3", "Courts SP4"],
    "Cafeteria": ["Main Dining CF1", "Quick Bites CF2", "Outdoor Seating CF3"],
    "Hostel Block A": ["Common Room HA1", "Study Lounge HA2", "Recreation HA3"],
    "Hostel Block B": ["Common Room HB1", "Study Lounge HB2"],
}

CAMPUS_COORDS = {
    "Library": (12.9716, 77.5946),
    "Engineering Block A": (12.9720, 77.5952),
    "Engineering Block B": (12.9724, 77.5940),
    "Science Complex": (12.9712, 77.5960),
    "Student Center": (12.9718, 77.5930),
    "Admin Block": (12.9730, 77.5935),
    "Sports Complex": (12.9708, 77.5970),
    "Cafeteria": (12.9722, 77.5945),
    "Hostel Block A": (12.9700, 77.5950),
    "Hostel Block B": (12.9698, 77.5955),
}


def occupancy_pattern(hour, day_of_week, building_type):
    """Simulate realistic occupancy based on time and building type"""
    is_weekend = day_of_week >= 5

    if building_type == "academic":
        if is_weekend:
            base = 0.05
        elif 8 <= hour <= 18:
            base = 0.7 + 0.25 * np.sin((hour - 8) * np.pi / 10)
        elif 18 < hour <= 21:
            base = 0.3
        else:
            base = 0.02
    elif building_type == "study":
        if is_weekend:
            if 10 <= hour <= 20:
                base = 0.5
            else:
                base = 0.1
        elif 6 <= hour <= 22:
            base = 0.4 + 0.5 * np.sin((hour - 6) * np.pi / 16)
        else:
            base = 0.05
    elif building_type == "social":
        if 10 <= hour <= 22:
            base = 0.5 + (0.3 if is_weekend else 0.1) * np.random.random()
        else:
            base = 0.05
    elif building_type == "food":
        if hour in [8, 9, 12, 13, 19, 20]:
            base = 0.9
        elif hour in [7, 10, 11, 14, 18, 21]:
            base = 0.5
        else:
            base = 0.1
    elif building_type == "sports":
        if is_weekend and 8 <= hour <= 20:
            base = 0.6
        elif 16 <= hour <= 21:
            base = 0.7
        elif 6 <= hour <= 8:
            base = 0.4
        else:
            base = 0.05
    elif building_type == "residential":
        if 22 <= hour or hour <= 6:
            base = 0.85
        elif 7 <= hour <= 9:
            base = 0.6
        elif 10 <= hour <= 17:
            base = 0.2
        else:
            base = 0.5
    elif building_type == "admin":
        if is_weekend or hour < 9 or hour > 17:
            base = 0.02
        else:
            base = 0.6
    else:
        base = 0.3

    noise = np.random.normal(0, 0.05)
    return min(max(base + noise, 0), 1.0)


def generate_historical_data(days=30):
    """Generate 30 days of historical IoT sensor readings"""
    records = []
    end_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    current = start_dt
    while current <= end_dt:
        for building, info in BUILDINGS.items():
            occ_rate = occupancy_pattern(current.hour, current.weekday(), info["type"])
            occupied = int(occ_rate * info["capacity"])

            # Energy consumption (kWh) correlates with occupancy + base load
            base_energy = info["capacity"] * 0.01
            energy = base_energy + (occ_rate * info["capacity"] * 0.05) + np.random.normal(0, 0.5)

            # Environmental readings
            temp = 22 + 3 * np.sin((current.hour - 14) * np.pi / 12) + np.random.normal(0, 0.5)
            humidity = 55 + 10 * np.sin((current.hour - 6) * np.pi / 12) + np.random.normal(0, 2)
            co2 = 400 + 300 * occ_rate + np.random.normal(0, 20)
            noise_db = 30 + 40 * occ_rate + np.random.normal(0, 3)

            # Alerts
            alert = None
            if occ_rate > 0.9:
                alert = "HIGH_OCCUPANCY"
            elif co2 > 1000:
                alert = "HIGH_CO2"
            elif energy > base_energy * 3:
                alert = "ENERGY_SPIKE"

            records.append({
                "timestamp": current,
                "building": building,
                "building_type": info["type"],
                "capacity": info["capacity"],
                "occupied": max(0, occupied),
                "occupancy_rate": round(occ_rate, 3),
                "energy_kwh": round(max(0, energy), 2),
                "temperature_c": round(temp, 1),
                "humidity_pct": round(min(max(humidity, 20), 90), 1),
                "co2_ppm": round(max(400, co2), 0),
                "noise_db": round(max(25, noise_db), 1),
                "alert": alert,
                "lat": CAMPUS_COORDS[building][0] + np.random.normal(0, 0.0001),
                "lon": CAMPUS_COORDS[building][1] + np.random.normal(0, 0.0001),
            })
        current += timedelta(hours=1)

    return pd.DataFrame(records)


def generate_room_data():
    """Generate current room-level data"""
    records = []
    now = datetime.now()
    for building, rooms in ROOMS.items():
        btype = BUILDINGS[building]["type"]
        for room in rooms:
            cap = random.randint(20, 80)
            occ_rate = occupancy_pattern(now.hour, now.weekday(), btype)
            occ_rate += random.uniform(-0.1, 0.1)
            occ_rate = min(max(occ_rate, 0), 1.0)
            occupied = int(occ_rate * cap)

            score = (1 - occ_rate) * 0.6 + random.uniform(0, 0.4)
            features = []
            if random.random() > 0.5:
                features.append("WiFi")
            if random.random() > 0.6:
                features.append("AC")
            if random.random() > 0.7:
                features.append("Projector")
            if random.random() > 0.8:
                features.append("Whiteboard")
            if random.random() > 0.9:
                features.append("Standing Desks")

            records.append({
                "building": building,
                "room": room,
                "capacity": cap,
                "occupied": max(0, occupied),
                "occupancy_rate": round(occ_rate, 3),
                "comfort_score": round(min(score, 1.0), 2),
                "noise_level": random.choice(["Low", "Medium", "High"]),
                "features": ", ".join(features) if features else "Basic",
                "available": occ_rate < 0.8,
                "last_updated": now.strftime("%H:%M:%S"),
            })
    return pd.DataFrame(records)


def generate_events():
    """Generate upcoming campus events"""
    events = []
    now = datetime.now()
    event_types = [
        ("Tech Symposium", "Engineering Block A", 300),
        ("Cultural Fest", "Student Center", 800),
        ("Sports Day", "Sports Complex", 1500),
        ("Convocation", "Science Complex", 500),
        ("Hackathon", "Library", 200),
        ("Career Fair", "Admin Block", 400),
        ("Seminar: AI in Healthcare", "Engineering Block B", 150),
        ("Inter-College Debate", "Student Center", 250),
        ("Research Expo", "Science Complex", 350),
        ("Alumni Meet", "Cafeteria", 200),
    ]
    for i, (name, venue, expected) in enumerate(event_types):
        start = now + timedelta(days=random.randint(0, 14), hours=random.randint(9, 18))
        events.append({
            "event": name,
            "venue": venue,
            "start_time": start,
            "duration_hrs": random.choice([2, 3, 4]),
            "expected_attendance": expected,
            "registered": int(expected * random.uniform(0.5, 0.95)),
        })
    return pd.DataFrame(events).sort_values("start_time")


def generate_infrastructure_health():
    """Generate infrastructure health/utilization data"""
    infra = []
    for building, info in BUILDINGS.items():
        age_years = random.randint(2, 25)
        utilization = random.uniform(0.2, 0.95)
        maintenance_due = random.random() > 0.7
        energy_efficiency = random.uniform(0.4, 0.95)

        health_score = (
            (1 - age_years / 30) * 0.3
            + (1 - utilization) * 0.2
            + energy_efficiency * 0.3
            + (0 if maintenance_due else 0.2)
        )

        infra.append({
            "building": building,
            "type": info["type"],
            "age_years": age_years,
            "utilization_avg": round(utilization, 2),
            "energy_efficiency": round(energy_efficiency, 2),
            "maintenance_due": maintenance_due,
            "health_score": round(max(0.1, health_score), 2),
            "last_maintenance": (datetime.now() - timedelta(days=random.randint(30, 365))).date(),
            "next_maintenance": (datetime.now() + timedelta(days=random.randint(7, 90))).date(),
            "issues": random.randint(0, 5),
        })
    return pd.DataFrame(infra)


def save_all_data(path="data"):
    os.makedirs(path, exist_ok=True)
    print("Generating historical data (30 days)...")
    hist = generate_historical_data(30)
    hist.to_csv(f"{path}/historical.csv", index=False)

    print("Generating room data...")
    rooms = generate_room_data()
    rooms.to_csv(f"{path}/rooms.csv", index=False)

    print("Generating events...")
    events = generate_events()
    events.to_csv(f"{path}/events.csv", index=False)

    print("Generating infrastructure health...")
    infra = generate_infrastructure_health()
    infra.to_csv(f"{path}/infrastructure.csv", index=False)

    print("Done.")
    return hist, rooms, events, infra


if __name__ == "__main__":
    save_all_data()
