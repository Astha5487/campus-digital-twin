"""Global configuration for Smart Campus Digital Twin"""

APP_TITLE = "Smart Campus Digital Twin"
APP_ICON = "🏛️"
VERSION = "1.0.0"

# Campus identity
CAMPUS_NAME = "IIT Roorkee — Digital Twin"
CAMPUS_CENTER = (12.9716, 77.5946)

# Refresh
AUTO_REFRESH_SECONDS = 30

# Thresholds
OCCUPANCY_HIGH = 0.85
OCCUPANCY_CRITICAL = 0.95
CO2_HIGH = 1000
CO2_CRITICAL = 1500
ENERGY_SPIKE_MULTIPLIER = 2.5
TEMP_HIGH = 30
NOISE_HIGH = 70

# Streamlit page config
PAGE_CONFIG = dict(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
