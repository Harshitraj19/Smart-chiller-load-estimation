"""Central configuration: paths, constants and domain assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "rfr_data"
PLANT_XLSX = DATA_DIR / "table_data.xlsx"
WEATHER_XLSX = DATA_DIR / "sensor_data.xlsx"

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
REPORTS_DIR = ARTIFACTS / "reports"

FREQ = "10min"
STEPS_PER_HOUR = 6
STEPS_PER_DAY = 144

TARGET = "CH Load"          # chiller loading, % of full load (primary target)
TARGET_RT = "RT"            # cooling delivered, refrigeration tons (secondary target)

# Hold out the final N days of every monthly block as an untouched test set.
HOLDOUT_DAYS = 7
CV_FOLDS = 5
PURGE = "1D"                # gap between CV train/test blocks to stop lag leakage
SEED = 42

# --- Plant facts derived from the data (see reports/data_audit.json) -------------
# CH Load == kW_CHH * 0.2947  ->  chiller full-load electrical input ~339 kW
CHILLER_KW_PER_PCT = 1 / 0.2947
# RT / (CH Load / 100) median ~540  ->  rated chiller capacity in tons
CHILLER_RATED_RT = 540.0
DESIGN_DELTA_T_F = 8.0      # achievable chilled-water delta T at high load (degF, data: ~7.3)
STAGE_UP_PCT = 90.0         # bring a second chiller online above this loading

# Calendar knowledge available in advance (occupancy proxies).
# Dates are generic, editable: Christmas/New-Year holiday window and public holidays.
FESTIVE_WINDOWS = [("12-22", "01-01")]
PUBLIC_HOLIDAYS = ["2023-12-25", "2024-01-01", "2024-03-29", "2024-08-15"]
