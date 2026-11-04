"""Feature engineering.

Three *forecasting* tiers, each using only information available at forecast time:

    exogenous  : weather + calendar only (works with a weather forecast, no history needed)
    day_ahead  : + load history that is >= 24 h old  (plan tomorrow's operation)
    hour_ahead : + load history that is >= 1 h old   (intraday dispatch)

plus a *plant* tier ("digital twin"): chiller loading given the cooling demand (RT) and
water/condenser temperatures. Electrical columns (kW_*, Precent_*) are never used as
inputs because CH Load is literally kW_CHH * 0.2947 (target leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

H, D = C.STEPS_PER_HOUR, C.STEPS_PER_DAY


# ----------------------------------------------------------------------- weather ----
def psychrometrics(temp_c, rh, p_kpa: float = 101.325) -> dict:
    t, rh = np.asarray(temp_c, float), np.asarray(rh, float)
    psat = 0.61094 * np.exp(17.625 * t / (t + 243.04))           # kPa (Magnus)
    pv = rh / 100 * psat
    w = 0.622 * pv / (p_kpa - pv)                                  # kg/kg dry air
    a = np.log(np.clip(rh, 1e-3, None) / 100) + 17.625 * t / (243.04 + t)
    return {
        "DewPoint": 243.04 * a / (17.625 - a),
        "HumRatio": w * 1000,                                      # g/kg
        "Enthalpy": 1.006 * t + w * (2501 + 1.86 * t),             # kJ/kg
    }


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Temp"], out["RH"], out["WBT"] = df["Temp"], df["RH"], df["WBT"]
    for k, v in psychrometrics(df["Temp"], df["RH"]).items():
        out[k] = v
    # building thermal inertia: trailing means / trends of outdoor conditions
    for name, n in [("1h", H), ("3h", 3 * H), ("6h", 6 * H), ("24h", D)]:
        out[f"Temp_mean_{name}"] = df["Temp"].rolling(n, min_periods=max(1, n // 2)).mean()
    out["Enthalpy_mean_3h"] = out["Enthalpy"].rolling(3 * H, min_periods=H).mean()
    out["WBT_mean_3h"] = df["WBT"].rolling(3 * H, min_periods=H).mean()
    out["Temp_trend_1h"] = df["Temp"] - df["Temp"].shift(H)
    out["Temp_max_24h"] = df["Temp"].rolling(D, min_periods=D // 2).max()
    return out


# ---------------------------------------------------------------------- calendar ----
def _in_windows(idx: pd.DatetimeIndex, windows) -> np.ndarray:
    md = idx.strftime("%m-%d")
    mask = np.zeros(len(idx), bool)
    for start, end in windows:
        mask |= (md >= start) | (md <= end) if start > end else (md >= start) & (md <= end)
    return mask


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    mod = idx.hour * 60 + idx.minute
    out["hour"] = idx.hour
    out["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    out["dow"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month                      # season / occupancy proxy
    hol = pd.to_datetime(C.PUBLIC_HOLIDAYS).date
    out["is_holiday"] = np.isin(idx.date, hol).astype(int)
    out["festive"] = _in_windows(idx, C.FESTIVE_WINDOWS).astype(int)
    out["occupied_hours"] = ((idx.hour >= 7) & (idx.hour < 23)).astype(int)
    return out


# --------------------------------------------------------------------- load lags ----
def lag_features(y: pd.Series, horizon: int, prefix: str = "y") -> pd.DataFrame:
    """Lags/rolling stats of the target, all at least `horizon` steps old."""
    out = pd.DataFrame(index=y.index)
    base = y.shift(horizon)                                  # newest admissible value
    if horizon < D:                                          # intraday history
        for k in [0, 1, 3, 6, 12, 18]:
            out[f"{prefix}_lag_{(horizon + k) * 10}m"] = y.shift(horizon + k)
        out[f"{prefix}_mean_1h"] = base.rolling(H, min_periods=3).mean()
        out[f"{prefix}_mean_3h"] = base.rolling(3 * H, min_periods=H).mean()
        out[f"{prefix}_std_3h"] = base.rolling(3 * H, min_periods=H).std()
        out[f"{prefix}_slope_1h"] = base - y.shift(horizon + H)
    # same time yesterday / last week (always >= 24 h old)
    for k, name in [(D, "24h"), (2 * D, "48h"), (7 * D, "7d")]:
        out[f"{prefix}_lag_{name}"] = y.shift(k)
        out[f"{prefix}_win_{name}"] = y.shift(k - H).rolling(2 * H + 1, min_periods=H).mean()
    out[f"{prefix}_daymean_24h"] = y.shift(max(horizon, D)).rolling(D, min_periods=D // 2).mean()
    out[f"{prefix}_daymax_24h"] = y.shift(max(horizon, D)).rolling(D, min_periods=D // 2).max()
    return out


TIERS = {
    "exogenous": {"label": "Weather + calendar", "horizon": None,
                  "use": "Any time, from a weather forecast — no plant history needed"},
    "day_ahead": {"label": "Day-ahead (24 h)", "horizon": D,
                  "use": "Plan tomorrow: staging, pre-cooling, energy purchase"},
    "hour_ahead": {"label": "Hour-ahead (1 h)", "horizon": H,
                   "use": "Intraday dispatch: pump/tower speed, chiller sequencing"},
    "plant": {"label": "Plant digital twin", "horizon": None,
              "use": "Expected chiller loading for a cooling demand & setpoints (what-if, FDD)"},
}

PLANT_INPUTS = ["RT", "GPM", "CHWS", "DeltaCHW", "CDHI", "CDLO"]
# physics: more cooling or more lift -> more compressor work; warmer CHW supply -> less
PLANT_MONOTONE = {"RT": 1, "RT_mean_30m": 1, "CHWS": -1, "CDHI": 1, "Lift": 1}


def plant_history(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Measured plant signals (not electrical) that are at least `horizon` steps old."""
    ok = df["valid_target"]
    out = pd.DataFrame(index=df.index)
    for c in ["RT", "CHWR", "GPM", "DeltaCHW"]:
        s = df[c].where(ok).shift(horizon)
        out[f"{c}_lag_h"] = s
        out[f"{c}_mean_1h_h"] = s.rolling(H, min_periods=3).mean()
    out["Temp_change_since_obs"] = df["Temp"] - df["Temp"].shift(horizon)
    return out


def plant_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df[PLANT_INPUTS].copy()
    p["Lift"] = df["CDHI"] - df["CHWS"]                     # compressor lift proxy (degF)
    p["RT_mean_30m"] = df["RT"].rolling(3, min_periods=1).mean()
    w = weather_features(df)[["Temp", "RH", "WBT", "Enthalpy"]]
    return pd.concat([p, w], axis=1)


def build_features(df: pd.DataFrame, tier: str, target: str = C.TARGET) -> pd.DataFrame:
    if tier == "plant":
        return plant_features(df)
    parts = [weather_features(df), calendar_features(df.index)]
    if tier in ("day_ahead", "hour_ahead"):
        y = df[target].where(df["valid_target"])             # off/dropouts -> NaN history
        h = TIERS[tier]["horizon"]
        parts += [lag_features(y, h), plant_history(df, h)]
    elif tier != "exogenous":
        raise ValueError(tier)
    return pd.concat(parts, axis=1)
