"""Turn load forecasts into an equipment schedule and an energy estimate.

The plant today runs its pumps and cooling-tower fans at a near-constant 46-48 Hz whatever
the load (aux power ~70 kW even below 30 % chiller load, chilled-water delta-T collapsing to
~3.5 degF). A forecast lets the operator size flow/speed to the *expected* load instead.

Energy model (deliberately simple and transparent; every assumption is a parameter):
  * chiller kW        = CH Load % x CHILLER_KW_PER_PCT        (exact plant relationship)
  * required CHW flow = RT x 24 / design delta-T              (tonnage formula)
  * VFD speed         = clip(required / design flow, min speed, 1)
  * aux kW (VFD)      = aux kW at 50 Hz x speed^3             (affinity laws)
  * aux kW (today)    = observed constant-speed operation, regressed on load
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import stull_wet_bulb
from .features import build_features


@dataclass
class PlantConstants:
    aux_kw_50hz: float          # pumps + tower fans at full speed
    design_gpm: float           # max chilled-water flow (pumps at 50 Hz)
    aux_today_a: float          # today's aux kW = a + b * CH Load
    aux_today_b: float


def plant_constants(df: pd.DataFrame) -> PlantConstants:
    on = df[df["valid_target"]]
    aux = on["kW_CHP"] + on["kW_CDS"] + on["kW_CT"]
    hz = on["Hz_CHP"].clip(lower=1)
    aux_50 = float(np.median(aux / (hz / 50) ** 3))
    b, a = np.polyfit(on[C.TARGET], aux, 1)
    return PlantConstants(aux_kw_50hz=aux_50, design_gpm=float(on["GPM"].quantile(0.99)),
                          aux_today_a=float(a), aux_today_b=float(b))


# ---------------------------------------------------------------- synthetic weather --
def diurnal_shapes(df: pd.DataFrame) -> pd.DataFrame:
    """Typical normalised temperature shape and RH profile by time of day."""
    w = df[["Temp", "RH"]].dropna()
    day = w.index.normalize()
    tmin = w["Temp"].groupby(day).transform("min")
    tmax = w["Temp"].groupby(day).transform("max")
    shape = ((w["Temp"] - tmin) / (tmax - tmin).replace(0, np.nan)).clip(0, 1)
    tod = w.index.hour * 60 + w.index.minute
    out = pd.DataFrame({"t_shape": shape.groupby(tod).mean(),
                        "rh_dev": (w["RH"] - w["RH"].groupby(day).transform("mean")).groupby(tod).mean()})
    return out


def typical_day(df: pd.DataFrame, month: int) -> dict:
    w = df.loc[df.index.month == month, ["Temp", "RH"]].dropna()
    if w.empty:
        w = df[["Temp", "RH"]].dropna()
    d = w.resample("D").agg({"Temp": ["min", "max"], "RH": "mean"}).dropna()
    return {"tmin": float(d[("Temp", "min")].median()), "tmax": float(d[("Temp", "max")].median()),
            "rh": float(d[("RH", "mean")].median())}


def weather_profile(date, tmin, tmax, rh_mean, shapes: pd.DataFrame, days: int = 2) -> pd.DataFrame:
    """10-minute weather for `days` days ending on `date` (earlier days = warm-up)."""
    idx = pd.date_range(pd.Timestamp(date) - pd.Timedelta(days=days - 1), periods=days * C.STEPS_PER_DAY,
                        freq=C.FREQ)
    tod = idx.hour * 60 + idx.minute
    s = shapes.reindex(tod)
    temp = tmin + (tmax - tmin) * s["t_shape"].to_numpy()
    rh = np.clip(rh_mean + s["rh_dev"].to_numpy(), 20, 100)
    return pd.DataFrame({"Temp": temp, "RH": rh, "WBT": stull_wet_bulb(temp, rh)}, index=idx)


# ------------------------------------------------------------------------ forecasting --
def predict_bundle(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    X = X[bundle["features"]]
    p = bundle["model"].predict(X)
    lo = bundle["q10"].predict(X) - bundle["q_adj"]
    hi = bundle["q90"].predict(X) + bundle["q_adj"]
    lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    return pd.DataFrame({"pred": p, "p10": lo, "p90": hi}, index=X.index)


def forecast_day(bundles: dict, weather: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """Exogenous-tier forecast of CH Load (%) and RT for the last day of `weather`."""
    frame = weather.copy()
    frame["valid_target"] = False
    X = build_features(frame, "exogenous")
    for k, v in (overrides or {}).items():
        X[k] = v
    day = X.index[-C.STEPS_PER_DAY:]
    load = predict_bundle(bundles["load"], X.loc[day])
    rt = predict_bundle(bundles["rt"], X.loc[day])
    out = load.add_prefix("load_").join(rt.add_prefix("rt_"))
    return out.join(weather.loc[day])


# ------------------------------------------------------------------------- schedule --
def schedule(fc: pd.DataFrame, pc: PlantConstants, delta_t_f: float = C.DESIGN_DELTA_T_F, min_speed: float = 0.65,
             stage_up_pct: float = C.STAGE_UP_PCT, tariff: float = 0.12, co2_kg_kwh: float = 0.45,
             plan_on: str = "p90") -> tuple[pd.DataFrame, dict]:
    """Hourly operating plan + energy comparison (forecast-driven vs today's practice)."""
    h = fc.resample("1h").mean()
    load_plan = h["load_p90"] if plan_on == "p90" else h["load_pred"]
    rt_plan = h["rt_p90"] if plan_on == "p90" else h["rt_pred"]
    plan = pd.DataFrame(index=h.index)
    plan["Forecast load %"] = h["load_pred"]
    plan["P10 %"], plan["P90 %"] = h["load_p10"], h["load_p90"]
    plan["Forecast RT"] = h["rt_pred"]
    plan["Chillers on"] = np.where(load_plan > stage_up_pct, 2, 1)
    plan["CHW flow (GPM)"] = np.clip(rt_plan * 24 / delta_t_f, 0, pc.design_gpm)
    speed = np.clip(plan["CHW flow (GPM)"] / pc.design_gpm, min_speed, 1.0)
    plan["Pump/fan speed %"] = speed * 100
    plan["VFD Hz"] = speed * 50
    plan["Chiller kW"] = h["load_pred"] * C.CHILLER_KW_PER_PCT
    plan["Aux kW (today)"] = pc.aux_today_a + pc.aux_today_b * h["load_pred"]
    plan["Aux kW (planned)"] = pc.aux_kw_50hz * speed ** 3
    plan["Temp °C"] = h["Temp"]
    kwh_today = float((plan["Chiller kW"] + plan["Aux kW (today)"]).sum())
    kwh_plan = float((plan["Chiller kW"] + plan["Aux kW (planned)"]).sum())
    saved = kwh_today - kwh_plan
    summary = {
        "peak_load_pct": float(h["load_pred"].max()),
        "peak_hour": h["load_pred"].idxmax().strftime("%H:%M"),
        "mean_load_pct": float(h["load_pred"].mean()),
        "cooling_ton_hours": float(h["rt_pred"].sum()),
        "kwh_today": kwh_today, "kwh_planned": kwh_plan, "kwh_saved": saved,
        "pct_saved": saved / kwh_today * 100 if kwh_today else 0.0,
        "aux_pct_saved": float((plan["Aux kW (today)"] - plan["Aux kW (planned)"]).sum()
                               / plan["Aux kW (today)"].sum() * 100),
        "cost_saved": saved * tariff, "co2_saved_kg": saved * co2_kg_kwh,
        "second_chiller_hours": int((plan["Chillers on"] > 1).sum()),
    }
    return plan, summary
