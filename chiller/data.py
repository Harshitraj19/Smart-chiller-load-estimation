"""Loading, cleaning and aligning the plant (BMS) and weather data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

PLANT_DROP = ["CH3", "CH4", "kW_CHS", "Hz_CHS"]   # always zero -> no information


def stull_wet_bulb(temp_c, rh):
    """Wet-bulb temperature (Stull 2011). Reproduces the site sensor's WBT_C to <0.1 degC."""
    t, rh = np.asarray(temp_c, float), np.asarray(rh, float)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


def load_plant() -> pd.DataFrame:
    df = pd.read_excel(C.PLANT_XLSX)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = (df.drop(columns=PLANT_DROP).drop_duplicates("DateTime")
            .sort_values("DateTime").set_index("DateTime"))
    df.columns = [c.replace(" ", "") if c.startswith(("Hz_", "Precent_")) else c for c in df.columns]
    return df.rename(columns={"WBT": "WBT_plant_F"})   # plant's own WBT probe (mostly stuck)


def load_weather() -> pd.DataFrame:
    df = pd.read_excel(C.WEATHER_XLSX)
    df.columns = ["DateTime", "RH", "Temp", "WBT"]
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df.drop_duplicates("DateTime").sort_values("DateTime").set_index("DateTime")


def build_dataset() -> tuple[pd.DataFrame, dict]:
    """Return a regular 10-minute grid (gaps kept as NaN rows) with plant + weather data.

    * Weather is logged at irregular offsets (e.g. 00:05:34); it is time-interpolated onto
      the plant's 10-minute stamps, never bridging gaps longer than 30 minutes.
    * Rows with CH Load == 0 are compressor-off / telemetry drop-outs (flow = 0 while pumps
      run, mostly single isolated readings). They are flagged and excluded as targets.
    """
    plant, weather = load_plant(), load_weather()
    grid = pd.date_range(plant.index.min(), plant.index.max(), freq=C.FREQ)

    w = weather.reindex(weather.index.union(grid)).interpolate(method="time", limit=3,
                                                               limit_area="inside")
    w = w.reindex(grid)
    # also blank weather that was interpolated across a long logging gap
    nearest = weather.index[np.clip(weather.index.searchsorted(grid), 0, len(weather) - 1)]
    far = np.abs((nearest - grid).total_seconds()) > 30 * 60
    w.loc[far, :] = np.nan

    df = plant.reindex(grid).join(w)
    df.index.name = "DateTime"
    df["observed"] = df[C.TARGET].notna()
    df["chiller_off"] = df[C.TARGET].eq(0)
    df["valid_target"] = df["observed"] & ~df["chiller_off"] & df["Temp"].notna()
    # month-long blocks (Dec-23, Mar-24, Aug-24); the lone 2024-04-01 00:00 row joins March
    block = pd.Series(df.index.to_period("M").astype(str), index=df.index)
    sizes = block[df["observed"]].value_counts()
    for tiny in sizes[sizes < C.STEPS_PER_DAY].index:      # fold stray rows into previous block
        block[block == tiny] = np.nan
    df["block"] = block.ffill()

    audit = audit_data(plant, weather, df)
    return df, audit


def audit_data(plant: pd.DataFrame, weather: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Facts about the data that drive modelling decisions (saved for the dashboard)."""
    on = plant[plant[C.TARGET] > 0]
    ratio_kw = (on[C.TARGET] / on["kW_CHH"]).median()
    rt_formula = on["GPM"] * on["DeltaCHW"] / 24
    gaps = plant.index.to_series().diff()
    return {
        "plant_rows": int(len(plant)),
        "weather_rows": int(len(weather)),
        "start": str(plant.index.min()), "end": str(plant.index.max()),
        "months": sorted(plant.index.to_period("M").astype(str).unique().tolist()),
        "rows_per_month": plant.index.to_period("M").astype(str).value_counts().sort_index().to_dict(),
        "long_gaps": [{"from": str(plant.index[i - 1]), "to": str(plant.index[i]),
                       "hours": round(g.total_seconds() / 3600, 1)}
                      for i, g in enumerate(gaps) if pd.notna(g) and g > pd.Timedelta("1h")],
        "chiller_off_rows": int((plant[C.TARGET] == 0).sum()),
        "valid_target_rows": int(df["valid_target"].sum()),
        "weather_duplicates_removed": int(pd.read_excel(C.WEATHER_XLSX).duplicated("DateTime").sum()),
        "leakage": {
            "ch_load_over_kw_chh": round(float(ratio_kw), 5),
            "corr_ch_load_kw_chh": round(float(on[C.TARGET].corr(on["kW_CHH"])), 6),
            "corr_ch_load_kw_tot": round(float(on[C.TARGET].corr(on["kW_Tot"])), 4),
            "corr_rt_vs_gpm_dt_24": round(float(on["RT"].corr(rt_formula)), 7),
        },
        "stuck_sensor": {
            "plant_WBT_mode_value": float(plant["WBT_plant_F"].mode()[0]),
            "plant_WBT_mode_share": round(float((plant["WBT_plant_F"] == plant["WBT_plant_F"].mode()[0]).mean()), 3),
        },
        "wbt_stull_max_abs_err": round(float(np.abs(
            stull_wet_bulb(weather["Temp"], weather["RH"]) - weather["WBT"]).max()), 3),
        "rated_rt_estimate": round(float((on["RT"] / (on[C.TARGET] / 100)).median()), 1),
    }
