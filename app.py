"""Smart Chiller Load Estimation — operations dashboard.

    streamlit run app.py        (run `python train.py` first)
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from chiller import config as C
from chiller import planner as P
from chiller.features import TIERS, plant_features

st.set_page_config(page_title="Smart Chiller Load", page_icon="❄️", layout="wide")

# ------------------------------------------------------------------ design tokens ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "rgba(42,120,214,0.16)"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
SEQ = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
MONTH_COLORS = {"2023-12": BLUE, "2024-03": ORANGE, "2024-08": AQUA}
MONTH_NAMES = {"2023-12": "Dec 2023", "2024-03": "Mar 2024", "2024-08": "Aug 2024"}

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1400px;}}
  h1, h2, h3 {{letter-spacing: -0.01em;}}
  .hero {{padding: 1.3rem 1.5rem; border-radius: 14px; background: linear-gradient(120deg,#0d366b,#2a78d6);
          color: #fff; margin-bottom: 1rem;}}
  .hero h1 {{color:#fff; margin:0 0 .3rem 0; font-size: 2rem;}}
  .hero p {{color:#e6f0fd; margin:0; font-size: 1.02rem;}}
  .kicker {{color:{MUTED}; text-transform: uppercase; font-size: .74rem; letter-spacing: .08em; font-weight:600;}}
  .note {{color:{INK2}; font-size:.9rem;}}
  .pill {{display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem; font-weight:600;
          margin-right:.35rem;}}
  .pill.good {{background:#e3f5e3; color:#006300;}}
  .pill.warn {{background:#fff3d6; color:#7a5200;}}
  .pill.bad  {{background:#fbe3e3; color:#9b1c1c;}}
  .finding {{border-left: 3px solid {BLUE}; padding: .35rem .8rem; margin: .35rem 0; background:#fcfcfb;
             border-radius: 0 8px 8px 0;}}
  div[data-testid="stMetric"] {{background:#fcfcfb; border:1px solid rgba(11,11,11,.08); border-radius:12px;
                                padding:.7rem .9rem;}}
  div[data-testid="stMetricValue"] {{font-size: 1.65rem;}}
</style>""", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 360, hover: str = "x unified", legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode=hover,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fcfcfb",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None) if legend else None,
        showlegend=legend, hoverlabel=dict(bgcolor="#fff", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED)
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------------------ loaders ----
@st.cache_data
def load_metrics():
    return json.loads((C.REPORTS_DIR / "metrics.json").read_text())


@st.cache_data
def load_dataset():
    return pd.read_parquet(C.ARTIFACTS / "dataset.parquet")


@st.cache_data
def load_holdout(tier: str, target: str = "load"):
    suffix = "" if target == "load" else "__RT"
    return pd.read_parquet(C.REPORTS_DIR / f"holdout_{tier}{suffix}.parquet")


@st.cache_resource
def load_bundle(tier: str, target: str = C.TARGET):
    return joblib.load(C.MODELS_DIR / f"{tier}__{target.replace(' ', '_')}.joblib")


@st.cache_data
def constants_and_shapes():
    df = load_dataset()
    return P.plant_constants(df), P.diurnal_shapes(df)


if not (C.REPORTS_DIR / "metrics.json").exists():
    st.error("No trained models found. Run `python train.py` first, then reload this page.")
    st.stop()

M = load_metrics()
DF = load_dataset()
PC, SHAPES = constants_and_shapes()
BUNDLES = {"load": load_bundle("exogenous"), "rt": load_bundle("exogenous", C.TARGET_RT)}
FORECAST_TIERS = ["exogenous", "day_ahead", "hour_ahead"]


def best(tier, key, hourly=False, split="ho"):
    b = M["tiers"][tier]["best"]
    return b[f"{split}_hourly_{key}" if hourly else f"{split}_{key}"]


# ------------------------------------------------------------------------ sidebar ----
with st.sidebar:
    st.markdown("### ❄️ Smart Chiller")
    page = st.radio("Navigate", ["Overview", "Data & insights", "Model performance", "Day planner",
                                 "Plant digital twin", "Methodology"], label_visibility="collapsed", key="page")
    st.divider()
    a = M["audit"]
    st.caption(f"Data: {a['plant_rows']:,} plant rows · {len(a['months']) - 1} months "
               f"(Dec-23, Mar-24, Aug-24) · 10-min BMS logs")
    st.caption(f"Models trained {M['generated']}")


# ======================================================================= OVERVIEW ====
def page_overview():
    st.markdown("""<div class="hero"><h1>Smart Chiller Load Estimation</h1>
    <p>Forecasting a hotel's chiller load from weather, time and occupancy signals — so the plant runs
    only the capacity it needs, instead of full speed around the clock.</p></div>""", unsafe_allow_html=True)

    fc_plan = typical_plan(8)
    c = st.columns(4)
    c[0].metric("Hour-ahead accuracy (hourly R²)", f"{best('hour_ahead', 'R2', True):.3f}",
                help="Unseen holdout weeks, hourly resolution. 1.0 = perfect.")
    c[1].metric("Day-ahead error (hourly MAE)", f"{best('day_ahead', 'MAE', True):.1f} pts",
                help="Mean absolute error in chiller-load percentage points on the holdout weeks.")
    c[2].metric("Weather-only model (hourly R²)", f"{best('exogenous', 'R2', True):.3f}",
                help="No plant history at all — only a weather forecast and the calendar.")
    c[3].metric("Est. plant energy saved / day", f"{fc_plan['kwh_saved']:,.0f} kWh",
                f"{fc_plan['pct_saved']:.0f}% of chiller-plant energy",
                help="Typical August day: forecast-driven variable-speed pumps & fans vs today's "
                     "near-constant 47–48 Hz operation. See Day planner for assumptions.")

    st.markdown("#### Three forecasting horizons + a plant model")
    rows = []
    for t in FORECAST_TIERS + ["plant"]:
        r = M["tiers"][t]
        rows.append({"Model": TIERS[t]["label"], "Use": TIERS[t]["use"], "Selected algorithm": r["selected"],
                     "Holdout MAE (10-min)": r["best"]["ho_MAE"], "Holdout MAE (hourly)": r["best"]["ho_hourly_MAE"],
                     "Holdout R² (hourly)": r["best"]["ho_hourly_R2"],
                     "80% interval coverage": r["interval"]["holdout_coverage"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "Holdout MAE (10-min)": st.column_config.NumberColumn(format="%.2f"),
        "Holdout MAE (hourly)": st.column_config.NumberColumn(format="%.2f"),
        "Holdout R² (hourly)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        "80% interval coverage": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption("MAE is in chiller-load percentage points (CH Load ranges 0–95%). Holdout = the last 7 days of "
               "each month, never used for training or model selection.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### What the data audit found")
        lk = a["leakage"]
        findings = [
            f"**The previous models' 99.6% R² was target leakage.** `CH Load` is exactly `kW_CHH × "
            f"{lk['ch_load_over_kw_chh']}` (corr {lk['corr_ch_load_kw_chh']}), and `kW_Tot` contains `kW_CHH` "
            f"(corr {lk['corr_ch_load_kw_tot']}). Predicting load from power is circular — useless for planning.",
            "**`sensor_data.csv` has corrupted timestamps** (`00:42.0` — Excel mm:ss). The old outer merge "
            "matched zero rows and mean-imputed every weather value. We rebuilt from the original .xlsx "
            "and time-interpolated weather onto the 10-min plant grid.",
            f"**The plant's own WBT probe is stuck** at {a['stuck_sensor']['plant_WBT_mode_value']} °F "
            f"{a['stuck_sensor']['plant_WBT_mode_share']:.0%} of the time; the weather station's WBT matches "
            f"Stull's formula to ±{a['wbt_stull_max_abs_err']} °C, so it was used instead.",
            f"**{a['chiller_off_rows']} compressor-off / drop-out readings** (load = 0 while pumps run) were "
            "excluded as targets, not imputed.",
            "**Only 3 non-contiguous months** exist, and the Christmas week (load −50%) occurs once — the "
            "single hardest period for any model.",
            "**Pumps & tower fans run at ~47–48 Hz regardless of load** and chilled-water ΔT collapses to "
            "~3.5 °F at low load: a constant-flow plant with large, forecastable savings.",
        ]
        for f in findings:
            st.markdown(f'<div class="finding">{md(f)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Holdout week: hour-ahead forecast vs actual")
        h = load_holdout("hour_ahead")
        h = h[h.index.month == 8].resample("1h").mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.index, y=h["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h.index, y=h["p10"], fill="tonexty", fillcolor=BAND, line=dict(width=0),
                                 name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h.index, y=h["actual"], name="Actual", line=dict(color=INK2, width=2)))
        fig.add_trace(go.Scatter(x=h.index, y=h["pred"], name="Forecast", line=dict(color=BLUE, width=2)))
        fig.update_yaxes(title="Chiller load (%)")
        show(style(fig, 400))
        st.caption("25–31 Aug 2024, hourly means. These days were never seen during training.")


def md(text: str) -> str:
    """Tiny markdown -> html for bold/code inside custom divs."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


@st.cache_data
def typical_plan(month: int):
    td = P.typical_day(DF, month)
    d = date(2024, month, 21)
    wx = P.weather_profile(d, td["tmin"], td["tmax"], td["rh"], SHAPES)
    fc = P.forecast_day(BUNDLES, wx)
    return P.schedule(fc, PC)[1]


# ================================================================ DATA & INSIGHTS ====
def page_data():
    st.title("Data & insights")
    st.markdown('<p class="note">What drives the chiller load: time of day, weather and occupancy '
                '(proxied by day type, holidays and season).</p>', unsafe_allow_html=True)
    df = DF[DF["valid_target"]].copy()
    df["block"] = df["block"].astype(str)

    blk = st.segmented_control("Month", list(MONTH_NAMES), format_func=MONTH_NAMES.get,
                               default="2023-12")
    blk = blk or "2023-12"
    d = df[df["block"] == blk]
    hd = d[[C.TARGET, "RT", "Temp", "RH"]].resample("1h").mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        subplot_titles=("Chiller load (%) — hourly mean", "Outdoor temperature (°C)"))
    fig.add_trace(go.Scatter(x=hd.index, y=hd[C.TARGET], name="Chiller load", line=dict(color=BLUE, width=1.6)),
                  1, 1)
    fig.add_trace(go.Scatter(x=hd.index, y=hd["Temp"], name="Temperature", line=dict(color=ORANGE, width=1.6)),
                  2, 1)
    if blk == "2023-12":
        fig.add_vrect(x0="2023-12-22", x1="2023-12-30", fillcolor="rgba(137,135,129,.12)", line_width=0,
                      annotation_text="Christmas week", annotation_position="top left",
                      annotation_font_color=INK2, row=1, col=1)
    fig.update_annotations(font_size=12, font_color=INK2)
    show(style(fig, 470, legend=False))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Average daily load profile by month")
        fig = go.Figure()
        prof = df.groupby(["block", df.index.hour])[C.TARGET].mean().unstack(0)
        for b in MONTH_NAMES:
            fig.add_trace(go.Scatter(x=prof.index, y=prof[b], name=MONTH_NAMES[b],
                                     line=dict(color=MONTH_COLORS[b], width=2)))
            fig.add_annotation(x=23, y=prof[b].iloc[-1], text=MONTH_NAMES[b], showarrow=False, xanchor="left",
                               xshift=6, font=dict(color=INK2, size=11))
        fig.update_xaxes(title="Hour of day", dtick=3, range=[0, 26])
        fig.update_yaxes(title="Chiller load (%)")
        show(style(fig, 340))
        st.caption("Night trough ≈ 37%, afternoon peak 66–85%. March runs hottest despite similar weather to "
                   "August — a seasonal occupancy effect the models capture through the calendar.")
    with c2:
        st.markdown(f"##### Load by hour and weekday — {MONTH_NAMES[blk]}")
        hm = d.pivot_table(index=d.index.dayofweek, columns=d.index.hour, values=C.TARGET, aggfunc="mean")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=[days[i] for i in hm.index], colorscale=SEQ,
                                   colorbar=dict(title="%", thickness=10), xgap=2, ygap=2,
                                   hovertemplate="%{y} %{x}:00 — %{z:.1f}%<extra></extra>"))
        fig.update_xaxes(title="Hour of day", dtick=3)
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 340, hover="closest", legend=False))
        st.caption("Time of day dominates; weekday vs weekend differences are small for this hotel.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Weather sensitivity (hourly means)")
        hh = df[[C.TARGET, "Temp", "block"]].copy()
        hh = hh.groupby([hh["block"], pd.Grouper(freq="1h")])[[C.TARGET, "Temp"]].mean().reset_index()
        fig = go.Figure()
        for b in MONTH_NAMES:
            s = hh[hh["block"] == b]
            fig.add_trace(go.Scattergl(x=s["Temp"], y=s[C.TARGET], mode="markers", name=MONTH_NAMES[b],
                                       marker=dict(color=MONTH_COLORS[b], size=6, opacity=0.55,
                                                   line=dict(color="#fcfcfb", width=1))))
        fig.update_xaxes(title="Outdoor temperature (°C)")
        fig.update_yaxes(title="Chiller load (%)")
        show(style(fig, 340, hover="closest"))
        corr = df[[C.TARGET, "Temp", "WBT", "RH"]].corr()[C.TARGET]
        st.caption(f"Correlation with load: temperature {corr['Temp']:.2f}, wet-bulb {corr['WBT']:.2f}, "
                   f"RH {corr['RH']:.2f}. Hot hours are also occupied hours — the models separate the two.")
    with c4:
        st.markdown("##### Auxiliary equipment barely follows the load")
        on = df.assign(aux=df["kW_CHP"] + df["kW_CDS"] + df["kW_CT"])
        on["bin"] = pd.cut(on[C.TARGET], [0, 30, 45, 60, 75, 100], labels=["<30", "30–45", "45–60", "60–75", ">75"])
        g = on.groupby("bin", observed=True).agg(aux=("aux", "mean"), hz=("Hz_CHP", "mean"),
                                                 dt=("DeltaCHW", "mean"))
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g["aux"], marker=dict(color=BLUE, cornerradius=4),
                               text=[f"{v:.0f} kW<br>{h:.0f} Hz" for v, h in zip(g["aux"], g["hz"])],
                               textposition="outside", textfont=dict(color=INK2),
                               hovertemplate="Load %{x}%: %{y:.1f} kW<extra></extra>", name="Aux kW"))
        fig.update_xaxes(title="Chiller load band (%)")
        fig.update_yaxes(title="Pumps + tower fans (kW)", range=[0, g["aux"].max() * 1.25])
        show(style(fig, 340, hover="closest", legend=False))
        st.caption(f"Chilled-water ΔT: {g['dt'].iloc[0]:.1f} °F at low load vs {g['dt'].iloc[-1]:.1f} °F at "
                   "high load — classic constant-flow ‘low ΔT syndrome’. This is the energy the planner recovers.")

    st.markdown("##### Daily mean load — the occupancy signal")
    dm = df[C.TARGET].resample("D").mean().dropna()
    colors = [MUTED if (i.month == 12 and 22 <= i.day <= 29) else BLUE for i in dm.index]
    fig = go.Figure(go.Bar(x=[i.strftime("%d %b %y") for i in dm.index], y=dm.values,
                           marker=dict(color=colors, cornerradius=3),
                           hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fig.update_xaxes(type="category", tickangle=-60, nticks=30)
    fig.update_yaxes(title="Daily mean load (%)")
    show(style(fig, 300, hover="closest", legend=False))
    st.caption("Grey bars: 22–29 Dec. Load fell by up to half with no weather change — an occupancy effect "
               "(the problem statement's hotel-occupancy driver), encoded as a festive-period calendar flag.")


# ============================================================== MODEL PERFORMANCE ====
def page_models():
    st.title("Model performance")
    st.markdown('<p class="note">Every model was scored with purged, week-blocked cross-validation; the '
                'winner per horizon was chosen on CV only and then evaluated once on the untouched holdout '
                '(last 7 days of each month).</p>', unsafe_allow_html=True)
    tier = st.segmented_control("Horizon", FORECAST_TIERS + ["plant"], format_func=lambda t: TIERS[t]["label"],
                                default="hour_ahead") or "hour_ahead"
    res = M["tiers"][tier]
    hourly = st.toggle("Score at hourly resolution (planning interval, ASHRAE Guideline 14)", value=True)
    pre = "hourly_" if hourly else ""

    b = res["best"]
    c = st.columns(5)
    c[0].metric("Selected model", res["selected"].replace("Ensemble: ", "Ensemble ·")[:34])
    c[1].metric("Holdout MAE", f"{b[f'ho_{pre}MAE']:.2f} pts")
    c[2].metric("Holdout R²", f"{b[f'ho_{pre}R2']:.3f}")
    c[3].metric("Holdout CV(RMSE)", f"{b[f'ho_{pre}CV_RMSE']:.1f}%")
    c[4].metric("80% interval coverage", f"{res['interval']['holdout_coverage']:.0f}%",
                f"±{res['interval']['mean_width'] / 2:.1f} pts half-width", delta_color="off")

    cv_ok = b["ho_hourly_CV_RMSE"] < 30
    nmbe_ok = abs(b["ho_hourly_NMBE"]) < 10
    st.markdown(
        f'<span class="pill {"good" if cv_ok else "bad"}">{"✓" if cv_ok else "✗"} Hourly CV(RMSE) '
        f'{b["ho_hourly_CV_RMSE"]:.1f}% (ASHRAE G14 limit 30%)</span>'
        f'<span class="pill {"good" if nmbe_ok else "bad"}">{"✓" if nmbe_ok else "✗"} NMBE '
        f'{b["ho_hourly_NMBE"]:+.1f}% (limit ±10%)</span>', unsafe_allow_html=True)
    if res["interval"]["holdout_coverage"] < 77:
        st.caption(f"Intervals were calibrated to 80% on cross-validation; on the holdout they cover "
                   f"{res['interval']['holdout_coverage']:.0f}% because the holdout weeks are harder than "
                   "average (Christmas week, the late-March peak). Treat P90 as a planning value, not a hard "
                   "ceiling.")

    st.markdown("##### Leaderboard")
    lb = pd.DataFrame(res["leaderboard"])
    view = pd.DataFrame({
        "Model": lb["model"],
        "CV MAE": lb[f"cv_{pre}MAE"], "CV R²": lb[f"cv_{pre}R2"],
        "Holdout MAE": lb[f"ho_{pre}MAE"], "Holdout RMSE": lb[f"ho_{pre}RMSE"], "Holdout R²": lb[f"ho_{pre}R2"],
        "CV(RMSE) %": lb[f"ho_{pre}CV_RMSE"], "Within ±5 pts": lb[f"ho_{pre}Within5"],
    }).sort_values("CV MAE")
    view.insert(0, "", np.where(view["Model"] == res["selected"], "★", ""))
    st.dataframe(view, hide_index=True, use_container_width=True, column_config={
        **{k: st.column_config.NumberColumn(format="%.3f") for k in ["CV MAE", "CV R²", "Holdout MAE",
                                                                      "Holdout RMSE", "Holdout R²"]},
        "CV(RMSE) %": st.column_config.NumberColumn(format="%.1f"),
        "Within ±5 pts": st.column_config.NumberColumn(format="%.0f%%")})

    c1, c2 = st.columns([1, 1])
    with c1:
        v = view.sort_values("Holdout MAE", ascending=False)
        colors = [BLUE if m == res["selected"] else (AXIS if m.startswith("Baseline") else "#86b6ef")
                  for m in v["Model"]]
        fig = go.Figure(go.Bar(y=[m[:40] for m in v["Model"]], x=v["Holdout MAE"], orientation="h",
                               marker=dict(color=colors, cornerradius=4),
                               text=[f"{x:.2f}" for x in v["Holdout MAE"]], textposition="outside",
                               textfont=dict(color=INK2), hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Holdout MAE (load-% points) — lower is better",
                         range=[0, v["Holdout MAE"].max() * 1.18])
        fig.update_layout(title=dict(text="Error by model (★ selected = dark blue, baselines = grey)",
                                     font_size=13))
        show(style(fig, 380, hover="closest", legend=False))
    with c2:
        imp = pd.Series(res["shap_importance"]).head(15)[::-1]
        fig = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}: %{x:.2f} pts<extra></extra>"))
        fig.update_xaxes(title="Mean |SHAP| contribution (load-% points)")
        fig.update_layout(title=dict(text="What the model relies on (SHAP, holdout)", font_size=13))
        show(style(fig, 380, hover="closest", legend=False))

    st.markdown("##### Holdout forecast vs actual")
    h = load_holdout(tier)
    blk = st.segmented_control("Holdout week", ["2023-12", "2024-03", "2024-08"], format_func=MONTH_NAMES.get,
                               default="2024-08", key="ho_blk") or "2024-08"
    hb = h[h.index.strftime("%Y-%m") == blk]
    if hourly:
        hb = hb.resample("1h").mean().dropna(subset=["actual"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hb.index, y=hb["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hb.index, y=hb["p10"], fill="tonexty", fillcolor=BAND, line=dict(width=0),
                             name="80% interval"))
    fig.add_trace(go.Scatter(x=hb.index, y=hb["actual"], name="Actual", line=dict(color=INK2, width=1.8)))
    fig.add_trace(go.Scatter(x=hb.index, y=hb["pred"], name="Forecast", line=dict(color=BLUE, width=2)))
    fig.update_yaxes(title="Chiller load (%)")
    show(style(fig, 380))
    blk_m = {k: v for k, v in zip(["MAE", "R2"], [np.mean(np.abs(hb["pred"] - hb["actual"])),
             1 - np.sum((hb["pred"] - hb["actual"]) ** 2) / np.sum((hb["actual"] - hb["actual"].mean()) ** 2)])}
    st.caption(f"{MONTH_NAMES[blk]} holdout week: MAE {blk_m['MAE']:.2f} pts, R² {blk_m['R2']:.3f}"
               + (" — includes the Christmas week, the hardest period in the dataset." if blk == "2023-12" else ""))

    c3, c4 = st.columns(2)
    hh = h.resample("1h").mean().dropna(subset=["actual"]) if hourly else h
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=hh["actual"], y=hh["pred"], mode="markers", name="Holdout points",
                                   marker=dict(color=BLUE, size=6, opacity=0.45, line=dict(color="#fcfcfb", width=1))))
        lim = [0, 100]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect forecast",
                                 line=dict(color=MUTED, dash="dash", width=1.5)))
        fig.update_xaxes(title="Actual load (%)", range=lim)
        fig.update_yaxes(title="Forecast load (%)", range=lim)
        fig.update_layout(title=dict(text="Predicted vs actual", font_size=13))
        show(style(fig, 360, hover="closest"))
    with c4:
        err = (hh["pred"] - hh["actual"]).abs().groupby(hh.index.hour).mean()
        fig = go.Figure(go.Bar(x=err.index, y=err.values, marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{x}:00 — MAE %{y:.2f}<extra></extra>"))
        fig.update_xaxes(title="Hour of day", dtick=2)
        fig.update_yaxes(title="MAE (load-% points)")
        fig.update_layout(title=dict(text="Where errors happen", font_size=13))
        show(style(fig, 360, hover="closest", legend=False))

    if tier != "plant":
        sec = M["secondary"][tier]["best"]
        st.info(f"**Cooling demand (tons) model for this horizon** — holdout hourly MAE "
                f"{sec['ho_hourly_MAE']:.1f} RT, R² {sec['ho_hourly_R2']:.3f}. Used by the planner to size "
                f"chilled-water flow.")


# ==================================================================== DAY PLANNER ====
def page_planner():
    st.title("Day planner")
    st.markdown('<p class="note">Enter tomorrow\'s weather forecast and day type. The weather + calendar '
                'model forecasts the load (with an 80% band) and turns it into an hourly operating plan.</p>',
                unsafe_allow_html=True)
    with st.container(border=True):
        c = st.columns([1.1, 1, 1, 1, 1.2])
        d = c[0].date_input("Date", value=date(2024, 8, 26), min_value=date(2023, 1, 1), max_value=date(2027, 12, 31))
        td = P.typical_day(DF, d.month if d.month in (12, 3, 8) else 8)
        tmin = c[1].number_input("Min temp (°C)", 15.0, 40.0, round(td["tmin"], 1), 0.5)
        tmax = c[2].number_input("Max temp (°C)", 15.0, 45.0, round(td["tmax"], 1), 0.5)
        rh = c[3].number_input("Mean RH (%)", 30.0, 100.0, round(td["rh"], 0), 1.0)
        md_ = d.strftime("%m-%d")
        auto_festive = md_ >= "12-22" or md_ <= "01-01"
        auto_hol = d.isoformat() in C.PUBLIC_HOLIDAYS
        with c[4]:
            festive = st.checkbox("Festive / low-occupancy period", value=auto_festive)
            holiday = st.checkbox("Public holiday", value=auto_hol)
        with st.expander("Plant & tariff assumptions"):
            e = st.columns(5)
            dt = e[0].slider("Design CHW ΔT (°F)", 5.0, 12.0, C.DESIGN_DELTA_T_F, 0.5)
            mins = e[1].slider("Min pump/fan speed", 0.4, 0.9, 0.65, 0.05)
            basis = e[2].radio("Plan capacity on", ["p90", "median"], horizontal=True,
                               format_func=lambda x: "P90 (safe)" if x == "p90" else "Median")
            tariff = e[3].number_input("Tariff ($/kWh)", 0.0, 1.0, 0.12, 0.01)
            co2 = e[4].number_input("Grid CO₂ (kg/kWh)", 0.0, 1.5, 0.45, 0.05)
    if tmax < tmin:
        st.warning("Max temperature must be ≥ min temperature.")
        return
    if d.month not in (12, 3, 8):
        st.warning("Training data covers only Dec, Mar and Aug. Forecasts for other months extrapolate the "
                   "seasonal pattern — treat them as indicative.")

    wx = P.weather_profile(d, tmin, tmax, rh, SHAPES)
    fc = P.forecast_day(BUNDLES, wx, {"festive": int(festive), "is_holiday": int(holiday)})
    plan, s = P.schedule(fc, PC, delta_t_f=dt, min_speed=mins, tariff=tariff, co2_kg_kwh=co2, plan_on=basis)

    k = st.columns(5)
    k[0].metric("Peak load", f"{s['peak_load_pct']:.0f}%", f"at {s['peak_hour']}", delta_color="off")
    k[1].metric("Average load", f"{s['mean_load_pct']:.0f}%")
    k[2].metric("Cooling delivered", f"{s['cooling_ton_hours']:,.0f} ton·h")
    k[3].metric("Energy saved", f"{s['kwh_saved']:,.0f} kWh", f"−{s['pct_saved']:.1f}% plant · "
                f"−{s['aux_pct_saved']:.0f}% aux", delta_color="off")
    k[4].metric("Cost / CO₂ saved", f"${s['cost_saved']:,.0f}", f"{s['co2_saved_kg']:,.0f} kg CO₂",
                delta_color="off")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.07,
                        subplot_titles=("Forecast chiller load (%)", "Outdoor temperature (°C)"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["load_p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=fc.index, y=fc["load_p10"], fill="tonexty", fillcolor=BAND, line=dict(width=0),
                             name="80% interval"), 1, 1)
    fig.add_trace(go.Scatter(x=fc.index, y=fc["load_pred"], name="Forecast", line=dict(color=BLUE, width=2.2)), 1, 1)
    fig.add_hline(y=C.STAGE_UP_PCT, line=dict(color=MUTED, dash="dot", width=1), row=1, col=1,
                  annotation_text="2nd chiller threshold", annotation_font_color=MUTED)
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Temp"], name="Temperature", line=dict(color=ORANGE, width=1.8)), 2, 1)
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_annotations(font_size=12)
    show(style(fig, 470))

    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown("##### Plant power: today's practice vs forecast-driven")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=plan.index.hour, y=plan["Chiller kW"] + plan["Aux kW (today)"], name="Today (fixed speed)",
                             marker=dict(color=AXIS, cornerradius=3)))
        fig.add_trace(go.Bar(x=plan.index.hour, y=plan["Chiller kW"] + plan["Aux kW (planned)"], name="Planned (VFD)",
                             marker=dict(color=BLUE, cornerradius=3)))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
        fig.update_xaxes(title="Hour", dtick=2)
        fig.update_yaxes(title="Plant kW")
        show(style(fig, 360))
    with c2:
        st.markdown("##### Hourly operating plan")
        tbl = plan.copy()
        tbl.index = tbl.index.strftime("%H:%M")
        cols = ["Forecast load %", "P90 %", "Forecast RT", "Chillers on", "CHW flow (GPM)", "VFD Hz",
                "Chiller kW", "Aux kW (today)", "Aux kW (planned)"]
        st.dataframe(tbl[cols].round(1), use_container_width=True, height=360)
        st.download_button("Download plan (CSV)", plan.round(2).to_csv().encode(), f"chiller_plan_{d}.csv",
                           "text/csv")
    with st.expander("How the savings are estimated"):
        st.markdown(f"""
- **Chiller kW** = forecast load % × {C.CHILLER_KW_PER_PCT:.2f} kW (exact plant relationship found in the data).
- **Required chilled-water flow** = forecast RT × 24 / ΔT ({dt:.1f} °F) — sized on the **{'P90' if basis == 'p90' else 'median'}** forecast so the plant is not starved on a hot afternoon.
- **Pump & fan speed** = required ÷ design flow ({PC.design_gpm:,.0f} GPM), never below {mins:.0%} (chiller minimum flow / tower limits).
- **Aux power (planned)** = {PC.aux_kw_50hz:.1f} kW at 50 Hz × speed³ (affinity laws).
- **Aux power (today)** = {PC.aux_today_a:.1f} + {PC.aux_today_b:.3f} × load % kW — regressed on the plant's actual near-constant-speed operation.
- The chiller's own energy is held equal in both scenarios, so the saving is conservative (no CHW reset or staging gains counted).
""")


# ============================================================= PLANT DIGITAL TWIN ====
def page_twin():
    st.title("Plant digital twin")
    res = M["tiers"]["plant"]
    st.markdown(f'<p class="note">Physics-constrained model of the chiller: given the cooling demand and water '
                f'temperatures, what loading should the chiller need? Holdout R² '
                f'{res["best"]["ho_R2"]:.3f} at 10-min resolution. Uses no electrical inputs (no leakage). '
                f'Monotonic constraints guarantee physically sensible what-ifs.</p>', unsafe_allow_html=True)
    bundle = load_bundle("plant")
    twin = bundle["explainer"]   # monotone-constrained LightGBM

    def predict(rt, chws, dchw, cdhi, cdlo, temp, rh):
        from chiller.data import stull_wet_bulb
        gpm = rt * 24 / max(dchw, 0.1)
        row = pd.DataFrame({"RT": [rt], "GPM": [gpm], "CHWS": [chws], "DeltaCHW": [dchw], "CDHI": [cdhi],
                            "CDLO": [cdlo], "Temp": [temp], "RH": [rh],
                            "WBT": [float(stull_wet_bulb(temp, rh))]},
                           index=pd.DatetimeIndex([pd.Timestamp("2024-08-01 14:00")]))
        X = plant_features(row)[bundle["features"]]
        return float(twin.predict(X)[0])

    with st.container(border=True):
        c = st.columns(7)
        rt = c[0].number_input("Cooling demand (RT)", 50.0, 560.0, 350.0, 10.0)
        chws = c[1].number_input("CHW supply (°F)", 42.0, 52.0, 45.5, 0.5)
        dchw = c[2].number_input("CHW ΔT (°F)", 1.0, 12.0, 6.0, 0.5)
        cdhi = c[3].number_input("Cond. leaving (°F)", 75.0, 100.0, 88.0, 0.5)
        cdlo = c[4].number_input("Cond. entering (°F)", 70.0, 95.0, 82.0, 0.5)
        temp = c[5].number_input("Outdoor temp (°C)", 20.0, 40.0, 31.0, 0.5)
        rh = c[6].number_input("Outdoor RH (%)", 30.0, 100.0, 70.0, 1.0)
    if cdlo >= cdhi:
        st.warning("Condenser entering temperature should be below leaving temperature.")
    load = predict(rt, chws, dchw, cdhi, cdlo, temp, rh)
    kw = load * C.CHILLER_KW_PER_PCT
    k = st.columns(4)
    k[0].metric("Expected chiller load", f"{load:.1f}%")
    k[1].metric("Expected chiller power", f"{kw:.0f} kW")
    k[2].metric("Chiller efficiency", f"{kw / rt:.3f} kW/RT")
    k[3].metric("Compressor lift", f"{cdhi - chws:.1f} °F")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### What-if: chilled-water supply reset")
        xs = np.arange(42.0, 50.5, 0.5)
        ys = np.array([predict(rt, x, dchw, cdhi, cdlo, temp, rh) for x in xs]) * C.CHILLER_KW_PER_PCT
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Chiller kW",
                                   line=dict(color=BLUE, width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2))))
        fig.add_vline(x=chws, line=dict(color=MUTED, dash="dot"), annotation_text="current",
                      annotation_font_color=MUTED)
        fig.update_xaxes(title="CHW supply setpoint (°F)")
        fig.update_yaxes(title="Chiller kW")
        show(style(fig, 340, hover="closest", legend=False))
        per_f = (ys[0] - ys[-1]) / (xs[-1] - xs[0])
        st.caption(f"≈ {per_f:.1f} kW less per °F of CHW reset at {rt:.0f} RT "
                   f"({per_f / max(kw, 1) * 100:.1f}% of chiller power per °F). Industry rule of thumb: 1–2%/°F.")
    with c2:
        st.markdown("##### Efficiency monitoring on holdout weeks")
        h = load_holdout("plant")
        dres = (h["actual"] - h["pred"]).resample("D").mean().dropna()
        thr = 2 * (h["actual"] - h["pred"]).resample("D").mean().std()
        colors = [CRIT if v > thr else (GOOD if v < -thr else "#86b6ef") for v in dres.values]
        fig = go.Figure(go.Bar(x=[i.strftime("%d %b %y") for i in dres.index], y=dres.values,
                               marker=dict(color=colors, cornerradius=3),
                               hovertemplate="%{x}: %{y:+.2f} pts<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=AXIS))
        fig.update_xaxes(type="category", tickangle=-60)
        fig.update_yaxes(title="Actual − expected load (pts)")
        show(style(fig, 340, hover="closest", legend=False))
        st.caption(f"Days where the chiller works harder than the twin expects (> +{thr:.1f} pts, red) flag "
                   "possible fouling, refrigerant or sensor faults; green = better than expected.")


# =================================================================== METHODOLOGY ====
def page_method():
    st.title("Methodology")
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#fcfcfb", color="#c3c2b7", fontname="Segoe UI", fontsize=11];
      edge [color="#898781"];
      a [label="BMS plant log\\n13,330 × 31 (10-min)"]; b [label="Weather station\\nTemp · RH · WBT"];
      c [label="Clean & align\\n10-min grid · dedupe\\ntime-interpolate weather\\ndrop compressor-off"];
      d [label="Features\\npsychrometrics · inertia\\ncalendar/occupancy\\nhorizon-safe lags"];
      e [label="Purged week-blocked CV\\n9 models + tuning\\n+ top-3 ensemble"];
      f [label="Holdout test\\nlast 7 days / month"]; g [label="Conformal\\n80% intervals"];
      h [label="Deployed models\\n(refit on all data)"]; i [label="Planner · twin\\ndashboard"];
      a -> c; b -> c; c -> d -> e -> f; e -> g -> h; f -> h -> i; }
    """, use_container_width=True)
    st.markdown(f"""
#### Problem framing
The goal is to **estimate chiller load ahead of time** from the factors named in the brief — weather, day of week,
occupancy and local events — so the plant can be run at the capacity actually needed. The dataset has no direct
occupancy feed, so occupancy is represented by calendar signals known in advance: time of day, weekday/weekend,
public holidays, a festive-period flag and month (season).

Two quantities are modelled: **CH Load** (chiller loading, % of full load — primary) and **RT** (refrigeration tons
of cooling delivered — used to size water flow).

#### Why the earlier notebooks were not usable
* Inputs `kW_Tot` and `GPM × ΔCHW` *are* the answer: `CH Load = kW_CHH × 0.2947` and `RT = GPM × ΔT / 24` exactly.
  A model fed those has R² ≈ 1 and zero forecasting value.
* The CSV weather timestamps were corrupted, so the merge silently produced mean-imputed weather.
* A random 80/20 split on 10-minute data leaks neighbouring samples into the test set.

#### Horizons
| Model | Inputs | When to use |
|---|---|---|
| Weather + calendar | weather forecast, calendar | any day, no plant history |
| Day-ahead | + load & plant history ≥ 24 h old | plan tomorrow |
| Hour-ahead | + load & plant history ≥ 1 h old | intraday dispatch |
| Plant digital twin | RT, water & condenser temperatures, weather — no kW | what-if, fault detection |

#### Validation
* **Holdout:** the last {M['config']['holdout_days_per_month']} days of each month, never used for fitting or model choice.
* **Cross-validation:** {M['config']['cv_folds']}-fold, whole calendar weeks held out together, with a {M['config']['purge']} purge on both
  sides so lag features cannot see test targets.
* **Metrics:** MAE, RMSE, R², CV(RMSE) and NMBE (ASHRAE Guideline 14), at 10-min and hourly resolution.
* **Noise floor:** the chiller short-cycles — even a perfect hourly forecast leaves ≈3.3 pts MAE at 10-min
  resolution, so hourly metrics are the planning-relevant ones.
* **Intervals:** quantile LightGBM (P10/P90) calibrated by split-conformal adjustment on out-of-fold residuals.

#### Limitations
Only three months of data (Dec, Mar, Aug) and one holiday season. Forecasts for other months extrapolate.
A real occupancy feed (room nights, events) would be the single biggest accuracy upgrade.
""")
    with st.expander("Data audit (raw)"):
        st.json(M["audit"])


{"Overview": page_overview, "Data & insights": page_data, "Model performance": page_models,
 "Day planner": page_planner, "Plant digital twin": page_twin, "Methodology": page_method}[page]()
