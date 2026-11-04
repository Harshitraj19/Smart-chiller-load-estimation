# Smart Chiller Load Estimation

**Problem.** A hotel's chiller load depends on occupancy, outdoor weather, day of week and nearby
attractions. The goal is to estimate that load *in advance*, so the plant can run the capacity it
actually needs instead of running equipment at full capacity when it isn't required.

**Solution.** A leakage-free forecasting system at three horizons plus a physics-constrained plant
model. Models are validated with purged, week-blocked cross-validation and a held-out test set, and
everything is served through an operations dashboard that turns forecasts into an hourly equipment
plan with an energy-savings estimate.

---

## Results

Holdout = the **last 7 days of each month** (Dec-23, Mar-24, Aug-24, 2,904 ten-minute samples).
These rows were never used for training, tuning or model selection. Errors are in chiller-load
percentage points (CH Load runs 0–95 %). "Hourly" = 10-minute forecasts averaged to the hour, which
is the planning interval and the ASHRAE Guideline 14 reference interval.

| Model | Inputs | Selected algorithm | Hourly R² | Hourly MAE | 10-min R² | Hourly CV(RMSE) | Best baseline (hourly MAE) |
|---|---|---|---|---|---|---|---|
| **Hour-ahead** | weather, calendar, load & plant history ≥ 1 h old | Ensemble (LightGBM + ExtraTrees + RF) | **0.932** | **3.64** | 0.879 | 8.6 % | persistence: 4.63 |
| **Day-ahead** | weather, calendar, history ≥ 24 h old | Ensemble (ExtraTrees + RF + LightGBM) | **0.862** | **5.50** | 0.821 | 12.2 % | same time yesterday: 5.94 |
| **Weather + calendar** | weather forecast + calendar only | Extra Trees | **0.843** | **5.77** | 0.800 | 13.1 % | typical-week profile: 10.78 |
| **Plant digital twin** | cooling tons, water/condenser temps, weather (no kW) | Ensemble (LightGBM + RF + ExtraTrees) | **0.976** | **2.15** | 0.931 | 5.2 % | — |

* All four pass **ASHRAE Guideline 14** calibration criteria (hourly CV(RMSE) < 30 %, |NMBE| < 10 %).
* Cooling-demand (tons) models for flow sizing: hourly R² 0.928 (hour-ahead), 0.843 (day-ahead), 0.807
  (weather only).
* **Energy:** on a typical August day the planner estimates **~840 kWh/day saved (≈ 13 % of
  chiller-plant energy, 46 % of pump + tower-fan energy)** by running pumps and fans at the speed the
  forecast load needs, instead of the near-constant 47–48 Hz they run at today. Every assumption is
  editable in the dashboard.
* **Plant twin sanity check:** it predicts 1.8 % chiller power saved per °F of chilled-water reset, inside
  the 1–2 %/°F industry rule of thumb. Monotonic constraints keep its what-ifs physically consistent.

### Noise floor
The chiller short-cycles around its hourly trend. Even a *perfect* hourly forecast leaves ≈ 3.3 points of
MAE at 10-minute resolution, so hourly metrics are the meaningful ones for planning.

---

## What the data audit found (why the earlier notebooks' 99.6 % R² was not real)

| Finding | Evidence | Action |
|---|---|---|
| **Target leakage**: `CH Load` is literally chiller power | `CH Load = kW_CHH × 0.2947` (corr 0.99999); `kW_Tot` includes `kW_CHH` | No `kW_*` / `Precent_*` inputs anywhere |
| `RT` is a formula of flow × ΔT | `RT = GPM × ΔCHW / 24` (corr 0.9999999) | `GPM × ΔT` only used in the plant twin, never in forecasts |
| **Corrupted timestamps** in `sensor_data.csv` | times like `00:42.0` (Excel mm:ss); the old outer merge matched 0 rows and mean-imputed all weather | Rebuilt from the original `.xlsx`; weather time-interpolated onto the 10-min plant grid |
| Plant WBT probe stuck | 72.7 °F for 80 % of readings | Weather-station WBT used (matches Stull's formula to ±0.1 °C) |
| 230 compressor-off / dropout rows | load = 0 and flow = 0 while pumps run; mostly isolated single readings | Excluded as targets, not imputed |
| Random 80/20 split on 10-min data | neighbouring samples leak into test | Purged week-blocked CV + chronological holdout |
| Constant-speed auxiliaries | pumps/fans at 46–48 Hz at every load; ΔCHW drops to 3.5 °F at low load | The source of the forecastable savings |

---

## Approach

1. **Data** (`chiller/data.py`) – load both `.xlsx` files, dedupe, place them on a regular 10-minute grid
   (gaps stay as gaps so lags are correct), interpolate weather, flag compressor-off rows.
2. **Features** (`chiller/features.py`)
   * Weather: temperature, RH, wet-bulb, dew point, humidity ratio, enthalpy, and trailing 1/3/6/24 h means
     to capture building thermal inertia.
   * Calendar / occupancy proxies: time of day (cyclic), weekday/weekend, occupied hours, public holiday,
     festive period (Christmas–New Year), month (season). The data has no occupancy feed, so these stand in
     for it. The Christmas week shows a ~50 % load drop with no weather change.
   * Load & plant history, always at least one horizon old: lags, rolling means, same-time-yesterday /
     last-week windows, lagged RT and return-water temperature.
3. **Models** (`chiller/modeling.py`) – typical-week and naive baselines, Ridge, Random Forest, Extra Trees,
   HistGradientBoosting, XGBoost, LightGBM (24-config random search), and a top-3 ensemble.
4. **Validation** – 5-fold CV where whole calendar weeks are held out together, with a 1-day purge on both
   sides. Models are selected on CV only; the ensemble is preferred unless a single model beats it by more
   than 1 % on CV (averaging lowers variance). *This tie-break rule was added after the first full run,
   where Extra Trees beat the ensemble on day-ahead CV by 0.0005 MAE but generalised worse. It is applied
   identically to every tier, and for the weather-only tier it still selects the single model.*
5. **Uncertainty** – quantile LightGBM (P10/P90), widened by split-conformal calibration on out-of-fold
   residuals. Holdout coverage is 67–80 % against the 80 % target, because the holdout weeks (Christmas,
   the late-March peak) are harder than average. Treat P90 as a planning value, not a guaranteed ceiling.
6. **Explainability** – SHAP contributions from LightGBM. The top drivers are time of day / occupied hours
   and temperature for the weather-only model, and recent load and cooling tons for the hour-ahead model.
7. **Planner** (`chiller/planner.py`) – turns a forecast into hourly chiller staging, chilled-water flow,
   VFD speed (affinity laws, minimum-speed floor) and kWh / cost / CO₂ versus today's operation.

---

## Run it

```bash
pip install -r requirements.txt
python train.py                       # full pipeline, ~40 min on 16 cores (writes artifacts/)
python train.py --fast                # quicker, fewer tuning trials
python train.py --tiers day_ahead     # retrain one tier, merge into existing reports
python -m streamlit run app.py        # dashboard at http://localhost:8501
```

> `python -m streamlit` works even when pip did a user install and the `streamlit` command isn't on
> your PATH (common on Windows: "The term 'streamlit' is not recognized").

Trained models are included in `artifacts/`, so the dashboard runs without retraining.

### Dashboard pages
| Page | What it does |
|---|---|
| Overview | headline accuracy, savings, audit findings, holdout forecast vs actual |
| Data & insights | load and weather time series, hour × weekday heatmap, monthly profiles, weather sensitivity, the constant-speed finding, the occupancy (Christmas) signal |
| Model performance | per-horizon leaderboard (CV + holdout, 10-min or hourly), ASHRAE G14 badges, SHAP, forecast vs actual with intervals, error by hour |
| Day planner | enter a date, min/max temperature, RH and day type, then get a 24 h forecast with an 80 % band, hourly operating plan, energy/cost/CO₂ savings, and a CSV download |
| Plant digital twin | expected chiller load/kW for any operating point, chilled-water reset what-if, holdout efficiency monitoring (fault-detection flags) |
| Methodology | pipeline diagram, validation design, limitations |

---

## Project layout

```
chiller/            config.py · data.py · features.py · modeling.py · planner.py
train.py            training / evaluation / deployment pipeline
app.py              Streamlit dashboard
artifacts/          dataset.parquet · models/*.joblib · reports/metrics.json, leaderboard.csv, holdout_*.parquet
rfr_data/           original data (table_data.xlsx = plant BMS log, sensor_data.xlsx = weather station)
chiller_rfr.ipynb, prediction2.ipynb, streamlit_prediction2.py, *.csv   earlier prototype (kept for reference)
```

## Limitations and next steps
* Only **three non-contiguous months** (Dec, Mar, Aug) and **one** holiday season. Forecasts for other months
  extrapolate the seasonal pattern.
* No direct occupancy data. A room-nights / events feed is the single biggest accuracy upgrade available,
  especially for holiday periods.
* The savings model is first-order (affinity laws, constant chiller efficiency). Validate it against a VFD
  trial before committing capital.
