"""End-to-end training pipeline.

    python train.py            # full run (model zoo + tuning + ensembles + intervals)
    python train.py --fast     # fewer tuning trials

Outputs (artifacts/):
    dataset.parquet                 cleaned 10-minute grid used by the dashboard
    models/<tier>__<target>.joblib  deployed models (refit on all data)
    reports/metrics.json            leaderboards, holdout metrics, selections, audit
    reports/holdout_<tier>.parquet  holdout predictions (from train-only fits)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from chiller import config as C
from chiller import data, features as F, modeling as M

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(C.SEED)
ENSEMBLE_TOL = 0.01          # prefer the ensemble unless a single model is >1 % better on CV
PRIMARY_TIERS = ["exogenous", "day_ahead", "hour_ahead", "plant"]
SECONDARY_TIERS = ["exogenous", "day_ahead", "hour_ahead"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- tuning --
HAND_CANDIDATES = [
    {},
    dict(num_leaves=31, min_child_samples=100, learning_rate=0.02, n_estimators=1500,
         colsample_bytree=0.5, reg_lambda=10),
    dict(objective="l1", num_leaves=15, min_child_samples=60, learning_rate=0.03,
         n_estimators=1500, colsample_bytree=0.6),
    dict(objective="huber", num_leaves=15, min_child_samples=60, learning_rate=0.03,
         n_estimators=1500, colsample_bytree=0.6),
]


def sample_params():
    return dict(
        objective=str(RNG.choice(["regression", "huber", "l1"])),
        num_leaves=int(RNG.choice([7, 15, 31, 63])),
        min_child_samples=int(RNG.choice([20, 40, 80, 150])),
        learning_rate=float(RNG.choice([0.015, 0.03, 0.05])),
        n_estimators=int(RNG.choice([600, 1000, 1500, 2500])),
        colsample_bytree=float(RNG.choice([0.4, 0.6, 0.8])),
        subsample=float(RNG.choice([0.7, 0.85, 1.0])),
        reg_lambda=float(RNG.choice([0.0, 1.0, 5.0, 15.0])),
    )


def make_lgbm(params: dict, y_scale: float, monotone: list | None = None):
    p = dict(params)
    if monotone is not None and p.get("objective") == "l1":
        p["objective"] = "huber"                         # l1 does not support monotone constraints
    if p.get("objective") == "huber":
        p["alpha"] = 0.5 * y_scale                       # huber transition in target units
    if monotone is not None:
        p["monotone_constraints"] = monotone
        p["monotone_constraints_method"] = "advanced"
    return M.lgbm(**p)


def tune(X, y, n_trials, monotone=None):
    trials = HAND_CANDIDATES + [sample_params() for _ in range(n_trials)]
    best, rows = None, []
    scale = float(y.std())
    for i, p in enumerate(trials):
        r, oof = M.cross_validate(make_lgbm(p, scale, monotone), X, y)
        rows.append({**p, "MAE": r["MAE"], "R2": r["R2"]})
        if best is None or r["MAE"] < best[1]["MAE"]:
            best = (p, r, oof)
    log(f"   tuning: {len(trials)} trials, best CV MAE {best[1]['MAE']:.3f}  {best[0]}")
    return best, rows


# --------------------------------------------------------------------------- per tier -
def rows_for(df, X, tier):
    m = df["valid_target"].copy()
    if tier == "plant":
        m &= X[F.PLANT_INPUTS].notna().all(axis=1) & df["RT"].gt(0)
    return m


def run_tier(df, ho, tier, target, n_trials, full_zoo=True):
    X = F.build_features(df, tier, target)
    use = rows_for(df, X, tier)
    tr, te = use & ~ho, use & ho
    Xtr, ytr, Xte, yte = X[tr], df.loc[tr, target], X[te], df.loc[te, target]
    log(f"[{tier} | {target}] train {len(Xtr)}  holdout {len(Xte)}  features {X.shape[1]}")

    monotone = None
    if tier == "plant":
        monotone = [F.PLANT_MONOTONE.get(c, 0) for c in X.columns]

    zoo = M.model_zoo(tier) if full_zoo else {}
    zoo.pop("LightGBM", None)
    board, oofs, hold, fitted = [], {}, {}, {}

    def evaluate(name, model, cv=None):
        if cv is None:
            cv = M.cross_validate(model, Xtr, ytr)
        r_cv, oof = cv
        fm, p = M.fit_predict(model, Xtr, ytr, Xte)
        oofs[name], hold[name], fitted[name] = oof, p, model
        board.append({"model": name,
                      **{f"cv_{k}": v for k, v in r_cv.items()},
                      **{f"cv_hourly_{k}": v for k, v in M.hourly_metrics(Xtr.index, ytr, oof).items()},
                      **{f"ho_{k}": v for k, v in M.metrics(yte, p).items()},
                      **{f"ho_hourly_{k}": v for k, v in M.hourly_metrics(Xte.index, yte, p).items()}})
        log(f"   {name:34s} CV MAE {r_cv['MAE']:6.3f} R2 {r_cv['R2']:.3f} | "
            f"holdout MAE {board[-1]['ho_MAE']:6.3f} R2 {board[-1]['ho_R2']:.3f}")

    for name, model in zoo.items():
        evaluate(name, model)

    (best_p, best_r, best_oof), trials = tune(Xtr, ytr, n_trials, monotone)
    tuned = make_lgbm(best_p, float(ytr.std()), monotone)
    evaluate("LightGBM (tuned)", tuned, (best_r, best_oof))

    # ensemble of the three best learned models (by CV MAE)
    learned = [b for b in board if not b["model"].startswith("Baseline")]
    top = [b["model"] for b in sorted(learned, key=lambda b: b["cv_MAE"])[:3]]
    if len(top) >= 2:
        ens_name = "Ensemble: " + " + ".join(top)
        oof = np.mean([oofs[n] for n in top], axis=0)
        p = np.mean([hold[n] for n in top], axis=0)
        oofs[ens_name], hold[ens_name] = oof, p
        fitted[ens_name] = M.EnsembleRegressor([fitted[n] for n in top])
        board.append({"model": ens_name,
                      **{f"cv_{k}": v for k, v in M.metrics(ytr, oof).items()},
                      **{f"cv_hourly_{k}": v for k, v in M.hourly_metrics(Xtr.index, ytr, oof).items()},
                      **{f"ho_{k}": v for k, v in M.metrics(yte, p).items()},
                      **{f"ho_hourly_{k}": v for k, v in M.hourly_metrics(Xte.index, yte, p).items()}})
        log(f"   {ens_name:34s} CV MAE {board[-1]['cv_MAE']:6.3f} | holdout MAE {board[-1]['ho_MAE']:6.3f}"
            f" R2 {board[-1]['ho_R2']:.3f}")

    # model selection strictly on CV (holdout is only for reporting). Averaging cuts variance,
    # so the ensemble is preferred unless a single model beats it on CV by > ENSEMBLE_TOL.
    singles = [b for b in board if not b["model"].startswith(("Baseline", "Ensemble"))]
    best_single = min(singles, key=lambda b: b["cv_MAE"])
    ens = [b for b in board if b["model"].startswith("Ensemble")]
    if ens and ens[0]["cv_MAE"] <= best_single["cv_MAE"] * (1 + ENSEMBLE_TOL):
        best_name = ens[0]["model"]
    else:
        best_name = best_single["model"]
    log(f"   -> selected: {best_name}")

    # ---- prediction intervals: quantile LightGBM + split-conformal calibration on OOF
    qp = {k: v for k, v in best_p.items() if k != "objective"}
    q_models = {a: M.lgbm(**qp, objective="quantile", alpha=a) for a in (0.1, 0.9)}
    q_oof = {a: M.cross_validate(q_models[a], Xtr, ytr)[1] for a in q_models}
    scores = np.maximum(q_oof[0.1] - ytr.values, ytr.values - q_oof[0.9])
    q_adj = float(np.quantile(scores, 0.8))
    q_hold = {a: M.fit_predict(q_models[a], Xtr, ytr, Xte)[1] for a in q_models}
    lo, hi = q_hold[0.1] - q_adj, q_hold[0.9] + q_adj
    coverage = float(np.mean((yte.values >= lo) & (yte.values <= hi)) * 100)
    log(f"   80% interval: conformal adj {q_adj:+.2f}, holdout coverage {coverage:.1f}%,"
        f" mean width {np.mean(hi - lo):.1f}")

    # ---- explanations (mean |SHAP| on holdout from the tuned LightGBM, train-only fit)
    lg = M.fit_predict(tuned, Xtr, ytr, Xte)[0]
    contrib = lg.predict(Xte, pred_contrib=True)[:, :-1]
    shap_imp = (pd.Series(np.abs(contrib).mean(0), index=X.columns)
                .sort_values(ascending=False))

    # ---- holdout predictions frame for the dashboard
    hframe = pd.DataFrame({"actual": yte.values, "pred": hold[best_name], "p10": lo, "p90": hi},
                          index=Xte.index)
    for n, p in hold.items():
        hframe[f"m::{n}"] = p

    # ---- deployment: refit selected + quantile models on ALL labelled rows
    Xall, yall = X[use], df.loc[use, target]
    deployed = M.fit_predict(fitted[best_name], Xall, yall, Xall.iloc[:1])[0]
    deployed_q = {a: M.fit_predict(q_models[a], Xall, yall, Xall.iloc[:1])[0] for a in q_models}
    joblib.dump({"tier": tier, "target": target, "features": list(X.columns),
                 "model": deployed, "q10": deployed_q[0.1], "q90": deployed_q[0.9],
                 "q_adj": q_adj, "explainer": M.lgbm_component(deployed) or
                 M.fit_predict(tuned, Xall, yall, Xall.iloc[:1])[0],
                 "name": best_name, "trained_rows": int(len(Xall))},
                C.MODELS_DIR / f"{tier}__{target.replace(' ', '_')}.joblib", compress=3)

    best_row = next(b for b in board if b["model"] == best_name)
    return {
        "tier": tier, "target": target, "selected": best_name,
        "n_train": int(len(Xtr)), "n_holdout": int(len(Xte)), "n_features": int(X.shape[1]),
        "features": list(X.columns), "leaderboard": board, "best": best_row,
        "tuning_trials": trials, "best_params": best_p,
        "interval": {"nominal": 80, "holdout_coverage": coverage, "conformal_adj": q_adj,
                     "mean_width": float(np.mean(hi - lo))},
        "shap_importance": shap_imp.round(4).to_dict(),
    }, hframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--tiers", nargs="+", choices=PRIMARY_TIERS,
                    help="retrain only these CH Load tiers and merge into the existing metrics.json")
    args = ap.parse_args()
    n_trials = 6 if args.fast else 20
    primary = args.tiers or PRIMARY_TIERS
    secondary = [] if args.tiers else SECONDARY_TIERS
    t0 = time.time()
    for d in (C.MODELS_DIR, C.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log("building dataset ...")
    df, audit = data.build_dataset()
    df.to_parquet(C.ARTIFACTS / "dataset.parquet")
    ho = M.holdout_mask(df)
    log(f"grid {len(df)} rows, labelled {int(df.valid_target.sum())}, "
        f"holdout {int((ho & df.valid_target).sum())}")

    results = {"audit": audit, "config": {
        "holdout_days_per_month": C.HOLDOUT_DAYS, "cv_folds": C.CV_FOLDS, "purge": C.PURGE,
        "holdout_windows": {b: [str(g.index.min()), str(g.index.max())]
                            for b, g in df[ho & df.observed].groupby("block")}},
        "tiers": {}, "secondary": {}}
    metrics_path = C.REPORTS_DIR / "metrics.json"
    if args.tiers and metrics_path.exists():                 # partial retrain: keep other tiers
        prev = json.loads(metrics_path.read_text())
        results["tiers"], results["secondary"] = prev["tiers"], prev["secondary"]

    for tier in primary:
        res, hframe = run_tier(df, ho, tier, C.TARGET, n_trials)
        results["tiers"][tier] = res
        hframe.to_parquet(C.REPORTS_DIR / f"holdout_{tier}.parquet")

    for tier in secondary:                                   # cooling tons, for pump/flow planning
        res, hframe = run_tier(df, ho, tier, C.TARGET_RT, max(4, n_trials // 3), full_zoo=False)
        results["secondary"][tier] = {k: res[k] for k in
                                      ("selected", "best", "interval", "n_train", "n_holdout")}
        hframe.to_parquet(C.REPORTS_DIR / f"holdout_{tier}__RT.parquet")

    results["runtime_min"] = round((time.time() - t0) / 60, 1)
    results["generated"] = time.strftime("%Y-%m-%d %H:%M")
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    lb = pd.concat([pd.DataFrame(r["leaderboard"]).assign(tier=t) for t, r in results["tiers"].items()])
    lb.to_csv(C.REPORTS_DIR / "leaderboard.csv", index=False)
    log(f"done in {results['runtime_min']} min")


if __name__ == "__main__":
    main()
