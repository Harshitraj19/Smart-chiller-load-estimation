"""Model zoo, metrics and leakage-safe validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


# ----------------------------------------------------------------------- metrics ----
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": rmse,
        "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        "MAPE": float(np.mean(np.abs(e) / np.clip(np.abs(y), 1e-6, None)) * 100),
        "CV_RMSE": rmse / float(y.mean()) * 100,           # ASHRAE G14: <30 % hourly
        "NMBE": float(e.sum() / (len(y) * y.mean()) * 100),  # ASHRAE G14: within +-10 %
        "Within5": float(np.mean(np.abs(e) <= 5) * 100),    # % of points within +-5 load-pts
    }


def hourly_metrics(index: pd.DatetimeIndex, y, p) -> dict:
    """Metrics after averaging to hourly resolution (the ASHRAE G14 reference interval)."""
    f = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}, index=index).resample("1h").mean().dropna()
    return metrics(f["y"], f["p"])


# ------------------------------------------------------------------------ models ----
class ProfileBaseline(BaseEstimator, RegressorMixin):
    """Average load for each (day-of-week, hour) slot — the 'typical week' a plant operator uses."""
    def fit(self, X, y):
        key = X["dow"].astype(int) * 24 + X["hour"].astype(int)
        self.table_ = pd.Series(np.asarray(y), index=key.values).groupby(level=0).mean()
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        key = X["dow"].astype(int) * 24 + X["hour"].astype(int)
        return key.map(self.table_).fillna(self.mean_).to_numpy()


class ColumnBaseline(BaseEstimator, RegressorMixin):
    """Naive forecast that copies a lag column (persistence / same-time-yesterday)."""
    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X[self.column].fillna(self.mean_).to_numpy()


def lgbm(**kw):
    p = dict(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=C.SEED, verbose=-1, n_jobs=-1)
    p.update(kw)
    return lgb.LGBMRegressor(**p)


def model_zoo(tier: str) -> dict:
    zoo = {
        "Ridge (linear)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5,
                                               n_jobs=-1, random_state=C.SEED),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.7,
                                           n_jobs=-1, random_state=C.SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=800, learning_rate=0.04,
                                                              l2_regularization=1.0, random_state=C.SEED),
        "XGBoost": xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                                    random_state=C.SEED, n_jobs=-1, tree_method="hist"),
        "LightGBM": lgbm(),
    }
    if tier != "plant":
        zoo = {"Baseline: typical week profile": ProfileBaseline(), **zoo}
    if tier == "day_ahead":
        zoo = {"Baseline: same time yesterday": ColumnBaseline("y_lag_24h"), **zoo}
    if tier == "hour_ahead":
        zoo = {"Baseline: persistence (1 h ago)": ColumnBaseline("y_lag_60m"), **zoo}
    return zoo


def needs_impute(model) -> bool:
    return isinstance(model, (RandomForestRegressor, ExtraTreesRegressor))


def fit_predict(model, Xtr, ytr, Xte):
    m = clone(model)
    if needs_impute(m):
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), m)
    m.fit(Xtr, ytr)
    return m, m.predict(Xte)


# -------------------------------------------------------------------- validation ----
def holdout_mask(df: pd.DataFrame) -> pd.Series:
    """Last HOLDOUT_DAYS of every monthly block -> untouched final test set."""
    obs = df[df["observed"]]
    end = obs.index.to_series().groupby(obs["block"]).max()
    cut = df["block"].map(end) - pd.Timedelta(days=C.HOLDOUT_DAYS)
    return pd.Series(df.index > cut.values, index=df.index)


def purged_week_folds(index: pd.DatetimeIndex, n_folds: int = C.CV_FOLDS):
    """Blocked CV: whole calendar weeks are held out together and training rows within
    PURGE of any test week are removed, so lag features cannot see test targets."""
    week = index.to_period("W-SUN").astype(str)
    weeks = np.array(sorted(set(week)))
    rng = np.random.default_rng(C.SEED)
    assign = dict(zip(weeks, rng.permutation(len(weeks)) % n_folds))
    fold = np.array([assign[w] for w in week])
    purge = pd.Timedelta(C.PURGE)
    t = index.values
    for k in range(n_folds):
        te = fold == k
        keep = ~te
        for w in weeks[[assign[w] == k for w in weeks]]:
            sel = week == w
            lo, hi = t[sel].min() - purge, t[sel].max() + purge
            keep &= ~((t >= lo) & (t <= hi))
        yield np.where(keep)[0], np.where(te)[0]


def cross_validate(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict, np.ndarray]:
    oof = np.full(len(y), np.nan)
    for tr, te in purged_week_folds(X.index):
        _, oof[te] = fit_predict(model, X.iloc[tr], y.iloc[tr], X.iloc[te])
    return metrics(y, oof), oof


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Equal-weight average of fitted member models (members fitted via fit_predict)."""
    def __init__(self, members: list):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [fit_predict(m, X, y, X.iloc[:1])[0] for m in self.members]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.fitted_], axis=0)


def lgbm_component(model):
    """Return the LightGBM booster inside a deployed model (for SHAP explanations)."""
    if isinstance(model, lgb.LGBMRegressor):
        return model
    for m in getattr(model, "fitted_", []):
        if isinstance(m, lgb.LGBMRegressor):
            return m
    return None
