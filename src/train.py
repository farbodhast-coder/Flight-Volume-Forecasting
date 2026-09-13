"""
flight_forecast/src/train.py
─────────────────────────────────────────────────────────────────────────────
European Airport Flight-Volume Forecasting — XGBoost Pipeline
Author : <your-name>
License: MIT  (see LICENSE in project root)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

TARGET: str = "FLT_TOT_1"
DROP_ALWAYS: list[str] = ["YEAR", "FLT_DATE", "MONTH_MON"]  # temporal / text leakage
LEAKAGE_COLS: list[str] = ["FLT_TOT_IFR_2", "FLT_DEP_1", "FLT_ARR_1",
                            "FLT_DEP_IFR_2", "FLT_ARR_IFR_2"]
CATEGORICAL_COLS: list[str] = ["APT_ICAO", "APT_NAME", "STATE_NAME", "Pivot Label"]

XGB_PARAMS: dict = {
    "n_estimators":      200,
    "learning_rate":     0.1,
    "max_depth":         5,
    "min_child_weight":  3,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "early_stopping_rounds": 50,
    "eval_metric":       "rmse",
    "n_jobs":            -1,
}

XGB_PARAMS_CLEAN: dict = {
    "n_estimators":  100,
    "max_depth":     3,
    "random_state":  42,
    "n_jobs":        -1,
}

# ── Data helpers ─────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV and return a raw DataFrame."""
    log.info("Loading data from: %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the raw DataFrame:
    - Fill NaN with 0
    - Encode known categorical columns (if present)
    - Drop text / date columns not used for modelling
    - Keep only numeric columns
    """
    log.info("Preprocessing …")
    df = df.copy()
    df.fillna(0, inplace=True)

    present_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in present_cats:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        log.debug("  Encoded column: %s", col)

    # Drop pure-text columns that won't encode usefully
    drop_text = [c for c in DROP_ALWAYS if c in df.columns]
    df.drop(columns=drop_text, inplace=True)

    # Keep only numeric columns (safety net)
    df = df.select_dtypes(include=[np.number])
    log.info("Features after preprocessing: %d", df.shape[1] - 1)
    return df


def split_features(df: pd.DataFrame,
                   drop_leakage: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). Optionally remove data-leakage columns."""
    extra_drop = LEAKAGE_COLS if drop_leakage else []
    cols_to_drop = [c for c in ([TARGET] + extra_drop) if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]
    return X, y

# ── Training & evaluation ────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                X_val: pd.DataFrame,
                y_val: pd.Series,
                params: dict) -> xgb.XGBRegressor:
    """Fit an XGBRegressor and return it."""
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate(model: xgb.XGBRegressor,
             X_test: pd.DataFrame,
             y_test: pd.Series,
             label: str = "Model") -> dict[str, float]:
    """Predict, compute metrics, log results, and return a metrics dict."""
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    _log_metrics(label, metrics)
    return metrics


def _log_metrics(label: str, m: dict[str, float]) -> None:
    bar = "─" * 44
    log.info(bar)
    log.info("  %-30s", label)
    log.info(bar)
    log.info("  R²   : %8.4f", m["r2"])
    log.info("  MAE  : %12,.0f  flights", m["mae"])
    log.info("  RMSE : %12,.0f  flights", m["rmse"])
    log.info(bar)

# ── Feature analysis ─────────────────────────────────────────────────────────

def top_correlations(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """Return the top-n features most correlated with the target."""
    corr = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    log.info("Top %d correlations with %s:", n, TARGET)
    for feat, val in corr.head(n).items():
        log.info("  %-30s %.4f", feat, val)
    return corr.head(n)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model to forecast European airport flight volumes."
    )
    parser.add_argument(
        "data",
        metavar="DATA_CSV",
        help="Path to the input CSV (e.g. e_f_d.csv)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, metavar="FRAC",
        help="Fraction of data held out for testing (default: 0.20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, metavar="K",
        help="Number of top-correlation features to display (default: 10)",
    )
    return parser.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()

    # 1 — Load & preprocess
    raw = load_data(args.data)
    df  = preprocess(raw)

    # 2 — Correlation analysis (before leakage removal)
    top_correlations(df, n=args.top_k)

    # ── Model A: Full feature set (includes correlated flight columns) ────────
    log.info("=== Model A: All features ===")
    X, y = split_features(df, drop_leakage=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    model_a = train_model(X_train, y_train, X_test, y_test, XGB_PARAMS)
    evaluate(model_a, X_test, y_test, label="Model A — Full features")

    # ── Model B: Leakage-free feature set ────────────────────────────────────
    log.info("=== Model B: Leakage-free features ===")
    X_clean, y_clean = split_features(df, drop_leakage=True)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clean, y_clean, test_size=args.test_size, random_state=args.seed
    )
    model_b = train_model(Xc_train, yc_train, Xc_test, yc_test, XGB_PARAMS_CLEAN)
    evaluate(model_b, Xc_test, yc_test, label="Model B — Leakage-free")

    elapsed = time.perf_counter() - t0
    log.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
