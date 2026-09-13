"""
flight_forecast/src/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration — edit this file to tune the pipeline without touching
any training logic.
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Paths and column definitions."""
    target: str = "FLT_TOT_1"

    # Columns always removed before training
    drop_always: list[str] = field(default_factory=lambda: [
        "YEAR", "FLT_DATE", "MONTH_MON",
    ])

    # High-correlation columns that constitute data leakage
    leakage_cols: list[str] = field(default_factory=lambda: [
        "FLT_TOT_IFR_2",
        "FLT_DEP_1",
        "FLT_ARR_1",
        "FLT_DEP_IFR_2",
        "FLT_ARR_IFR_2",
    ])

    # Columns to label-encode
    categorical_cols: list[str] = field(default_factory=lambda: [
        "APT_ICAO",
        "APT_NAME",
        "STATE_NAME",
        "Pivot Label",
    ])


@dataclass
class SplitConfig:
    """Train / test split settings."""
    test_size: float = 0.20
    random_seed: int = 42


@dataclass
class ModelAConfig:
    """XGBoost hyperparameters for Model A (full feature set)."""
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 5
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    n_jobs: int = -1


@dataclass
class ModelBConfig:
    """XGBoost hyperparameters for Model B (leakage-free)."""
    n_estimators: int = 100
    max_depth: int = 3
    random_state: int = 42
    n_jobs: int = -1


# ── Convenience singletons ────────────────────────────────────────────────────

data_cfg    = DataConfig()
split_cfg   = SplitConfig()
model_a_cfg = ModelAConfig()
model_b_cfg = ModelBConfig()
