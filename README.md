# ✈️ Flight Forecast

> **XGBoost-powered flight-volume forecasting for European airports**  
> Predict daily total flight movements from EUROCONTROL data with two model configurations — one full-featured, one data-leakage-free.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Models](#models)
7. [Feature Engineering Notes](#feature-engineering-notes)
8. [Configuration](#configuration)
9. [Results](#results)
10. [License](#license)

---

## Overview

This project trains a gradient-boosted regression model to forecast **`FLT_TOT_1`** — the total number of daily flights (IFR + VFR) at European airports. It ships two model variants:

| Model | Feature set | Purpose |
|-------|-------------|---------|
| **Model A** | All numeric features | Establish an upper-bound benchmark |
| **Model B** | Leakage-free features only | Production-realistic forecast |

Model B deliberately removes columns that are defined as components of the target (departures + arrivals = total flights) to produce an honest evaluation of generalisability.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | EUROCONTROL Airport Traffic |
| File | `e_f_d.csv` |
| Rows | 688,099 |
| Columns | 14 |
| Date range | 2016 – 2022 |
| Airports | 332 across 42 European states |

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `YEAR` | int | Calendar year |
| `MONTH_NUM` | int | Month number (1–12) |
| `MONTH_MON` | str | Month abbreviation (JAN – DEC) |
| `FLT_DATE` | datetime | Full date of the record |
| `APT_ICAO` | str | ICAO airport code (e.g. `EGLL`) |
| `APT_NAME` | str | Airport common name |
| `STATE_NAME` | str | Country name |
| `FLT_DEP_1` | int | Total departures (IFR + VFR) ⚠️ |
| `FLT_ARR_1` | int | Total arrivals (IFR + VFR) ⚠️ |
| `FLT_TOT_1` | int | **Target — total flights** |
| `FLT_DEP_IFR_2` | float | IFR departures ⚠️ |
| `FLT_ARR_IFR_2` | float | IFR arrivals ⚠️ |
| `FLT_TOT_IFR_2` | float | IFR total flights ⚠️ |
| `Pivot Label` | str | Human-readable airport label |

> ⚠️ Columns marked with this symbol are **data leakage** with respect to `FLT_TOT_1` and are removed in Model B.

---

## Project Structure

```
flight_forecast/
├── src/
│   ├── train.py        # Main training pipeline (entry point)
│   |── config.py     # Centralised hyperparameter & column config
|   └── e_f_d.csv     # dataset
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

---

## Quick Start

### 1 — Clone / download

```bash
git clone https://github.com/<your-org>/flight-forecast.git
cd flight-forecast
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run training

```bash
python src/train.py path/to/e_f_d.csv
```

---

## Usage

```
usage: train.py [-h] [--test-size FRAC] [--seed SEED] [--top-k K] DATA_CSV

positional arguments:
  DATA_CSV           Path to the input CSV (e.g. e_f_d.csv)

options:
  -h, --help         show this help message and exit
  --test-size FRAC   Fraction of data held out for testing (default: 0.20)
  --seed SEED        Random seed for reproducibility (default: 42)
  --top-k K          Number of top-correlation features to display (default: 10)
```

### Examples

```bash
# Default run (80/20 split, seed 42)
python src/train.py data/e_f_d.csv

# Custom split and seed
python src/train.py data/e_f_d.csv --test-size 0.15 --seed 123

# Show top 15 correlated features
python src/train.py data/e_f_d.csv --top-k 15
```

---

## Models

### Model A — Full Feature Set

Uses every numeric column after encoding. Serves as an upper-bound benchmark. Because `FLT_DEP_1 + FLT_ARR_1 = FLT_TOT_1` by definition, this model has near-perfect information and its R² is expected to be very high (> 0.99).

**XGBoost hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `learning_rate` | 0.1 |
| `max_depth` | 5 |
| `min_child_weight` | 3 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `early_stopping_rounds` | 50 |

---

### Model B — Leakage-Free

Removes all columns that are direct components of or surrogates for the target before training. This is the model to use for real-world forecasting scenarios where only historical context (airport identity, month, year) is available at inference time.

**Removed columns**

```python
["FLT_TOT_IFR_2", "FLT_DEP_1", "FLT_ARR_1", "FLT_DEP_IFR_2", "FLT_ARR_IFR_2"]
```

**XGBoost hyperparameters**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 100 |
| `max_depth` | 3 |

---

## Feature Engineering Notes

- **Categorical encoding** — `APT_ICAO`, `APT_NAME`, `STATE_NAME`, and `Pivot Label` are ordinally encoded with `sklearn.LabelEncoder`. For production use, consider target-encoding or embeddings to capture airport-level traffic patterns better.
- **Missing values** — ~70% of IFR columns (`FLT_DEP_IFR_2`, `FLT_ARR_IFR_2`, `FLT_TOT_IFR_2`) are `NaN` (airports that don't report IFR separately). These are filled with `0`. This is sound for Model A but redundant for Model B where those columns are dropped.
- **Time features** — `YEAR`, `FLT_DATE`, and `MONTH_MON` are dropped from the feature matrix to avoid trivial time-based leakage. `MONTH_NUM` is retained as an ordinal seasonal signal.
- **No scaling needed** — XGBoost is invariant to monotonic feature transforms, so no normalisation or standardisation is applied.

---

## Configuration

All column definitions and hyperparameters live in `src/config.py`. Edit that file to change any setting without touching training logic.

```python
# src/config.py — example: tighten the regularisation for Model B
@dataclass
class ModelBConfig:
    n_estimators: int = 150   # was 100
    max_depth:    int = 4     # was 3
    reg_alpha:    float = 0.1 # L1 regularisation
    reg_lambda:   float = 1.0 # L2 regularisation
    random_state: int = 42
    n_jobs:       int = -1
```

---

## Results

Expected output on the default 80/20 split with seed 42:

```
────────────────────────────────────────────
  Model A — Full features
────────────────────────────────────────────
  R²   :   ~0.9999
  MAE  :          ~1  flights
  RMSE :          ~3  flights
────────────────────────────────────────────

────────────────────────────────────────────
  Model B — Leakage-free
────────────────────────────────────────────
  R²   :   ~0.85–0.92
  MAE  :       ~30–60  flights
  RMSE :       ~80–150 flights
────────────────────────────────────────────
```

Model A's near-perfect score is expected and confirms leakage is present. Model B's score reflects genuine predictive power from airport identity and seasonal features alone.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ and gradient boosting</p>
