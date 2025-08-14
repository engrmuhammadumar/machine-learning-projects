#!/usr/bin/env python3
"""
ARIMA Quickstart for AAMR (Age-Adjusted Mortality Rate)

Usage (example):
  python arima_quickstart_aamr.py --excel data/overall.xlsx --sheet "overall" --year-col "Year" --target-col "Age Adjusted Rate" --h 10 --test-years 5

If you don't know exact column names, you can omit --year-col or --target-col and the script will try to detect them.
This script:
  1) Loads an annual AAMR time series from an Excel file
  2) Runs stationarity checks and diagnostics
  3) Splits into train/test (last N years = test)
  4) Finds a good ARIMA model (auto_arima)
  5) Backtests on the test period
  6) Refits on full data and forecasts H years ahead
  7) Writes forecast CSV + PNG plots to ./outputs/

Dependencies:
  pip install pandas numpy matplotlib statsmodels pmdarima openpyxl
"""

import argparse
import os
import re
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

def _find_col(cols, patterns):
    cols_lower = {c.lower(): c for c in cols}
    for pat in patterns:
        regex = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if regex.search(c):
                return c
        for lc, orig in cols_lower.items():
            if regex.search(lc):
                return orig
    return None

def load_series_from_excel(path: str,
                           sheet: Optional[str],
                           year_col: Optional[str],
                           target_col: Optional[str]) -> pd.Series:
    """Return a pandas Series indexed by year (pd.PeriodIndex with freq='Y')"""
    df = pd.read_excel(path, sheet_name=sheet)
    # Try to detect year & target columns if missing
    if year_col is None:
        year_col = _find_col(
            df.columns,
            patterns=[r"^year\s*code$", r"^year$", r"year", r"yr"]
        )
        if year_col is None:
            raise ValueError("Could not detect a Year column. Use --year-col.")
    if target_col is None:
        target_col = _find_col(
            df.columns,
            patterns=[
                r"aamr",
                r"age.*adjust.*(rate|death.*rate)",
                r"age[-_\s]*adjusted[-_\s]*(rate|death.*rate)",
                r"age.*adj.*rate"
            ]
        )
        if target_col is None:
            raise ValueError("Could not detect an AAMR/age-adjusted rate column. Use --target-col.")
    # Clean and coerce
    ts = df[[year_col, target_col]].copy()
    ts = ts.dropna()
    # Some Year columns might be strings; coerce
    ts[year_col] = pd.to_numeric(ts[year_col], errors="coerce")
    ts[target_col] = pd.to_numeric(ts[target_col], errors="coerce")
    ts = ts.dropna()
    ts = ts.sort_values(year_col)
    # Build a PeriodIndex for annual data
    idx = pd.PeriodIndex(ts[year_col].astype(int), freq="Y")
    y = pd.Series(ts[target_col].values, index=idx, name="AAMR")
    return y

def adf_summary(y: pd.Series) -> dict:
    res = adfuller(y, autolag="AIC")
    return {
        "ADF stat": res[0],
        "p-value": res[1],
        "lags used": res[2],
        "nobs": res[3],
        "crit values": res[4],
        "icbest": res[5],
    }

def train_test_split_series(y: pd.Series, test_years: int) -> Tuple[pd.Series, pd.Series]:
    if test_years <= 0 or test_years >= len(y):
        raise ValueError("test_years must be between 1 and len(y)-1")
    return y.iloc[:-test_years], y.iloc[-test_years:]

def rolling_backtest(train: pd.Series, test: pd.Series, order_hint=None) -> dict:
    """
    Rolling-origin evaluation: refit on expanding window and predict 1-step ahead.
    Returns MAE/RMSE/MAPE.
    """
    preds = []
    actuals = []
    hist = train.copy()
    for t in test.index:
        # Fit auto ARIMA on 'hist'
        model = auto_arima(
            hist,
            seasonal=False,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            start_d=0, max_d=2,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            information_criterion="aicc",
            with_intercept=True,
            trace=False,
        )
        fc = model.predict(n_periods=1)[0]
        preds.append(fc)
        actuals.append(test.loc[t])
        # expand window
        hist = pd.concat([hist, pd.Series([test.loc[t]], index=[t])])
    preds = np.array(preds)
    actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, preds)
    rmse = mean_squared_error(actuals, preds, squared=False)
    mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}

def fit_auto_arima(y: pd.Series):
    model = auto_arima(
        y,
        seasonal=False,
        start_p=0, start_q=0,
        max_p=8, max_q=8,
        start_d=0, max_d=2,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aicc",
        with_intercept=True,
        trace=False,
    )
    return model

def plot_series(y: pd.Series, title: str, path: str):
    plt.figure(figsize=(10, 4))
    y.plot(marker="o")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("AAMR")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_residual_diagnostics(residuals: pd.Series, out_prefix: str):
    # Residual timeseries
    plt.figure(figsize=(10, 3.5))
    plt.plot(residuals.values)
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_residuals.png", dpi=150)
    plt.close()

    # ACF residuals
    fig = plt.figure(figsize=(10, 3.5))
    plot_acf(residuals, lags=min(24, len(residuals)-1), zero=False)
    plt.title("Residual ACF")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_resid_acf.png", dpi=150)
    plt.close()

    # PACF residuals
    fig = plt.figure(figsize=(10, 3.5))
    plot_pacf(residuals, lags=min(24, len(residuals)-1), zero=False, method="ywm")
    plt.title("Residual PACF")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_resid_pacf.png", dpi=150)
    plt.close()

def ljung_box_pvalue(residuals: pd.Series, lags=12) -> float:
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[-1])

def make_outputs_dir(base="outputs"):
    os.makedirs(base, exist_ok=True)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to Excel file (.xlsx) containing annual AAMR data")
    ap.add_argument("--sheet", default=None, help="Sheet name (optional)")
    ap.add_argument("--year-col", default=None, help="Year column (e.g., 'Year' or 'Year Code')")
    ap.add_argument("--target-col", default=None, help="AAMR column (e.g., 'Age Adjusted Rate')")
    ap.add_argument("--h", type=int, default=10, help="Forecast horizon in years")
    ap.add_argument("--test-years", type=int, default=5, help="Holdout years for backtesting")
    ap.add_argument("--outdir", default="outputs", help="Where to save plots and CSVs")
    args = ap.parse_args()

    outdir = make_outputs_dir(args.outdir)
    # 1) Load
    y = load_series_from_excel(args.excel, args.sheet, args.year_col, args.target_col)
    # Basic plot
    plot_series(y, "AAMR Over Time", os.path.join(outdir, "series.png"))
    # 2) ADF test
    adf = adf_summary(y)
    print("ADF Test:", adf)
    # 3) Train/test
    train, test = train_test_split_series(y, args.test_years)
    # 4) Rolling backtest
    bt = rolling_backtest(train, test)
    print("Backtest:", bt)
    # 5) Fit on full data
    model = fit_auto_arima(y)
    print("Selected order:", model.order)
    # Diagnostics
    resid = pd.Series(model.resid(), index=y.index[-len(model.resid()):])
    plot_residual_diagnostics(resid, os.path.join(outdir, "diagnostics"))
    pval = ljung_box_pvalue(resid, lags=min(12, len(resid)//2))
    print(f"Ljung-Box p-value (no autocorr in residuals): {pval:.4f}")
    # 6) Forecast H years
    fc_index = pd.period_range(y.index[-1]+1, periods=args.h, freq="Y")
    fc_values = model.predict(n_periods=args.h)
    fc = pd.Series(fc_values, index=fc_index, name="forecast")
    # also get confints
    conf = model.predict(n_periods=args.h, return_conf_int=True)[1]
    conf_df = pd.DataFrame(conf, index=fc_index, columns=["lower", "upper"])
    # Save CSV
    out_csv = os.path.join(outdir, "forecast.csv")
    pd.concat([fc, conf_df], axis=1).to_csv(out_csv, index_label="Year")
    print(f"Saved forecast to {out_csv}")
    # Plot forecast
    plt.figure(figsize=(10, 4))
    y.plot(marker="o", label="history")
    # Create a DatetimeIndex for plotting compatibility
    fc_x = [p.to_timestamp('Y') for p in fc.index]
    plt.plot(fc_x, fc.values, marker="o", label="forecast")
    plt.fill_between(fc_x, conf_df["lower"].values, conf_df["upper"].values, alpha=0.2, label="95% CI")
    plt.title(f"ARIMA Forecast ({model.order})")
    plt.xlabel("Year")
    plt.ylabel("AAMR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forecast.png"), dpi=150)
    plt.close()
    print("Done. Check the outputs/ folder.")

if __name__ == "__main__":
    main()
