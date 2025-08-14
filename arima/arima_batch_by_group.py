#!/usr/bin/env python3
"""
Batch ARIMA for AAMR by group (e.g., Age group, Gender, State)

Usage (example):
  python arima_batch_by_group.py --excel data/age_groups.xlsx --sheet "age groups" --group-col "Age group" --year-col "Year" --target-col "Age Adjusted Rate" --h 10 --test-years 5 --outdir outputs_age_groups

If you don't know exact column names, you can omit --year-col/--target-col and the script will try to detect them.
This script will:
  * Validate minimum history per group
  * Auto-fit ARIMA for each group
  * Backtest and save metrics
  * Save per-group forecast CSV + PNGs
  * Aggregate all group forecasts into one CSV

Dependencies:
  pip install pandas numpy matplotlib statsmodels pmdarima openpyxl
"""

import argparse
import os
import re
import warnings
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

def _find_col(cols, patterns):
    for pat in patterns:
        regex = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if regex.search(c):
                return c
    return None

def detect_year_and_target(df: pd.DataFrame, year_col: Optional[str], target_col: Optional[str]):
    if year_col is None:
        year_col = _find_col(
            df.columns,
            [r"^year\s*code$", r"^year$", r"year", r"yr"]
        )
    if target_col is None:
        target_col = _find_col(
            df.columns,
            [
                r"aamr",
                r"age.*adjust.*(rate|death.*rate)",
                r"age[-_\s]*adjusted[-_\s]*(rate|death.*rate)",
                r"age.*adj.*rate"
            ]
        )
    if year_col is None or target_col is None:
        raise ValueError("Could not detect year/target columns; please specify them.")
    return year_col, target_col

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def series_from_group(df: pd.DataFrame, group_name, group_col, year_col, target_col) -> pd.Series:
    sub = df[df[group_col] == group_name][[year_col, target_col]].dropna()
    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
    sub = sub.dropna().sort_values(year_col)
    idx = pd.PeriodIndex(sub[year_col].astype(int), freq="Y")
    return pd.Series(sub[target_col].values, index=idx, name=str(group_name))

def backtest_expand(y: pd.Series, test_years: int) -> Dict[str, float]:
    train = y.iloc[:-test_years]
    test = y.iloc[-test_years:]
    preds, actuals = [], []
    hist = train.copy()
    for t in test.index:
        m = auto_arima(
            hist,
            seasonal=False,
            start_p=0, start_q=0,
            max_p=6, max_q=6,
            start_d=0, max_d=2,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            information_criterion="aicc",
            with_intercept=True,
            trace=False,
        )
        preds.append(m.predict(n_periods=1)[0])
        actuals.append(test.loc[t])
        hist = pd.concat([hist, pd.Series([test.loc[t]], index=[t])])
    preds = np.array(preds); actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, preds)
    rmse = mean_squared_error(actuals, preds, squared=False)
    mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}

def ljung_box_pvalue(residuals: pd.Series, lags=12) -> float:
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[-1])

def plot_group_forecast(group_name, y, fc, conf_df, out_png):
    plt.figure(figsize=(10, 4))
    y.plot(marker="o", label="history")
    fc_x = [p.to_timestamp('Y') for p in fc.index]
    plt.plot(fc_x, fc.values, marker="o", label="forecast")
    plt.fill_between(fc_x, conf_df["lower"].values, conf_df["upper"].values, alpha=0.2, label="95% CI")
    plt.title(f"{group_name} forecast")
    plt.xlabel("Year")
    plt.ylabel("AAMR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Excel file path (.xlsx)")
    ap.add_argument("--sheet", default=None, help="Sheet name")
    ap.add_argument("--group-col", required=True, help="Column to group by (e.g., 'Age group', 'Gender', 'State')")
    ap.add_argument("--year-col", default=None)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--test-years", type=int, default=5)
    ap.add_argument("--min-years", type=int, default=8, help="Skip groups with < min-years of data")
    ap.add_argument("--outdir", default="outputs_groups")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_excel(args.excel, sheet_name=args.sheet)
    # Detect columns if not provided
    ycol, tcol = detect_year_and_target(df, args.year_col, args.target_col)

    # Clean group column
    if args.group_col not in df.columns:
        # try case-insensitive match
        match = None
        for c in df.columns:
            if c.strip().lower() == args.group_col.strip().lower():
                match = c; break
        if match is None:
            raise ValueError(f"Group column '{args.group_col}' not found in columns: {list(df.columns)}")
        args.group_col = match

    groups = [g for g in df[args.group_col].dropna().unique()]
    all_rows = []
    metrics_rows = []

    for g in groups:
        y = series_from_group(df, g, args.group_col, ycol, tcol)
        if len(y) < args.min_years:
            print(f"Skipping '{g}' (only {len(y)} years)")
            continue

        # Backtest
        if args.test_years < len(y):
            bt = backtest_expand(y, args.test_years)
        else:
            bt = {"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan}

        # Fit and forecast
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
        resid = pd.Series(model.resid(), index=y.index[-len(model.resid()):])
        pval = ljung_box_pvalue(resid, lags=min(12, max(2, len(resid)//3)))

        fc_index = pd.period_range(y.index[-1]+1, periods=args.h, freq="Y")
        fc_values = model.predict(n_periods=args.h)
        fc = pd.Series(fc_values, index=fc_index, name="forecast")
        conf = model.predict(n_periods=args.h, return_conf_int=True)[1]
        conf_df = pd.DataFrame(conf, index=fc_index, columns=["lower", "upper"])

        # Save per-group CSV
        gsafe = re.sub(r"[^A-Za-z0-9]+", "_", str(g)).strip("_").lower() or "group"
        gout = os.path.join(args.outdir, gsafe)
        ensure_outdir(gout)
        pd.concat([fc, conf_df], axis=1).to_csv(os.path.join(gout, "forecast.csv"), index_label="Year")
        # Save plot
        plot_group_forecast(g, y, fc, conf_df, os.path.join(gout, "forecast.png"))
        # Collect aggregate
        df_fc = pd.concat([fc, conf_df], axis=1).reset_index().rename(columns={"index": "Year"})
        df_fc.insert(0, "Group", g)
        all_rows.append(df_fc)

        metrics_rows.append({
            "Group": g,
            "n_years": len(y),
            "order": str(model.order),
            "MAE": bt["MAE"],
            "RMSE": bt["RMSE"],
            "MAPE_%": bt["MAPE_%"],
            "LjungBox_p": pval,
        })
        print(f"Done {g} -> order {model.order}, MAPE {bt['MAPE_%']:.2f}%")

    if all_rows:
        agg = pd.concat(all_rows, ignore_index=True)
        agg.to_csv(os.path.join(args.outdir, "all_group_forecasts.csv"), index=False)
    if metrics_rows:
        metr = pd.DataFrame(metrics_rows)
        metr.to_csv(os.path.join(args.outdir, "model_metrics.csv"), index=False)
    print("All done. See:", args.outdir)

if __name__ == "__main__":
    main()
