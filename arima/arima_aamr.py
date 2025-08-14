import warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# --------- Helpers ---------
def find_aamr_col(df):
    cols = [c for c in df.columns if isinstance(c, str)]
    # direct matches
    for key in ["aamr", "age adjusted rate", "age-adjusted rate", "age adjusted", "age-adjusted", "age adjust"]:
        for c in cols:
            if key in c.lower():
                return c
    # fallback: contains both 'age' and 'adjust'
    cand = [c for c in cols if ("age" in c.lower() and "adjust" in c.lower())]
    if cand:
        return cand[0]
    raise ValueError("Could not find AAMR column. Please set AAMR_COL manually.")

def prep_series(df, year_col="Year", aamr_col=None):
    if year_col not in df.columns:
        # try common alternatives
        alt = [c for c in df.columns if str(c).strip().lower() in ["year code", "yr", "date"]]
        if alt:
            year_col = alt[0]
        else:
            raise ValueError("Couldn't find a Year column. Rename your year column to 'Year'.")
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[year_col])
    df = df.sort_values(year_col)

    if aamr_col is None:
        aamr_col = find_aamr_col(df)

    y = pd.to_numeric(df[aamr_col], errors="coerce")
    s = pd.Series(y.values, index=pd.to_datetime(df[year_col].astype(int), format="%Y"), name="AAMR")
    # annual frequency
    s = s.asfreq("A-DEC")
    return s, aamr_col

def auto_fit(train):
    # Auto-ARIMA (annual data; no seasonality)
    model = pm.auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        trace=False,
        information_criterion="aicc",
        error_action="ignore",
        suppress_warnings=True,
        max_p=5, max_q=5, max_d=2,
        with_intercept=True
    )
    return model

def diagnostics(model, resid, label="series"):
    lb = acorr_ljungbox(resid, lags=[8], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    print(f"[{label}] Ljung-Box p-value (lag 8): {lb_p:.4f}")

def plot_fit_forecast(y, fc_df, title, out_png=None):
    plt.figure(figsize=(9,5))
    plt.plot(y.index, y.values, label="Actual")
    plt.plot(fc_df.index, fc_df["mean"], label="Forecast")
    plt.fill_between(fc_df.index, fc_df["lower"], fc_df["upper"], alpha=0.2, label="95% PI")
    plt.title(title)
    plt.xlabel("Year"); plt.ylabel("AAMR")
    plt.legend()
    if out_png:
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
    plt.show()

def rolling_backtest(y, initial_year=2008, horizon=3, step=1):
    """
    Rolling-origin evaluation.
    initial_year: first year that ends the initial training window (e.g., 2008 -> train up to 2008)
    horizon: forecast horizon in years for each fold
    """
    years = pd.Index(y.index.year)
    metrics = []
    start_idx = years.get_loc(years[years == initial_year][0])
    for end in range(start_idx, len(y) - horizon, step):
        train = y.iloc[: end + 1]
        test = y.iloc[end + 1 : end + 1 + horizon]
        model = auto_fit(train)
        fc = model.predict(n_periods=len(test))
        mae = np.mean(np.abs(fc - test.values))
        mape = np.mean(np.abs((fc - test.values) / np.maximum(1e-8, test.values))) * 100.0
        rmse = np.sqrt(np.mean((fc - test.values) ** 2))
        metrics.append({"train_end": int(train.index[-1].year), "MAE": mae, "MAPE": mape, "RMSE": rmse})
    return pd.DataFrame(metrics)

def fit_forecast_one(y, label, out_dir="outputs", horizon_future=5, train_until_year=None):
    os.makedirs(out_dir, exist_ok=True)

    # Optionally hold out recent years (e.g., 2020–2023) to check accuracy
    if train_until_year is not None:
        train = y[y.index.year <= train_until_year]
        test  = y[y.index.year >  train_until_year]
    else:
        train, test = y, y.iloc[0:0]

    model = auto_fit(train)
    print(f"[{label}] Selected order: {model.order}")

    # Diagnostics on residuals
    resid = model.resid()
    diagnostics(model, resid, label)

    # In-sample + holdout forecast
    n_test = len(test)
    fc_in = None
    if n_test > 0:
        preds = model.predict(n_periods=n_test, return_conf_int=True, alpha=0.05)
        fc_vals, ci = preds
        idx = pd.date_range(start=test.index[0], periods=n_test, freq="A-DEC")
        fc_in = pd.DataFrame({"mean": fc_vals, "lower": ci[:,0], "upper": ci[:,1]}, index=idx)
        # Accuracy
        mae = np.mean(np.abs(fc_vals - test.values))
        mape = np.mean(np.abs((fc_vals - test.values) / np.maximum(1e-8, test.values))) * 100.0
        rmse = np.sqrt(np.mean((fc_vals - test.values) ** 2))
        print(f"[{label}] Holdout MAE={mae:.3f} | MAPE={mape:.2f}% | RMSE={rmse:.3f}")

    # Refit on full series, forecast future
    model_full = auto_fit(y)
    preds = model_full.predict(n_periods=horizon_future, return_conf_int=True, alpha=0.05)
    fc_vals, ci = preds
    idx = pd.date_range(start=y.index[-1] + pd.offsets.YearEnd(1), periods=horizon_future, freq="A-DEC")
    fc_out = pd.DataFrame({"mean": fc_vals, "lower": ci[:,0], "upper": ci[:,1]}, index=idx)

    # Combine for plotting
    plot_df = fc_out if fc_in is None else pd.concat([fc_in, fc_out])
    title = f"AAMR ARIMA — {label}"
    plot_fit_forecast(y, plot_df, title, out_png=os.path.join(out_dir, f"{label}_forecast.png"))

    # Save forecasts
    fc_out.to_csv(os.path.join(out_dir, f"{label}_future_forecast.csv"))
    return model_full, fc_out

# --------- Entry points ---------
def run_overall(excel_path, sheet_name=0, aamr_col=None, train_until_year=2019, horizon_future=5):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    y, col = prep_series(df, year_col="Year", aamr_col=aamr_col)
    print(f"Detected AAMR column: {col}")
    # Backtest (optional but recommended)
    if y.index.year.min() <= 2003:
        cv = rolling_backtest(y, initial_year=max(2008, int(y.index.year.min()) + 9), horizon=3, step=1)
        print(cv.tail(5))
        cv.to_csv("outputs/_overall_backtest.csv", index=False)
    # Fit & forecast
    return fit_forecast_one(y, label="overall", out_dir="outputs", horizon_future=horizon_future, train_until_year=train_until_year)

def run_by_category(excel_path, group_col, sheet_name=0, aamr_col=None, train_until_year=2019, horizon_future=5):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    groups = df[group_col].dropna().unique()
    results = {}
    for g in groups:
        sdf = df[df[group_col] == g]
        if len(sdf) < 8:
            continue
        y, col = prep_series(sdf, year_col="Year", aamr_col=aamr_col)
        label = f"{group_col}_{g}"
        print(f"\n=== {label} | AAMR col: {col} ===")
        model, fc = fit_forecast_one(y, label=label, out_dir="outputs", horizon_future=horizon_future, train_until_year=train_until_year)
        results[g] = fc
    return results

if __name__ == "__main__":
    # ---- EDIT THESE PATHS/NAMES ----
    EXCEL_FILE = r"D:\arima project\data\obesity+DM+HTN overall final.xlsx"   # or e.g. "Obesity+DM+HTN age groups final.xlsx"
    SHEET = 0
    # If auto-detection fails, set the exact column name, e.g. AAMR_COL = "Age Adjusted Rate"
    AAMR_COL = None

    # 1) OVERALL
    if os.path.exists(EXCEL_FILE):
        run_overall(EXCEL_FILE, sheet_name=SHEET, aamr_col=AAMR_COL, train_until_year=2019, horizon_future=5)
    else:
        print(f"File not found: {EXCEL_FILE}")

    # 2) EXAMPLES FOR OTHER FILES (uncomment and set correct file + group column)
    # run_by_category("Obesity+DM+HTN age groups final.xlsx", group_col="Age group", aamr_col=AAMR_COL)
    # run_by_category("obesity+DM+HTN gender final.xlsx",     group_col="Gender",    aamr_col=AAMR_COL)
    # run_by_category("obesity+DM+HTN race final.xlsx",       group_col="Race",      aamr_col=AAMR_COL)
    # run_by_category("Obesity+DM+HTN states corrected.xlsx", group_col="State",     aamr_col=AAMR_COL)
    # run_by_category("obesity+DM+HTN census region final.xlsx", group_col="Region", aamr_col=AAMR_COL)
    # run_by_category("Obesity+DM+HTN urbanization final.xlsx",  group_col="Urbanization", aamr_col=AAMR_COL)
    # run_by_category("Obesity+DM+HTN POD corrected.xlsx",       group_col="POD",    aamr_col=AAMR_COL)
