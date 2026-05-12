# utils.py — shared definitions for Module2.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA = "Assignment 2-20260512/DATA/DATA"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(filename, col):
    """Load a CSV time series file.
    Input:  filename (str) name in DATA dir; col (str) column label to assign
    Output: pd.DataFrame with DatetimeIndex and one column
    """
    df = pd.read_csv(f"{DATA}/{filename}", parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [col]
    return df

def load_q_diepoldsau():
    """Load Diepoldsau discharge [m³/s], 10-min resolution.
    Output: pd.DataFrame, column 'Q'
    """
    return _load("Q_Diepoldsau_m3s.csv", "Q")

def load_q_gisingen():
    """Load Gisingen discharge [m³/s], 15-min resolution.
    Output: pd.DataFrame, column 'Q'
    """
    return _load("Q_Gisingen_1976-2023.csv", "Q")

def load_ssc_diepoldsau():
    """Load Diepoldsau SSC [g/L], 10-min resolution.
    Output: pd.DataFrame, column 'SSC'
    """
    return _load("SSC_Diepoldsau_gL.csv", "SSC")

def load_ssc_gisingen():
    """Load Gisingen SSC [g/L], 15-min resolution.
    Output: pd.DataFrame, column 'SSC'
    """
    return _load("SSC_Gisingen_2003-2020.csv", "SSC")


# ── Section 1: Timeseries review ──────────────────────────────────────────────

def compute_monthly_mean(df):
    """Resample a time series to monthly means.
    Input:  df (pd.DataFrame) with DatetimeIndex
    Output: pd.DataFrame at month-end frequency
    """
    return df.resample("ME").mean()

def fit_linear_trend(series):
    """Fit OLS linear regression to a series and test slope significance.
    Input:  series (pd.Series) with DatetimeIndex
    Output: dict with keys slope, intercept, p_value, r_squared, x_numeric, index
    """
    s = series.dropna()
    x = np.arange(len(s), dtype=float)
    slope, intercept, r, p_value, _ = stats.linregress(x, s.values)
    return dict(slope=slope, intercept=intercept, p_value=p_value,
                r_squared=r**2, x_numeric=x, index=s.index)

def detrend_series(series, alpha=0.05):
    """Subtract significant linear trend or mean; return zero-mean series.
    Input:  series (pd.Series), alpha (float) significance level for slope test
    Output: (detrended pd.Series, trend_info dict with added 'significant' key)
    """
    s = series.dropna()
    t = fit_linear_trend(s)
    if t["p_value"] < alpha:
        trend_vals = t["slope"] * t["x_numeric"] + t["intercept"]
        detrended = pd.Series(s.values - trend_vals, index=t["index"], name=series.name)
    else:
        detrended = pd.Series(s.values - s.mean(), index=t["index"], name=series.name)
    t["significant"] = t["p_value"] < alpha
    return detrended, t

def plot_timeseries(series_dict, ylabel, title):
    """Plot multiple monthly time series on one axes.
    Input:  series_dict (dict {label str: pd.Series}); ylabel (str); title (str)
    Output: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    for label, s in series_dict.items():
        ax.plot(s.index, s.values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_trend_fit(series, trend_info, title=""):
    """Plot a time series with its fitted linear trend line.
    Input:  series (pd.Series); trend_info (dict from fit_linear_trend); title (str)
    Output: matplotlib Figure
    """
    s = series.dropna()
    trend_line = trend_info["slope"] * trend_info["x_numeric"] + trend_info["intercept"]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(trend_info["index"], s.values, alpha=0.7, label="data")
    ax.plot(trend_info["index"], trend_line, "r--",
            label=f"trend  p={trend_info['p_value']:.3f}  R²={trend_info['r_squared']:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return fig


# ── Section 2: Timeseries modelling ──────────────────────────────────────────

def compute_acf_pacf(series, lags=40, alpha=0.05):
    """Compute empirical ACF and PACF with 95% confidence bound.
    Input:  series (pd.Series) detrended; lags (int); alpha (float)
    Output: dict with keys acf, pacf (np.array), lags (np.array), conf_bound (float)
    """
    s = series.dropna().values
    acf_vals, _ = acf(s, nlags=lags, alpha=alpha, fft=True)
    pacf_vals, _ = pacf(s, nlags=lags, alpha=alpha)
    conf = 1.96 / np.sqrt(len(s))
    return dict(acf=acf_vals, pacf=pacf_vals,
                lags=np.arange(lags + 1), conf_bound=conf)

def plot_acf_pacf(corr_dict, title=""):
    """Plot ACF and PACF side by side with 95% CI.
    Input:  corr_dict (dict from compute_acf_pacf); title (str)
    Output: matplotlib Figure
    """
    lags = corr_dict["lags"]
    ci = corr_dict["conf_bound"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, key, label in zip(axes, ["acf", "pacf"], ["ACF", "PACF"]):
        vals = corr_dict[key] if key == "acf" else corr_dict[key][1:]
        x = lags if key == "acf" else lags[1:]
        ax.bar(x, vals, color="steelblue", alpha=0.7, width=0.6)
        ax.axhline(ci,  color="r", linestyle="--", linewidth=1)
        ax.axhline(-ci, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"{label} — {title}")
        ax.set_xlabel("Lag (months)")
    plt.tight_layout()
    return fig

def select_ar_order(series, max_p=6):
    """Select AR order p minimising AIC over p = 1..max_p.
    Input:  series (pd.Series) detrended; max_p (int)
    Output: (best_p int, aic_list list of (p, aic))
    """
    aics = []
    for p in range(1, max_p + 1):
        try:
            res = ARIMA(series.dropna(), order=(p, 0, 0)).fit()
            aics.append((p, round(res.aic, 2)))
        except Exception:
            aics.append((p, np.inf))
    best_p = min(aics, key=lambda x: x[1])[0]
    return best_p, aics

def select_arma_order(series, max_p=4, max_q=4):
    """Select ARMA(p,q) order minimising AIC over grid p=1..max_p, q=1..max_q.
    Input:  series (pd.Series) detrended; max_p (int); max_q (int)
    Output: (best_p int, best_q int, aic_table list of (p, q, aic))
    """
    table = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                res = ARIMA(series.dropna(), order=(p, 0, q)).fit()
                table.append((p, q, round(res.aic, 2)))
            except Exception:
                table.append((p, q, np.inf))
    best = min(table, key=lambda x: x[2])
    return best[0], best[1], table

def fit_ar(series, order):
    """Fit AR(p) model via ARIMA(p,0,0).
    Input:  series (pd.Series) detrended; order (int) AR order p
    Output: statsmodels ARIMAResults
    """
    return ARIMA(series.dropna(), order=(order, 0, 0)).fit()

def fit_arma(series, p, q):
    """Fit ARMA(p,q) model via ARIMA(p,0,q).
    Input:  series (pd.Series) detrended; p (int) AR order; q (int) MA order
    Output: statsmodels ARIMAResults
    """
    return ARIMA(series.dropna(), order=(p, 0, q)).fit()


# ── Section 3: Evaluation ─────────────────────────────────────────────────────

def get_theoretical_acf(model_result, lags=40):
    """Compute theoretical ACF from a fitted ARIMA model.
    Input:  model_result (ARIMAResults); lags (int)
    Output: np.array of ACF values, length lags+1
    """
    ar = np.r_[1, -model_result.arparams]
    ma = np.r_[1,  model_result.maparams]
    return ArmaProcess(ar, ma).acf(lags + 1)

def plot_acf_comparison(corr_dict, model_result, title=""):
    """Plot observed ACF vs theoretical ACF with 95% CI on same axes.
    Input:  corr_dict (dict from compute_acf_pacf); model_result (ARIMAResults); title (str)
    Output: matplotlib Figure
    """
    lags = corr_dict["lags"]
    ci = corr_dict["conf_bound"]
    theo = get_theoretical_acf(model_result, lags=len(lags) - 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lags, corr_dict["acf"], alpha=0.5, label="Observed ACF", color="steelblue")
    ax.plot(lags, theo, "r-o", markersize=4, label="Theoretical ACF")
    ax.axhline(ci,  color="k", linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(-ci, color="k", linestyle="--", linewidth=0.8)
    ax.set_title(f"Observed vs Theoretical ACF — {title}")
    ax.set_xlabel("Lag (months)")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_residual_acf(model_result, lags=40, title=""):
    """Plot ACF of model residuals with 95% CI.
    Input:  model_result (ARIMAResults); lags (int); title (str)
    Output: matplotlib Figure
    """
    resid = model_result.resid.dropna()
    ci = 1.96 / np.sqrt(len(resid))
    acf_vals = acf(resid, nlags=lags, fft=True)
    lag_arr = np.arange(lags + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(lag_arr, acf_vals, color="steelblue", alpha=0.7, width=0.6)
    ax.axhline(ci,  color="r", linestyle="--", linewidth=1, label="95% CI")
    ax.axhline(-ci, color="r", linestyle="--", linewidth=1)
    ax.set_title(f"Residual ACF — {title}")
    ax.set_xlabel("Lag (months)")
    ax.legend()
    plt.tight_layout()
    return fig

def portmanteau_test(model_result, lags=20):
    """Ljung-Box portmanteau test for residual independence.
    Input:  model_result (ARIMAResults); lags (int) number of lags to test
    Output: pd.DataFrame with columns lb_stat, lb_pvalue indexed by lag
    """
    return acorr_ljungbox(model_result.resid.dropna(), lags=lags, return_df=True)

def normality_test(model_result):
    """Probability plot and PPCC normality test on model residuals.
    Input:  model_result (ARIMAResults)
    Output: (ppcc float, p_value float, Figure)
    """
    resid = model_result.resid.dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    (osm, osr), _ = stats.probplot(resid, dist="norm", plot=ax)
    ppcc, p_val = stats.pearsonr(osm, osr)
    ax.set_title(f"Probability Plot  (PPCC={ppcc:.4f}, p={p_val:.4f})")
    plt.tight_layout()
    return ppcc, p_val, fig
