# utils.py — shared definitions for Module2.ipynb

import warnings
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
    return df.resample("M").mean()

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
    t["original_mean"] = float(s.mean())
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

def select_ar_order(series, max_p=13):
    """Select AR order p minimising AIC over p = 1..max_p.
    Input:  series (pd.Series) detrended; max_p (int)
    Output: (best_p int, aic_list list of (p, aic))
    """
    vals = series.dropna().values
    aics = []
    for p in range(1, max_p + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(vals, order=(p, 0, 0)).fit()
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
    vals = series.dropna().values
    table = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(vals, order=(p, 0, q)).fit()
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
    return ARIMA(series.dropna().values, order=(order, 0, 0)).fit()

def fit_arma(series, p, q):
    """Fit ARMA(p,q) model via ARIMA(p,0,q).
    Input:  series (pd.Series) detrended; p (int) AR order; q (int) MA order
    Output: statsmodels ARIMAResults
    """
    return ARIMA(series.dropna().values, order=(p, 0, q)).fit()


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
    resid = np.asarray(model_result.resid)
    resid = resid[~np.isnan(resid)]
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
    resid = np.asarray(model_result.resid)
    resid = resid[~np.isnan(resid)]
    return acorr_ljungbox(resid, lags=lags, return_df=True)

def normality_test(model_result):
    """Probability plot and PPCC normality test on model residuals.
    Input:  model_result (ARIMAResults)
    Output: (ppcc float, p_value float, Figure)
    """
    resid = np.asarray(model_result.resid)
    resid = resid[~np.isnan(resid)]
    fig, ax = plt.subplots(figsize=(5, 5))
    (osm, osr), _ = stats.probplot(resid, dist="norm", plot=ax)
    ppcc, p_val = stats.pearsonr(osm, osr)
    ax.set_title(f"Probability Plot  (PPCC={ppcc:.4f}, p={p_val:.4f})")
    plt.tight_layout()
    return ppcc, p_val, fig


# ── Section 4: Synthetic generation & sediment mass ──────────────────────────

def simulate_arma(model_result, n_months=120, n_simulations=10, seed=42):
    """Generate synthetic time series from a fitted ARMA model.
    Input:  model_result (ARIMAResults); n_months (int) length per series;
            n_simulations (int); seed (int) random seed for reproducibility
    Output: list of n_simulations np.arrays, each of length n_months
    """
    ar = np.r_[1, -model_result.arparams]
    ma = np.r_[1,  model_result.maparams]
    sigma = np.sqrt(model_result.sigma2)
    process = ArmaProcess(ar, ma)
    np.random.seed(seed)
    return [process.generate_sample(nsample=n_months, scale=sigma)
            for _ in range(n_simulations)]

def plot_synthetic_vs_observed(observed, simulations, title=""):
    """Plot synthetic series alongside the observed detrended series.
    Input:  observed (pd.Series) detrended zero-mean; simulations (list of np.array);
            title (str)
    Output: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, sim in enumerate(simulations):
        ax.plot(sim, color="steelblue", alpha=0.35,
                label="Synthetic" if i == 0 else "")
    ax.plot(observed.values, color="k", linewidth=1.2, label="Observed (normalised)")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.legend()
    plt.tight_layout()
    return fig

def compare_statistics(observed, simulations):
    """Compare mean, std, min, max of observed vs synthetic ensemble.
    Input:  observed (pd.Series); simulations (list of np.array)
    Output: pd.DataFrame with one row per source
    """
    rows = [{"source": "observed", "mean": observed.mean(),
             "std": observed.std(), "min": observed.min(), "max": observed.max()}]
    all_syn = np.concatenate(simulations)
    rows.append({"source": "synthetic (all)", "mean": all_syn.mean(),
                 "std": all_syn.std(), "min": all_syn.min(), "max": all_syn.max()})
    return pd.DataFrame(rows).set_index("source")

def compute_sediment_mass(q_monthly, ssc_monthly):
    """Compute monthly sediment mass flux M = Q * C [kg/s] over overlapping period.
    Input:  q_monthly (pd.Series) Q [m³/s]; ssc_monthly (pd.Series) SSC [g/L]
    Output: pd.Series M [kg/s]
    """
    q, c = q_monthly.align(ssc_monthly, join="inner")
    return (q * c).dropna()

def monthly_yearly_yields(mass_series):
    """Compute mean M grouped by calendar month and by year.
    Input:  mass_series (pd.Series) M [kg/s] with DatetimeIndex
    Output: (monthly pd.Series index 1-12 [kg/s], yearly pd.Series [kg/s])
    """
    monthly = mass_series.groupby(mass_series.index.month).mean()
    yearly  = mass_series.resample("A").mean()
    return monthly, yearly

def synthetic_mass_yields(q_sims, ssc_sims, q_mean, ssc_mean):
    """Compute mass yield statistics from synthetic Q and SSC simulations.
    Input:  q_sims (list of np.array) synthetic Q (zero-mean);
            ssc_sims (list of np.array) synthetic SSC (zero-mean);
            q_mean (float) original Q mean to restore physical scale;
            ssc_mean (float) original SSC mean to restore physical scale
    Output: (monthly np.array size 12, yearly_mean float)
    """
    all_M = []
    for q_sim, ssc_sim in zip(q_sims, ssc_sims):
        q_act = np.clip(q_sim + q_mean, 0, None)
        c_act = np.clip(ssc_sim + ssc_mean, 0, None)
        all_M.append(q_act * c_act)
    all_M = np.array(all_M)
    monthly = np.array([all_M[:, m::12].mean() for m in range(12)])
    n_years = all_M.shape[1] // 12
    yearly_mean = all_M[:, :n_years * 12].reshape(-1, 12).mean(axis=1).mean()
    return monthly, yearly_mean

def plot_mass_yields(obs_monthly, obs_yearly, syn_monthly, syn_yearly_mean, title=""):
    """Plot observed vs synthetic monthly and yearly sediment mass yields.
    Input:  obs_monthly (pd.Series) calendar-month means [kg/s];
            obs_yearly (pd.Series) yearly means [kg/s];
            syn_monthly (np.array size 12); syn_yearly_mean (float);
            title (str)
    Output: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    months = np.arange(1, 13)
    axes[0].bar(months - 0.2, obs_monthly.values, 0.4,
                label="Observed", color="steelblue")
    axes[0].bar(months + 0.2, syn_monthly, 0.4,
                label="Synthetic", color="orange", alpha=0.8)
    axes[0].set_title(f"Monthly mean M [kg/s] — {title}")
    axes[0].set_xlabel("Month")
    axes[0].legend()

    axes[1].bar(obs_yearly.index.year, obs_yearly.values,
                label="Observed", color="steelblue")
    axes[1].axhline(syn_yearly_mean, color="orange", linestyle="--",
                    linewidth=2, label=f"Synthetic mean ({syn_yearly_mean:.2f})")
    axes[1].set_title(f"Yearly mean M [kg/s] — {title}")
    axes[1].set_xlabel("Year")
    axes[1].legend()
    plt.tight_layout()
    return fig


# ── Section 5: Independence of Q and C ───────────────────────────────────────

def compute_correlation(q_monthly, ssc_monthly):
    """Compute Pearson and Spearman correlation between monthly Q and SSC.
    Input:  q_monthly (pd.Series); ssc_monthly (pd.Series)
    Output: dict with pearson_r, pearson_p, spearman_r, spearman_p
    """
    q, c = q_monthly.align(ssc_monthly, join="inner")
    mask = q.notna() & c.notna()
    q, c = q[mask].values, c[mask].values
    pr, pp = stats.pearsonr(q, c)
    sr, sp = stats.spearmanr(q, c)
    return dict(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp)

def plot_joint_distribution(q_monthly, ssc_monthly, title=""):
    """Scatter plot of (Q, C) with marginal histograms.
    Input:  q_monthly (pd.Series); ssc_monthly (pd.Series); title (str)
    Output: matplotlib Figure
    """
    q, c = q_monthly.align(ssc_monthly, join="inner")
    mask = q.notna() & c.notna()
    q, c = q[mask].values, c[mask].values

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(q, c, alpha=0.3, s=10, color="steelblue")
    ax_main.set_xlabel("Q [m³/s]")
    ax_main.set_ylabel("SSC [g/L]")
    ax_top.hist(q, bins=30, color="steelblue", alpha=0.7)
    ax_top.set_title(title)
    ax_right.hist(c, bins=30, orientation="horizontal", color="steelblue", alpha=0.7)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    plt.tight_layout()
    return fig
