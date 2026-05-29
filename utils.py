# utils.py — shared definitions for Module2.ipynb

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fmin
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

def plot_trend_fit(series, trend_info, title="", ax=None):
    """Plot a time series with its fitted linear trend line.
    Input:  series (pd.Series); trend_info (dict from fit_linear_trend); title (str);
            ax (matplotlib Axes or None) — if None, creates a new figure
    Output: matplotlib Figure (None when ax is provided)
    """
    s = series.dropna()
    trend_line = trend_info["slope"] * trend_info["x_numeric"] + trend_info["intercept"]
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = None
    ax.plot(trend_info["index"], s.values, alpha=0.7, label="data")
    ax.plot(trend_info["index"], trend_line, "r--",
            label=f"trend  p={trend_info['p_value']:.3f}  R²={trend_info['r_squared']:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.legend()
    if standalone:
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

def plot_acf_pacf(corr_dict, title="", axes=None):
    """Plot ACF and PACF side by side with 95% CI.
    Input:  corr_dict (dict from compute_acf_pacf); title (str);
            axes (array of 2 Axes or None) — if None, creates a new figure
    Output: matplotlib Figure (None when axes is provided)
    """
    lags = corr_dict["lags"]
    ci = corr_dict["conf_bound"]
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    else:
        fig = None
    for ax, key, label in zip(axes, ["acf", "pacf"], ["ACF", "PACF"]):
        vals = corr_dict[key] if key == "acf" else corr_dict[key][1:]
        x = lags if key == "acf" else lags[1:]
        ax.bar(x, vals, color="steelblue", alpha=0.7, width=0.6)
        ax.axhline(ci,  color="r", linestyle="--", linewidth=1, label="95% CI")
        ax.axhline(-ci, color="r", linestyle="--", linewidth=1)
        ax.set_title(f"{label} — {title}")
        ax.set_xlabel("Lag (months)")
        ax.legend(fontsize=7)
    if standalone:
        plt.tight_layout()
    return fig

def select_ar_order(series, max_p=4):
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
                res = ARIMA(vals, order=(p, 0, 0)).fit(method="innovations_mle")
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
                    res = ARIMA(vals, order=(p, 0, q)).fit(method="innovations_mle")
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
    return ARIMA(series.dropna().values, order=(order, 0, 0)).fit(
        method="innovations_mle", low_memory=True)

def fit_arma(series, p, q):
    """Fit ARMA(p,q) model via ARIMA(p,0,q).
    Input:  series (pd.Series) detrended; p (int) AR order; q (int) MA order
    Output: statsmodels ARIMAResults
    """
    return ARIMA(series.dropna().values, order=(p, 0, q)).fit(
        method="innovations_mle", low_memory=True)


# ── Section 3: Evaluation ─────────────────────────────────────────────────────

def get_theoretical_acf(model_result, lags=40):
    """Compute theoretical ACF from a fitted ARIMA model.
    Input:  model_result (ARIMAResults); lags (int)
    Output: np.array of ACF values, length lags+1
    """
    ar = np.r_[1, -model_result.arparams]
    ma = np.r_[1,  model_result.maparams]
    return ArmaProcess(ar, ma).acf(lags + 1)

def plot_acf_comparison(corr_dict, model_result, title="", ax=None):
    """Plot observed ACF vs theoretical ACF with 95% CI on same axes.
    Input:  corr_dict (dict from compute_acf_pacf); model_result (ARIMAResults); title (str);
            ax (matplotlib Axes or None) — if None, creates a new figure
    Output: matplotlib Figure (None when ax is provided)
    """
    lags = corr_dict["lags"]
    ci = corr_dict["conf_bound"]
    theo = get_theoretical_acf(model_result, lags=len(lags) - 1)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = None
    ax.bar(lags, corr_dict["acf"], alpha=0.5, label="Obs ACF", color="steelblue")
    ax.plot(lags, theo, "r-o", markersize=3, label="Theo ACF")
    ax.axhline(ci,  color="k", linestyle="--", linewidth=0.8, label="95% CI")
    ax.axhline(-ci, color="k", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lag (months)")
    ax.legend(fontsize=7)
    if standalone:
        plt.tight_layout()
    return fig

def plot_residual_acf(model_result, lags=40, title="", ax=None):
    """Plot ACF of model residuals with 95% CI.
    Input:  model_result (ARIMAResults); lags (int); title (str);
            ax (matplotlib Axes or None) — if None, creates a new figure
    Output: matplotlib Figure (None when ax is provided)
    """
    resid = np.asarray(model_result.resid)
    resid = resid[~np.isnan(resid)]
    ci = 1.96 / np.sqrt(len(resid))
    acf_vals = acf(resid, nlags=lags, fft=True)
    lag_arr = np.arange(lags + 1)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = None
    ax.bar(lag_arr, acf_vals, color="steelblue", alpha=0.7, width=0.6)
    ax.axhline(ci,  color="r", linestyle="--", linewidth=1, label="95% CI")
    ax.axhline(-ci, color="r", linestyle="--", linewidth=1)
    ax.set_title(f"Residual ACF — {title}", fontsize=9)
    ax.set_xlabel("Lag (months)")
    ax.legend(fontsize=7)
    if standalone:
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

# Filliben (1975) critical values for the normal PPCC test at α = 0.05.
# Reject H0 (normality) if PPCC < critical value for the given n.
_FILLIBEN_N  = [5,      10,     15,     20,     25,     30,     35,     40,
                45,     50,     60,     75,     100,    150,    200,    300,    500,    1000]
_FILLIBEN_CV = [0.8299, 0.9347, 0.9429, 0.9503, 0.9563, 0.9604, 0.9643, 0.9670,
                0.9693, 0.9715, 0.9740, 0.9771, 0.9812, 0.9854, 0.9879, 0.9905, 0.9935, 0.9960]

def normality_test(model_result, ax=None, title="", plot=True):
    """Probability plot and PPCC normality test on model residuals (Filliben 1975).
    H0: residuals are normally distributed.
    Test statistic: r = Pearson correlation between ordered residuals and Blom normal quantiles.
    Decision rule (α=5%): reject H0 if PPCC < r_{n, 0.05} from Filliben's table.
    Input:  model_result (ARIMAResults);
            ax (matplotlib Axes or None) — if None and plot=True, creates a new figure;
            title (str) label prepended to the plot title;
            plot (bool) — set False to compute PPCC/reject only without any figure
    Output: (ppcc float, reject bool, Figure or None)
    """
    resid = np.asarray(model_result.resid)
    resid = resid[~np.isnan(resid)]
    n = len(resid)
    # scipy probplot uses Blom plotting positions: q_i = (i - 3/8)/(n + 1/4) — matches lecture
    (osm, osr), _ = stats.probplot(resid, dist="norm")
    ppcc, _ = stats.pearsonr(osm, osr)
    cv_05 = float(np.interp(n, _FILLIBEN_N, _FILLIBEN_CV))
    reject = bool(ppcc < cv_05)
    if not plot and ax is None:
        return ppcc, reject, None
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = None
    ax.plot(osm, osr, "o", color="steelblue", markersize=4)
    slope, intercept = np.polyfit(osm, osr, 1)
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1)
    label = f"{title}\n" if title else ""
    result = "reject H₀" if reject else "accept H₀"
    ax.set_title(f"{label}PPCC={ppcc:.4f}  CV={cv_05:.4f}  {result}", fontsize=8)
    ax.set_xlabel("Theoretical quantiles (Blom)")
    ax.set_ylabel("Ordered residuals")
    if standalone:
        plt.tight_layout()
    return ppcc, reject, fig


# ── Section 4: Synthetic generation & sediment mass ──────────────────────────

def simulate_arma(model_result, n_months=120, n_simulations=10, seed=42):
    """Generate synthetic time series from a fitted ARMA model.
    Input:  model_result (ARIMAResults); n_months (int) length per series;
            n_simulations (int); seed (int) random seed for reproducibility
    Output: list of n_simulations np.arrays, each of length n_months
    """
    ar = np.r_[1, -model_result.arparams]
    ma = np.r_[1,  model_result.maparams]
    resid = np.asarray(model_result.resid)
    sigma = np.std(resid[~np.isnan(resid)])
    process = ArmaProcess(ar, ma)
    np.random.seed(seed)
    return [process.generate_sample(nsample=n_months, scale=sigma)
            for _ in range(n_simulations)]

def plot_synthetic_vs_observed(observed, simulations, title="", ax=None, n_months=150):
    """Plot synthetic series alongside the observed detrended series.
    Input:  observed (pd.Series) detrended zero-mean; simulations (list of np.array);
            title (str); ax (matplotlib Axes or None) — if None, creates a new figure;
            n_months (int) how many months to show (default 150)
    Output: matplotlib Figure (None when ax is provided)
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = None
    for i, sim in enumerate(simulations):
        ax.plot(sim[:n_months], color="steelblue", alpha=0.35,
                label="Synthetic" if i == 0 else "")
    ax.plot(observed.values[:n_months], color="k", linewidth=1.2, label="Observed")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.legend(fontsize=8)
    if standalone:
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


# ── Section 6 (exploratory): Seasonal adjustment ─────────────────────────────

def compute_seasonal_means(series):
    """Compute mean for each calendar month (seasonal climatology).
    Input:  series (pd.Series) detrended with DatetimeIndex
    Output: pd.Series indexed 1–12 with monthly mean values
    """
    return series.groupby(series.index.month).mean()

def remove_seasonality(series):
    """Monthly standardisation: subtract each month's mean and divide by its std.
    This removes both the seasonal mean and seasonal variance (heteroscedasticity),
    which is the Thomas-Fiering standard for stochastic hydrology.  The stored means
    and stds are used in recompose() to invert the transformation on synthetic series.
    Input:  series (pd.Series) detrended with DatetimeIndex
    Output: (standardised pd.Series,
             seasonal_means pd.Series indexed 1–12,
             seasonal_stds  pd.Series indexed 1–12)
    """
    seasonal_means = series.groupby(series.index.month).mean()
    seasonal_stds  = series.groupby(series.index.month).std()
    month_idx = series.index.month
    adjusted = (series - month_idx.map(seasonal_means)) / month_idx.map(seasonal_stds)
    return adjusted, seasonal_means, seasonal_stds

def remove_seasonality_fourier(series, n_harmonics=2):
    """Remove seasonality by subtracting a truncated Fourier series (harmonic regression).

    Fits: x(t) = Σ_{k=1}^{K} [a_k·cos(2πkt/12) + b_k·sin(2πkt/12)]
    using OLS, then subtracts the fitted cycle.  Unlike Thomas-Fiering monthly
    standardisation, this uses only 2K parameters for the seasonal mean and does
    not rescale the variance — residuals retain their original scale.

    Input:  series (pd.Series) detrended with DatetimeIndex
            n_harmonics (int) K — number of harmonic pairs (default 2)
    Output: (residuals pd.Series,
             seasonal_cycle pd.Series — fitted values at each observation,
             monthly_cycle np.array length 12 — mean fitted value per calendar month 0–11)
    """
    s = series.dropna()
    n = len(s)
    T = 12.0
    t = np.arange(n, dtype=float)
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * t / T))
        cols.append(np.sin(2 * np.pi * k * t / T))
    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, s.values, rcond=None)
    fitted = X @ coeffs
    residuals = pd.Series(s.values - fitted, index=s.index, name=s.name)
    seasonal_cycle = pd.Series(fitted, index=s.index, name="seasonal_cycle")
    m_idx = s.index.month - 1  # 0-indexed
    monthly_cycle = np.array([fitted[m_idx == m].mean() for m in range(12)])
    return residuals, seasonal_cycle, monthly_cycle


def plot_fourier_vs_monthly(series, seasonal_cycle, seasonal_means, title="", ax=None):
    """Bar comparison of Fourier-fitted vs Thomas-Fiering monthly seasonal means.
    Input:  series (pd.Series) detrended;
            seasonal_cycle (pd.Series) Fourier fitted values at each t;
            seasonal_means (pd.Series) Thomas-Fiering monthly means indexed 1–12;
            title (str); ax (matplotlib Axes or None)
    Output: matplotlib Figure (None when ax is provided)
    """
    s = series.dropna()
    m_idx = s.index.month
    fourier_monthly = np.array([seasonal_cycle.values[m_idx == m].mean() for m in range(1, 13)])
    months = np.arange(1, 13)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = None
    obs_monthly = np.array([s[s.index.month == m].mean() for m in range(1, 13)])
    obs_std     = np.array([s[s.index.month == m].std()  for m in range(1, 13)])
    ax.bar(months - 0.2, seasonal_means.values, 0.35,
           label="Thomas-Fiering", color="steelblue", alpha=0.8)
    ax.bar(months + 0.2, fourier_monthly, 0.35,
           label="Fourier fit", color="darkorange", alpha=0.8)
    ax.errorbar(months, obs_monthly, yerr=obs_std, fmt="ko-", markersize=5,
                linewidth=1.2, elinewidth=1.2, capsize=4, capthick=1.2,
                label="Observed mean ± std", zorder=5)
    ax.set_xticks(months)
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_title(title)
    ax.legend(fontsize=8)
    if standalone:
        plt.tight_layout()
    return fig


def seasonal_difference(series, period=12):
    """Seasonal differencing: Y_t = X_t - X_{t-period}.

    WHY: Monthly standardisation (subtract monthly mean, divide by monthly std)
    assumes the seasonal cycle has a fixed shape every year.  For SSC this does
    not hold — summer peaks scale with how much snow fell that winter, so the
    amplitude varies from year to year (multiplicative seasonality).  Monthly
    standardisation therefore leaves residual seasonal spikes in the ACF and
    forces a high AR order to absorb them.

    Seasonal differencing makes no assumption about the shape of the cycle: it
    simply subtracts the value from the same month one year ago, cancelling
    whatever seasonal pattern existed.  The differenced series Y_t represents
    the year-over-year *change* in SSC for that calendar month — a stationary,
    zero-mean quantity that can be modelled with a low-order ARMA.

    COST: loses the first `period` observations and the resulting synthetic
    series must be inverted before use (see invert_seasonal_diff).

    Input:  series (pd.Series) detrended with DatetimeIndex; period (int)
    Output: (differenced pd.Series,
             seed pd.Series — last `period` observed values needed to invert)
    """
    diff = series.diff(period).dropna()
    # Keep last 12 observed values: these seed the inversion X_t = Y_t + X_{t-12}
    seed = series.iloc[-period:]
    return diff, seed

def invert_seasonal_diff(simulations, seed, period=12):
    """Invert seasonal differencing to recover physical-scale synthetic series.

    WHY: the ARMA model was fitted on the differenced series Y_t = X_t - X_{t-12}.
    To generate synthetic SSC in physical (detrended) units we must undo that
    operation.  The recursion X_t = Y_t + X_{t-12} requires knowing X for the
    previous 12 months; we seed those from the last 12 observed detrended values
    so the simulation starts from the current state of the system.

    Input:  simulations (list of np.array) synthetic differenced residuals Y_t;
            seed (pd.Series) last `period` observed detrended X values;
            period (int)
    Output: list of np.array — synthetic X in detrended physical units
    """
    s = seed.values[-period:]
    result = []
    for d_sim in simulations:
        X = np.empty(len(d_sim))
        for t in range(len(d_sim)):
            # For the first 12 steps use observed seed; after that use synthetic X
            prev = s[t] if t < period else X[t - period]
            X[t] = d_sim[t] + prev
        result.append(X)
    return result

def plot_seasonal_pattern(seasonal_means_dict, ylabel, title):
    """Bar plot of monthly climatology for multiple series.
    Input:  seasonal_means_dict (dict {label: pd.Series indexed 1–12});
            ylabel (str); title (str)
    Output: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    months = np.arange(1, 13)
    width = 0.8 / len(seasonal_means_dict)
    for i, (label, sm) in enumerate(seasonal_means_dict.items()):
        ax.bar(months + (i - len(seasonal_means_dict) / 2 + 0.5) * width,
               sm.values, width, label=label, alpha=0.8)
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    return fig


# ── Section 7 (exploratory): Extreme value estimation via synthetic series ────

# Gumbel PPCC critical values r_{n, 0.05} (Looney & Gulledge 1985)
_S7_PPCC_N   = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100, 150, 200]
_S7_PPCC_R05 = [0.9260, 0.9383, 0.9460, 0.9505, 0.9538, 0.9563, 0.9582,
                0.9600, 0.9615, 0.9638, 0.9662, 0.9696, 0.9726, 0.9752]

def _ppcc_crit_s7(n):
    if n <= _S7_PPCC_N[0]:  return _S7_PPCC_R05[0]
    if n >= _S7_PPCC_N[-1]: return _S7_PPCC_R05[-1]
    for k in range(len(_S7_PPCC_N) - 1):
        if _S7_PPCC_N[k] <= n <= _S7_PPCC_N[k + 1]:
            t = (n - _S7_PPCC_N[k]) / (_S7_PPCC_N[k + 1] - _S7_PPCC_N[k])
            return _S7_PPCC_R05[k] + t * (_S7_PPCC_R05[k + 1] - _S7_PPCC_R05[k])

def fit_gumbel_s7(data):
    """MLE Gumbel (EV Type I) fit to annual maxima.
    Input:  data (np.array) annual maximum values
    Output: (loc float, scale float)
    """
    def neg_ll(p):
        if p[1] <= 0: return 1e10
        return -np.sum(stats.gumbel_r.logpdf(data, loc=p[0], scale=p[1]))
    s = np.std(data)
    res = fmin(neg_ll, [np.mean(data) - 0.5772 * s, s], disp=False)
    return float(res[0]), float(res[1])

def gumbel_quantile_s7(T, loc, scale):
    """T-year Gumbel return level for annual maxima.
    Input:  T (float) return period [years]; loc (float); scale (float)
    Output: float
    """
    y_T = -np.log(-np.log(1.0 - 1.0 / T))
    return loc + scale * y_T

def gumbel_se_s7(T, scale, n):
    """Kite (1977) asymptotic SE of the T-year Gumbel quantile.
    Input:  T (float); scale (float); n (int) sample size
    Output: float
    """
    y_T = -np.log(-np.log(1.0 - 1.0 / T))
    return (scale / np.sqrt(n)) * np.sqrt(1.1087 + 0.5140 * y_T + 0.6079 * y_T ** 2)

def ppcc_gumbel_s7(data, loc, scale):
    """PPCC test for Gumbel distribution (Looney & Gulledge 1985).
    Input:  data (np.array); loc (float); scale (float)
    Output: (r float, r_crit float, reject bool)
    """
    n = len(data)
    x = np.sort(data)
    q = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    w = stats.gumbel_r.ppf(q, loc=loc, scale=scale)
    r = float(np.corrcoef(x, w)[0, 1])
    r_crit = _ppcc_crit_s7(n)
    return r, r_crit, bool(r < r_crit)

def generate_q_synthetic_100yr(model_result_sa, seasonal_means, original_mean,
                                n_years=100, n_simulations=10, seed=42):
    """Generate physical Q synthetic series from a seasonally adjusted ARMA model.
    Input:  model_result_sa — ARIMAResults fitted on seasonally adjusted series
            seasonal_means  — pd.Series indexed 1–12 (monthly climatology of detrended series)
            original_mean   — float (mean subtracted during detrending)
            n_years         — int, length of each simulation in years
            n_simulations   — int, number of realisations
            seed            — int
    Output: list of n_simulations np.arrays of length n_years*12 (physical Q [m³/s])
    """
    n_months = n_years * 12
    sims_sa = simulate_arma(model_result_sa, n_months=n_months,
                             n_simulations=n_simulations, seed=seed)
    sm = seasonal_means.values  # length 12
    seasonal_cycle = np.tile(sm, n_years)  # exactly n_months long
    return [sim + seasonal_cycle + original_mean for sim in sims_sa]

def extract_annual_maxima(simulations):
    """Extract annual maxima from monthly synthetic series.
    Input:  simulations — list of np.arrays, each length = multiple of 12
    Output: np.array of all annual maxima (all simulations concatenated)
    """
    maxima = []
    for sim in simulations:
        n_years = len(sim) // 12
        for y in range(n_years):
            maxima.append(sim[y * 12:(y + 1) * 12].max())
    return np.array(maxima)

def return_period_comparison_table(obs_data, syn_data, T_list=None, z90=1.645):
    """Return period table comparing Gumbel fits to observed and synthetic annual maxima.
    Input:  obs_data (np.array); syn_data (np.array); T_list (list); z90 (float)
    Output: pd.DataFrame indexed by T [yr]
    """
    if T_list is None:
        T_list = [10, 30, 50, 100, 300]
    loc_o, sc_o = fit_gumbel_s7(obs_data)
    loc_s, sc_s = fit_gumbel_s7(syn_data)
    n_o, n_s = len(obs_data), len(syn_data)
    rows = []
    for T in T_list:
        xo = gumbel_quantile_s7(T, loc_o, sc_o)
        so = gumbel_se_s7(T, sc_o, n_o)
        xs = gumbel_quantile_s7(T, loc_s, sc_s)
        ss = gumbel_se_s7(T, sc_s, n_s)
        rows.append({
            "T (yr)": T,
            "Obs x_T":  round(xo, 1),
            "Obs CI_lo": round(xo - z90 * so, 1),
            "Obs CI_hi": round(xo + z90 * so, 1),
            "Syn x_T":  round(xs, 1),
            "Syn CI_lo": round(xs - z90 * ss, 1),
            "Syn CI_hi": round(xs + z90 * ss, 1),
        })
    return pd.DataFrame(rows).set_index("T (yr)")

def generate_q_log_synthetic_100yr(model_result_sa, seasonal_means_log, original_mean_log,
                                    n_years=100, n_simulations=10, seed=42):
    """Generate physical Q by simulating in log-space then back-transforming.
    Workflow: ARMA residuals (log-space) → + seasonal_means_log + original_mean_log
              → synthetic log(Q) → exp() → physical Q [m³/s]
    Input:  model_result_sa      — ARIMAResults fitted on log(Q) seasonally adjusted series
            seasonal_means_log   — pd.Series indexed 1–12 (monthly climatology of detrended log(Q))
            original_mean_log    — float (mean of detrended log(Q) that was subtracted)
            n_years, n_simulations, seed — same as generate_q_synthetic_100yr
    Output: list of n_simulations np.arrays of physical Q [m³/s], length n_years*12
    """
    n_months = n_years * 12
    sims_sa = simulate_arma(model_result_sa, n_months=n_months,
                             n_simulations=n_simulations, seed=seed)
    sm = seasonal_means_log.values
    seasonal_cycle = np.tile(sm, n_years)
    return [np.exp(sim + seasonal_cycle + original_mean_log) for sim in sims_sa]

def plot_return_period_comparison(obs_data, syn_data, title="",
                                   T_list=None, z90=1.645):
    """Semilog return period plot: observed vs synthetic annual maxima with Gumbel fits.
    Input:  obs_data (np.array); syn_data (np.array); title (str);
            T_list (list); z90 (float) for 90% CI
    Output: matplotlib Figure
    """
    if T_list is None:
        T_list = [10, 30, 50, 100, 300]
    T_curve = np.logspace(np.log10(2), np.log10(400), 300)
    loc_o, sc_o = fit_gumbel_s7(obs_data)
    loc_s, sc_s = fit_gumbel_s7(syn_data)
    n_o, n_s = len(obs_data), len(syn_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for data, loc, scale, n, color, label in [
        (obs_data, loc_o, sc_o, n_o, "steelblue",  f"Observed (n={n_o} yr)"),
        (syn_data, loc_s, sc_s, n_s, "darkorange", f"Synthetic (n={n_s} yr)"),
    ]:
        x_c  = np.array([gumbel_quantile_s7(T, loc, scale) for T in T_curve])
        se_c = np.array([gumbel_se_s7(T, scale, n)         for T in T_curve])
        ax.semilogx(T_curve, x_c, color=color, linewidth=2, label=label)
        ax.fill_between(T_curve, x_c - z90 * se_c, x_c + z90 * se_c,
                        color=color, alpha=0.15)
        ii  = np.arange(1, n + 1)
        q   = (ii - 0.44) / (n + 0.12)
        T_emp = 1.0 / (1.0 - q)
        ax.scatter(T_emp, np.sort(data), color=color, s=12, alpha=0.4, zorder=4)

    for T in T_list:
        ax.axvline(T, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_xlabel("Return period T [years]")
    ax.set_ylabel("Annual maximum Q [m³/s]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return fig
