# utils.py — shared definitions for Module2.ipynb

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def load_bafu_q(path="Assignment 1-20260327/BAFU_Diepoldsau/Q__Diepoldsau_m3s.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = ["Q"]
    return df


def plot_autocorrelation(series, lags=100, resample="D", title="Autocorrelation"):
    if resample:
        series = series.resample(resample).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    plt.tight_layout()
    plt.show()
