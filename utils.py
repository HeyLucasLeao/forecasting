from tinyshift.series import (
    adi_cv,
    theoretical_limit,
    foreca,
    stability_index,
    hurst_exponent,
)
import pandas as pd
import numpy as np


def remove_leading_zeros(group):
    """
    Removes leading zeros from series
    """
    first_non_zero_index = group["y"].ne(0).idxmax()
    return group.loc[first_non_zero_index:]


def is_obsolete(group, days_obsoletes):
    """
    Identify obsolote series
    """
    last_date = group["ds"].max()
    cutoff_date = last_date - pd.Timedelta(days=days_obsoletes)
    recent_data = group[group["ds"] >= cutoff_date]
    return (recent_data["y"] == 0).all()


def forecastability(X):
    """
    Calculate forecastability metrics for a time series.
    """
    return {
        "foreCA": foreca(X),
        "theoretical_limit": theoretical_limit(X),
        "stability_index": stability_index(X, detrend=True),
        "adi_cv": adi_cv(X),
        "hurst_exponent": hurst_exponent(X),
    }


def generate_lag(X, lag=1):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim > 1:
        raise ValueError("Input array must be one-dimensional.")

    return np.concatenate((np.nan * np.ones(lag), (X[lag:] - X[:-lag])))
