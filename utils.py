from tinyshift.series import (
    adi_cv,
    theoretical_limit,
    foreca,
    stability_index,
    hurst_exponent,
)
from statsmodels.tsa.stattools import adfuller, acf, pacf
import holidays
import plotly.graph_objs as go
import plotly.subplots as sp
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


def plot_acf_pacf_adf(df, variables, fig_type=None):
    """
    Create ACF/PACF plots with ADF test results for multiple variables.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input dataframe containing the time series data
    variables : list
        List of column names to analyze
    fig_type : str, optional
        Figure display type for plotly (e.g., 'browser', 'png', etc.)

    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive plot showing time series, ACF, PACF, and ADF results
    """
    N = len(variables)

    def create_acf_pacf_traces(data, nlags=30, color=None):
        N = len(data)
        conf = 1.96 / np.sqrt(N)
        acf_vals = acf(data, nlags=nlags)
        pacf_vals = pacf(data, nlags=nlags, method="yw")

        acf_bar = go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color=color)
        pacf_bar = go.Bar(
            x=list(range(len(pacf_vals))), y=pacf_vals, marker_color=color
        )

        band_upper = go.Scatter(
            x=list(range(nlags + 1)),
            y=[conf] * (nlags + 1),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
        band_lower = go.Scatter(
            x=list(range(nlags + 1)),
            y=[-conf] * (nlags + 1),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )

        return acf_bar, pacf_bar, band_upper, band_lower

    subplot_titles = []
    for var in variables:
        subplot_titles.extend([f"Series ({var})", f"ACF ({var})", f"PACF ({var})"])
    subplot_titles.extend(["ADF Results Summary", "", ""])

    fig = sp.make_subplots(rows=N + 1, cols=3, subplot_titles=subplot_titles)

    adf_results = {}

    for i, var in enumerate(variables, start=1):
        X = df[var].dropna()
        adf_stat, p_value = adfuller(X)[:2]
        adf_results[var] = f"ADF={adf_stat:.4f}, p={p_value:.4f}"
        fig.add_trace(
            go.Scatter(
                x=X.index,
                y=X,
                mode="lines",
                name=var,
                showlegend=False,
                line=dict(color="orange"),
            ),
            row=i,
            col=1,
        )

        acf_values, pacf_values, conf_up, conf_lo = create_acf_pacf_traces(
            X, color="orange"
        )

        fig.add_trace(acf_values, row=i, col=2)
        fig.add_trace(pacf_values, row=i, col=3)
        fig.add_trace(conf_up, row=i, col=2)
        fig.add_trace(conf_lo, row=i, col=2)
        fig.add_trace(conf_up, row=i, col=3)
        fig.add_trace(conf_lo, row=i, col=3)

    adf_text = "<br>".join([f"<b>{k}</b>: {v}" for k, v in adf_results.items()])

    fig.add_trace(
        go.Scatter(x=[0], y=[0], text=[adf_text], mode="text", showlegend=False),
        row=N + 1,
        col=1,
    )

    fig.update_layout(
        title="ACF/PACF with ADF Summary",
        height=1200,
        width=1300,
        showlegend=False,
    )

    for row in range(1, N + 1):
        fig.update_xaxes(title_text="Date", row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=1)
        fig.update_xaxes(title_text="Lag", row=row, col=2)
        fig.update_xaxes(title_text="Lag", row=row, col=3)
        fig.update_yaxes(title_text="ACF", row=row, col=2)
        fig.update_yaxes(title_text="PACF", row=row, col=3)

    fig.update_xaxes(visible=False, row=4, col=1)
    fig.update_yaxes(visible=False, row=4, col=1)

    return fig.show(fig_type)


def add_in_date_information(df, time_col):
    """
    Adds date-based features to the dataframe
    """
    df = df.copy()
    holidays_br = holidays.country_holidays("Brazil")
    df["month"] = df[time_col].dt.month
    df["is_holiday"] = np.array([timestamp in holidays_br for timestamp in df["ds"]])
    df["is_month_end"] = df[time_col].dt.is_month_end

    # Cyclical encoding for day of week (weekly sensasonality) and day of year (yearly seasonality)
    df["dow_sin"] = np.sin(2 * np.pi * df[time_col].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df[time_col].dt.dayofweek / 7)

    df["yr_sin"] = np.sin(2 * np.pi * df[time_col].dt.dayofyear / 12)
    df["yr_cos"] = np.cos(2 * np.pi * df[time_col].dt.dayofyear / 12)
    return df


def generate_lag(X, lag=1):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim > 1:
        raise ValueError("Input array must be one-dimensional.")

    return np.concatenate((np.nan * np.ones(lag), (X[lag:] - X[:-lag])))
