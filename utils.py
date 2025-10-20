from tinyshift.series import (
    adi_cv,
    theoretical_limit,
    foreca,
    stability_index,
    hurst_exponent,
)
from statsmodels.tsa.stattools import adfuller, acf, pacf
import plotly.graph_objs as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa import seasonal
from tinyshift.series import trend_significance
from statsmodels.stats.diagnostic import acorr_ljungbox

import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from typing import Union, List, Optional


def seasonal_decompose(
    X: Union[np.ndarray, List[float], pd.Series],
    model: str = "additive",
    filt: Optional[np.ndarray] = None,
    period: int = None,
    two_sided: bool = True,
    extrapolate_trend: int = 0,
    height: int = 1200,
    width: int = 1300,
    ljung_lags: int = 10,
    fig_type: Optional[str] = None,
):
    """
    Performs seasonal decomposition of a time series and plots the components.

    This function uses the `seasonal_decompose` method from statsmodels to separate the
    time series into observed, trend, seasonal, and residual components.
    Additionally, it calculates trend significance and the Ljung-Box test for
    residuals, displaying a summary in the plot.

    Parameters
    ----------
    X : array-like
        The time series to be decomposed. Expected to be an object with an index
        (e.g., pandas Series).
    model : {"additive", "multiplicative"}, default="additive"
        Type of seasonal model. If "additive", $X = T + S + R$. If
        "multiplicative", $X = T \cdot S \cdot R$.
    filt : array-like, optional
        Moving average filter for calculating the trend component. By default,
        a symmetric filter is used.
    period : int, optional
        Period of the series (number of observations per cycle). If `None` and `X` is
        a pandas Series, the period is inferred from the index frequency.
    two_sided : bool, default=True
        If `True` (default), uses a centered moving average filter. If `False`, uses
        a causal filter (future only).
    extrapolate_trend : int or str, default=0
        Number of points at the beginning and end to extrapolate the trend. If 0 (default),
        the trend is `NaN` at these extremes.
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    ljung_lags : int, default=10
        The number of lags to be used in the Ljung-Box test for residuals.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    The resulting plot is a Plotly `make_subplots` with 5 subplots:
    - Observed
    - Trend
    - Seasonal
    - Residuals
    - Summary (includes trend significance - $R^2$ and p-value - and the
      Ljung-Box test for residual autocorrelation).
    """

    index = X.index if hasattr(X, "index") else list(range(len(X)))

    if X.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    result = seasonal.seasonal_decompose(
        X,
        model=model,
        filt=filt,
        period=period,
        two_sided=two_sided,
        extrapolate_trend=extrapolate_trend,
    )
    fig = sp.make_subplots(
        rows=5,
        cols=1,
        subplot_titles=[
            "Observed",
            "Trend",
            "Seasonal",
            "Residuals",
            "Summary",
        ],
        row_heights=[4, 4, 4, 4, 1],
    )

    r_squared, p_value = trend_significance(X)
    trend_results = f"R²={r_squared:.4f}, p={p_value:.4f}"
    resid = result.resid[~np.isnan(result.resid)]
    ljung_box = acorr_ljungbox(resid, lags=[ljung_lags])
    ljung_stat, p_value = (
        ljung_box["lb_stat"].values[0],
        ljung_box["lb_pvalue"].values[0],
    )
    ljung_box = f"Stats={ljung_stat:.4f}, p={p_value:.4f}"
    summary = "<br>".join(
        [
            f"<b>{k}</b>: {v}"
            for k, v in {
                "Trend Significance": trend_results,
                "Ljung-Box Test": ljung_box,
            }.items()
        ]
    )

    for i, col in enumerate(["observed", "trend", "seasonal", "resid"]):
        color = colors[(i - 1) % num_colors]
        fig.add_trace(
            go.Scatter(
                x=index,
                y=getattr(result, col),
                mode="lines",
                hovertemplate=f"{col.capitalize()}: " + "%{y}<extra></extra>",
                line=dict(color=color),
            ),
            row=i + 1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=[0], y=[0], text=[summary], mode="text", showlegend=False),
        row=5,
        col=1,
    )

    fig.update_xaxes(visible=False, row=5, col=1)
    fig.update_yaxes(visible=False, row=5, col=1)

    fig.update_layout(
        title="Seasonal Decomposition",
        height=height,
        width=width,
        showlegend=False,
        hovermode="x",
    )

    return fig.show(fig_type)


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


def stationarity_check(
    df,
    height=1200,
    width=1300,
    fig_type=None,
):
    """
    Creates interactive ACF and PACF plots with ADF test results for multiple series.

    This function generates a comprehensive diagnostic visualization to assess the
    stationarity and autocorrelation structure of multiple time series in a single panel.
    The plot includes the series itself, its autocorrelation function (ACF) and partial
    autocorrelation function (PACF), and a summary of the Augmented Dickey-Fuller (ADF)
    test results.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the time series data. The index is used
        as the time axis.
    variables : list of str
        List of column names (variables) from the DataFrame to be analyzed.
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    The function generates a subplot structure where each row corresponds to a
    variable and displays:
    1. The Time Series.
    2. The Autocorrelation Function (ACF).
    3. The Partial Autocorrelation Function (PACF).
    The last row contains a summary of the ADF test results (statistic and p-value)
    for each variable, used to check for stationarity.
    """

    N = len(df.columns)
    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    def create_acf_pacf_traces(X, nlags=30, color=None):
        """
        Helper function to create ACF and PACF traces with confidence intervals.
        """

        N = len(X)
        conf = 1.96 / np.sqrt(N)
        acf_vals = acf(X, nlags=nlags)
        pacf_vals = pacf(X, nlags=nlags, method="yw")

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
    for var in df.columns:
        subplot_titles.extend([f"Series ({var})", f"ACF ({var})", f"PACF ({var})"])
    subplot_titles.extend(["ADF Results Summary", "", ""])

    fig = sp.make_subplots(rows=N + 1, cols=3, subplot_titles=subplot_titles)

    adf_results = {}

    for i, var in enumerate(df.columns, start=1):
        X = df[var].dropna()
        adf_stat, p_value = adfuller(X)[:2]
        adf_results[var] = f"ADF={adf_stat:.4f}, p={p_value:.4f}"
        color = colors[(i - 1) % num_colors]

        fig.add_trace(
            go.Scatter(
                x=X.index,
                y=X,
                mode="lines",
                name=var,
                showlegend=False,
                line=dict(color=color),
            ),
            row=i,
            col=1,
        )

        acf_values, pacf_values, conf_up, conf_lo = create_acf_pacf_traces(
            X, color=color
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
        height=height,
        width=width,
        showlegend=False,
    )

    for row in range(1, N + 1):
        fig.update_xaxes(title_text="Date", row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=1)
        fig.update_xaxes(title_text="Lag", row=row, col=2)
        fig.update_xaxes(title_text="Lag", row=row, col=3)
        fig.update_yaxes(title_text="ACF", row=row, col=2)
        fig.update_yaxes(title_text="PACF", row=row, col=3)

    fig.update_xaxes(visible=False, row=N + 1, col=1)
    fig.update_yaxes(visible=False, row=N + 1, col=1)

    return fig.show(fig_type)


def add_fourier_seasonality(df, time_col, seasonality):
    """
    Adds Fourier-based seasonal features to the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with time column
    time_col : str
        Name of the datetime column
    seasonalities : list, optional
        List of seasonalities to include. Options:
        ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        Default: ['weekly', 'yearly']

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Fourier seasonal features

    Examples:
    ---------
    # Basic usage with default seasonalities
    df = add_fourier_seasonality(df, 'ds')

    # Custom seasonalities
    df = add_fourier_seasonality(df, 'ds', seasonalities=['weekly', 'monthly', 'yearly'])

    # All seasonalities
    df = add_fourier_seasonality(df, 'ds', seasonalities=['daily', 'weekly', 'monthly', 'quarterly', 'yearly'])
    """
    df = df.copy()

    seasonality_config = {
        "daily": {"period": 24, "value_func": lambda dt: dt.hour, "name": "daily"},
        "weekly": {
            "period": 7,
            "value_func": lambda dt: dt.dayofweek,
            "name": "weekly",
        },
        "monthly": {"period": 12, "value_func": lambda dt: dt.month, "name": "monthly"},
        "quarterly": {
            "period": 4,
            "value_func": lambda dt: dt.quarter,
            "name": "quarterly",
        },
        "yearly": {
            "period": 365,
            "value_func": lambda dt: dt.dayofyear,
            "name": "yearly",
        },
    }

    for season in seasonality:
        if season not in seasonality_config:
            raise ValueError(
                f"Unknown seasonality: {season}. "
                f"Available options: {list(seasonality_config.keys())}"
            )

        config = seasonality_config[season]
        period = config["period"]
        values = config["value_func"](df[time_col].dt)
        name = config["name"]

        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)

    return df


def generate_lag(X, lag=1):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim > 1:
        raise ValueError("Input array must be one-dimensional.")

    return np.concatenate((np.nan * np.ones(lag), (X[lag:] - X[:-lag])))
