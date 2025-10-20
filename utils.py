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
from statsmodels.tsa.seasonal import seasonal_decompose


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
    colors = px.colors.qualitative.T10
    num_colors = len(colors)

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

    fig.update_xaxes(visible=False, row=N + 1, col=1)
    fig.update_yaxes(visible=False, row=N + 1, col=1)

    return fig.show(fig_type)


def plot_seasonal_decompose(
    x,
    model="additive",
    filt=None,
    period=None,
    two_sided=True,
    extrapolate_trend=0,
    fig_type=None,
):

    result = seasonal_decompose(
        x,
        model=model,
        filt=filt,
        period=period,
        two_sided=two_sided,
        extrapolate_trend=extrapolate_trend,
    )
    fig = sp.make_subplots(
        rows=4, cols=1, subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"]
    )
    for idx, col in enumerate(["observed", "trend", "seasonal", "resid"]):
        fig.add_trace(
            go.Scatter(
                x=result.observed.index,
                y=getattr(result, col),
                mode="lines",
                hovertemplate=f"{col.capitalize()}: " + "%{y}<extra></extra>",
            ),
            row=idx + 1,
            col=1,
        )

    fig.update_layout(
        title="Seasonal Decomposition",
        height=1200,
        width=1300,
        showlegend=False,
        hovermode="x",
    )

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

    # Seasonality configurations
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

    # Generate Fourier features for each requested seasonality
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

        # Generate sin and cos components
        df[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * values / period)

    return df


def generate_lag(X, lag=1):
    X = np.asarray(X, dtype=np.float64)

    if X.ndim > 1:
        raise ValueError("Input array must be one-dimensional.")

    return np.concatenate((np.nan * np.ones(lag), (X[lag:] - X[:-lag])))
