def seasonal_decompose(
    df: pd.DataFrame,
    periods: int | List[int],
    freq: str,
    identifier: str,
    time_col: str = "ds",
    target_col: str = "y",
    id_col: str = "unique_id",
    trend_forecaster: Optional[object] = AutoETS(model="ZZN"),
    njobs: int = -1,
    height: int = 1200,
    width: int = 1300,
    ljung_lags: int = 10,
    fig_type: Optional[str] = None,
):
    """
    Performs seasonal decomposition of a time series using MSTL and plots the components.

    This function uses the MSTL (Multiple Seasonal-Trend decomposition using Loess) method
    to separate a time series into trend, seasonal, and residual components for a specific
    identifier. It calculates trend significance and performs the Ljung-Box test for
    residuals, displaying a summary in the plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the time series data with columns for time,
        target values, and identifiers.
    periods : int or list of int
        Period(s) of the seasonal components. For multiple seasonality, provide
        a list of integers (e.g., [7, 365] for weekly and yearly patterns).
    freq : str
        Frequency of the time series (e.g., 'D' for daily, 'H' for hourly).
        Used by StatsForecast for proper time series handling.
    identifier : str
        Unique identifier value to filter the DataFrame for decomposition.
        Must exist in the `id_col` column.
    time_col : str, default='ds'
        Name of the column containing time/date values.
    target_col : str, default='y'
        Name of the column containing the target variable to decompose.
    id_col : str, default='unique_id'
        Name of the column containing unique identifiers.
    trend_forecaster : object, optional, default=AutoETS(model="ZZN")
        Forecasting model for the trend component. If None, uses simple trend extraction.
    njobs : int, default=-1
        Number of jobs for parallel processing. -1 uses all available processors.
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    ljung_lags : int, default=10
        Number of lags to use in the Ljung-Box test for residual autocorrelation.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If identifier is None or not found in the DataFrame.

    Notes
    -----
    The resulting plot contains subplots for each decomposition component plus a summary:
    - Each component from the MSTL decomposition (trend, seasonal patterns, residuals)
    - Summary panel showing trend significance (R² and p-value) and Ljung-Box test
      results for residual autocorrelation analysis.

    The MSTL method is particularly useful for time series with multiple seasonal patterns
    and provides robust decomposition even in the presence of outliers.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if identifier is None:
        raise ValueError("An identifier must be provided to filter the DataFrame.")

    df = df.copy()
    df = df.loc[df[id_col] == identifier]

    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    mstl_params = {"season_length": periods}
    if trend_forecaster is not None:
        mstl_params["trend_forecaster"] = trend_forecaster

    models = MSTL(**mstl_params)
    sf = StatsForecast(models=[models], freq=freq, n_jobs=njobs)
    sf.fit(df=df, time_col=time_col, target_col=target_col, id_col=id_col)

    result = sf.fitted_[0, 0].model_

    r_squared, p_value = trend_significance(result.trend)
    trend_results = f"R²={r_squared:.4f}, p={p_value:.4f}"
    resid = result.remainder
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

    subplot_titles = []
    for col in result.columns:
        subplot_titles.extend([f"{col.capitalize()}"])
    subplot_titles.extend(["Summary"])

    fig = sp.make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
    )

    for i, col in enumerate(result.columns):
        color = colors[(i - 1) % num_colors]
        fig.add_trace(
            go.Scatter(
                x=df.index,
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
        row=subplot_titles.index("Summary") + 1,
        col=1,
    )

    fig.update_xaxes(visible=False, row=subplot_titles.index("Summary") + 1, col=1)
    fig.update_yaxes(visible=False, row=subplot_titles.index("Summary") + 1, col=1)

    color = colors[(i - 1) % num_colors]

    fig.update_layout(
        title="Seasonal Decomposition",
        height=height,
        width=width,
        showlegend=False,
        hovermode="x",
    )

    return fig.show(fig_type)
