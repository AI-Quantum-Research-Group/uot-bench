import dash
from dash import Output, Input

from utils.bfm import (
    filtered_bfm_dataframe,
    stepsize_metric_figure,
    BFM_PERFORMANCE_COLUMNS,
)


@dash.callback(
    Output("bfm-performance-summary", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_performance(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if (
        len(dims) == 0
        or len(dists) == 0
        or len(stepsizes) == 0
        or len(sizes) == 0
        or len(pushforwards) == 0
    ):
        df = df.iloc[0:0]
    metrics = [m for m in BFM_PERFORMANCE_COLUMNS if m in df.columns]
    return stepsize_metric_figure(df, metrics, "Performance metrics vs stepsize")
