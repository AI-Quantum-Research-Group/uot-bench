import dash
from dash import Input, Output

from utils.bfm import filtered_bfm_dataframe, empty_figure
from .bfm_heatmap_utils import build_heatmap_figure

HEATMAP_METRICS = [
    ("runtime", "Runtime (s)", None),
    ("marginal_error_L2", "Marginal L2 Error", None),
    ("tv_mu_to_nu", "TV Distance Error", None),
]


@dash.callback(
    Output("bfm-stepsize-heatmaps", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_stepsize_heatmaps(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if df.empty:
        return empty_figure("Stepsize vs Pushforward Heatmaps")

    fig = build_heatmap_figure(df, HEATMAP_METRICS, "Stepsize vs Pushforward Heatmaps (median ± IQR)")
    if fig is None:
        return empty_figure("Stepsize vs Pushforward Heatmaps")
    return fig
