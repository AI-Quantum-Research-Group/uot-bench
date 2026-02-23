import dash
from dash import Input, Output

from utils.bfm import filtered_bfm_dataframe, empty_figure
from .bfm_heatmap_utils import build_heatmap_figure

MONGE_CORE_GROUPS = [
    [
        ("ma_residual_L1", "Monge-Ampère Residual (L1)", None),
        ("phi_is_convex", "Convexity Fraction of φ", None),
    ]
]


@dash.callback(
    Output("bfm-monge-core-heatmaps", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_monge_core_heatmaps(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if df.empty:
        return empty_figure("Stepsize vs Pushforward (Monge Core Diagnostics)")

    fig = build_heatmap_figure(
        df,
        MONGE_CORE_GROUPS,
        "Stepsize vs Pushforward (Monge Core Diagnostics)",
    )
    if fig is None:
        return empty_figure("Stepsize vs Pushforward (Monge Core Diagnostics)")
    return fig
