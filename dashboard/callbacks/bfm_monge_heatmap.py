import dash
from dash import Input, Output

from utils.bfm import filtered_bfm_dataframe, empty_figure
from .bfm_heatmap_utils import build_heatmap_figure

MONGE_HEATMAP_GROUPS = [
    [
        ("ma_residual_L1", "Monge-Ampère Residual (L1)", None),
        ("ma_residual_Linf", "Monge-Ampère Residual (L∞)", None),
    ],
    [
        ("detJ_neg_frac", "Determinant Negative Fraction", None),
        ("phi_is_convex", "Convexity Fraction of φ", None),
    ],
]


@dash.callback(
    Output("bfm-monge-heatmaps", "figure"),
    Input("bfm-dim-filter", "value"),
    Input("bfm-dist-filter", "value"),
    Input("bfm-stepsize-filter", "value"),
    Input("bfm-size-filter", "value"),
    Input("bfm-pushforward-filter", "value"),
)
def update_bfm_monge_heatmaps(dims, dists, stepsizes, sizes, pushforwards):
    dims, dists, stepsizes, sizes, pushforwards, df = filtered_bfm_dataframe(
        dims, dists, stepsizes, sizes, pushforwards
    )
    if df.empty:
        return empty_figure("Stepsize vs Pushforward (Monge Diagnostics)")

    fig = build_heatmap_figure(
        df,
        MONGE_HEATMAP_GROUPS,
        "Stepsize vs Pushforward (Monge Diagnostics, median ± IQR)",
    )
    if fig is None:
        return empty_figure("Stepsize vs Pushforward (Monge Diagnostics)")
    return fig
