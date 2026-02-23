import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from color_transfer.stats import extract_stats_options, get_stats_dataframe

dash.register_page(__name__, path="/color_transfer_stats", name="Color Transfer Statistics")

df, _status = get_stats_dataframe()
options = extract_stats_options(df)


def _filter_dropdown(label: str, filter_id: str, values: list, placeholder: str):
    return dbc.Col(
        [
            html.Label(label),
            dcc.Dropdown(
                id=filter_id,
                options=[{"label": value, "value": value} for value in values],
                value=values,
                multi=True,
                placeholder=placeholder,
                persistence=True,
            ),
        ],
        lg=3,
        md=6,
        sm=12,
    )


def _metric_card(title: str, graph_id: str):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(title, className="mb-3"),
                dcc.Graph(id=graph_id),
            ]
        ),
        className="mb-4",
    )

layout = dbc.Container(
    [
        html.H2("Color Transfer Statistics", className="mb-3"),
        html.P(
            "Explore runtime, distribution distance, map structure, and image quality metrics "
            "for color transfer experiments.",
            className="text-muted",
        ),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        _filter_dropdown(
                            "Solvers",
                            "ct-stat-solver-filter",
                            options["solvers"],
                            "Select solvers...",
                        ),
                        _filter_dropdown(
                            "Displacement Alpha",
                            "ct-stat-alpha-filter",
                            options["alphas"],
                            "Select alphas...",
                        ),
                        _filter_dropdown(
                            "Bins per channel",
                            "ct-stat-bins-filter",
                            options["bins"],
                            "Select bins...",
                        ),
                        _filter_dropdown(
                            "Regularization",
                            "ct-stat-reg-filter",
                            options["regs"],
                            "Select reg...",
                        ),
                        _filter_dropdown(
                            "Color space",
                            "ct-stat-color-space-filter",
                            options["color_spaces"],
                            "Select color spaces...",
                        ),
                    ],
                    className="gy-3 gx-3",
                )
            ),
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="ct-stat-general"), width=12, className="mb-4"),
            ]
        ),
        _metric_card("Runtime Metrics", "ct-stat-runtime"),
        _metric_card("Distribution Distance Metrics", "ct-stat-distance"),
        _metric_card("Map Structure Metrics", "ct-stat-map"),
        _metric_card("Marginal Error Metrics", "ct-stat-marginal-error"),
        _metric_card("Image Quality Metrics", "ct-stat-image-quality"),
    ],
    fluid=True,
    className="p-4",
)
