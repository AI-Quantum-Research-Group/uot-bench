from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from color_transfer.options import build_showcase_options
from color_transfer.showcase_helpers import (
    build_option_buttons,
    build_original_row,
    build_results_row,
    build_skeleton,
    encode_numpy_image,
    ensure_pair_columns,
)
from color_transfer.state import (
    build_problem_list,
    filter_showcase_dataframe,
    format_problem_label,
    step_problem_index,
)
from color_transfer.config import (
    COLOR_TRANSFER_DATA_FOLDER,
    ORIGINAL_IMAGES_FOLDER,
)
from color_transfer_data_loader import (
    extract_image_params,
    load_color_transfer_data,
    load_original_image_loader,
)


dash.register_page(__name__, path="/color_transfer", name="Color Transfer Showcase")

@lru_cache(maxsize=1)
def _load_showcase_state():
    """Initialise and cache dataframe and image loaders for the showcase."""
    df, load_result_image = load_color_transfer_data(COLOR_TRANSFER_DATA_FOLDER)
    load_original_image = load_original_image_loader(ORIGINAL_IMAGES_FOLDER)

    df = ensure_pair_columns(df)
    if not df.empty and "solver" in df.columns:
        df["solver"] = df["solver"].astype(str)

    showcase_options = build_showcase_options(df)
    return df, load_result_image, load_original_image, showcase_options


df, load_result_image, load_original_image, showcase_options = _load_showcase_state()


def _build_option_group(title: str, options: list, group: str):
    return html.Div(
        [
            html.Div(title, className="text-muted small mb-2"),
            html.Div(build_option_buttons(options, group), className="ct-button-wrap"),
        ],
        className="mb-3",
    )


def _empty_problem_view(message: str):
    return 0, 0, message, None, None, None


layout = dbc.Container(
    [
        html.H2("Color Transfer Analysis", className="mb-3"),
        html.P(
            "Browse original and transported images for each color transfer pair. "
            "Use the buttons or slider to switch problem instances.",
            className="text-muted",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    _build_option_group(
                                        "Bins per channel",
                                        showcase_options.bins,
                                        "bins",
                                    ),
                                    _build_option_group(
                                        "Solvers",
                                        showcase_options.solvers,
                                        "solver",
                                    ),
                                    _build_option_group(
                                        "Color spaces",
                                        showcase_options.color_spaces,
                                        "color_space",
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button("Prev", id="ct-prev-btn", color="secondary"),
                                                        dbc.Button("Next", id="ct-next-btn", color="secondary"),
                                                    ]
                                                ),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                dcc.Slider(
                                                    id="ct-problem-slider",
                                                    min=0,
                                                    max=max(len(df) - 1, 0),
                                                    step=1,
                                                    value=0,
                                                    marks=None,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                ),
                                                className="flex-grow-1",
                                            ),
                                        ],
                                        className="g-3 align-items-center",
                                        justify="center",
                                    ),
                                    html.Div(id="ct-problem-label", className="text-center text-muted mt-2"),
                                    html.Div(id="ct-result-info", className="text-center text-muted mt-2"),
                                ]
                            ),
                            className="mb-3",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(id="ct-original-row", className="mb-4"),
                                html.Div(id="ct-result-row"),
                                build_skeleton(),
                            ],
                            id="ct-images-wrapper",
                        ),
                    ],
                    md=8,
                ),
            ],
            className="g-4",
        ),
        dcc.Store(id="ct-selected-bins", data=showcase_options.bins[0] if showcase_options.bins else None),
        dcc.Store(
            id="ct-selected-solvers",
            data=showcase_options.solvers[0] if showcase_options.solvers else None,
        ),
        dcc.Store(
            id="ct-selected-color-spaces",
            data=showcase_options.color_spaces[0] if showcase_options.color_spaces else None,
        ),
        dbc.Modal(
            [
                dbc.ModalBody(html.Img(id="ct-modal-image", style={"width": "100%"}), className="p-0"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="ct-image-modal-close", color="secondary", className="ms-auto")
                ),
            ],
            id="ct-image-modal",
            size="xl",
            centered=True,
            is_open=False,
        ),
    ],
    fluid=True,
    className="p-4",
)


@dash.callback(
    [
        dash.Output({"type": "ct-option-btn", "group": "bins", "value": dash.ALL}, "color"),
        dash.Output({"type": "ct-option-btn", "group": "bins", "value": dash.ALL}, "outline"),
        dash.Output({"type": "ct-option-btn", "group": "solver", "value": dash.ALL}, "color"),
        dash.Output({"type": "ct-option-btn", "group": "solver", "value": dash.ALL}, "outline"),
        dash.Output({"type": "ct-option-btn", "group": "color_space", "value": dash.ALL}, "color"),
        dash.Output({"type": "ct-option-btn", "group": "color_space", "value": dash.ALL}, "outline"),
    ],
    [
        dash.Input("ct-selected-bins", "data"),
        dash.Input("ct-selected-solvers", "data"),
        dash.Input("ct-selected-color-spaces", "data"),
    ],
    [
        dash.State({"type": "ct-option-btn", "group": "bins", "value": dash.ALL}, "id"),
        dash.State({"type": "ct-option-btn", "group": "solver", "value": dash.ALL}, "id"),
        dash.State({"type": "ct-option-btn", "group": "color_space", "value": dash.ALL}, "id"),
    ],
)
def _sync_option_button_styles(
    selected_bins,
    selected_solvers,
    selected_color_spaces,
    bins_ids,
    solver_ids,
    color_space_ids,
):
    def _styles(selected_value, ids):
        selected = str(selected_value) if selected_value is not None else None
        colors = []
        outlines = []
        for entry in ids or []:
            value = entry.get("value") if isinstance(entry, dict) else None
            is_active = value is not None and value == selected
            colors.append("primary" if is_active else "secondary")
            outlines.append(not is_active)
        return colors, outlines

    bins_color, bins_outline = _styles(selected_bins, bins_ids)
    solver_color, solver_outline = _styles(selected_solvers, solver_ids)
    color_space_color, color_space_outline = _styles(selected_color_spaces, color_space_ids)
    return (
        bins_color,
        bins_outline,
        solver_color,
        solver_outline,
        color_space_color,
        color_space_outline,
    )


@dash.callback(
    [
        dash.Output("ct-selected-bins", "data"),
        dash.Output("ct-selected-solvers", "data"),
        dash.Output("ct-selected-color-spaces", "data"),
    ],
    [dash.Input({"type": "ct-option-btn", "group": dash.ALL, "value": dash.ALL}, "n_clicks")],
    [
        dash.State("ct-selected-bins", "data"),
        dash.State("ct-selected-solvers", "data"),
        dash.State("ct-selected-color-spaces", "data"),
    ],
)
def _toggle_option_buttons(
    _n_clicks,
    selected_bins,
    selected_solvers,
    selected_color_spaces,
):
    triggered = dash.ctx.triggered_id
    if not isinstance(triggered, dict):
        return selected_bins, selected_solvers, selected_color_spaces

    group = triggered.get("group")
    value = triggered.get("value")
    if group is None or value is None:
        return selected_bins, selected_solvers, selected_color_spaces

    if group == "bins":
        return value, selected_solvers, selected_color_spaces
    if group == "solver":
        return selected_bins, value, selected_color_spaces
    if group == "color_space":
        return selected_bins, selected_solvers, value
    return selected_bins, selected_solvers, selected_color_spaces


@dash.callback(
    [
        dash.Output("ct-problem-slider", "value"),
        dash.Output("ct-problem-slider", "max"),
        dash.Output("ct-problem-label", "children"),
        dash.Output("ct-original-row", "children"),
        dash.Output("ct-result-row", "children"),
        dash.Output("ct-result-info", "children"),
    ],
    [
        dash.Input("ct-prev-btn", "n_clicks"),
        dash.Input("ct-next-btn", "n_clicks"),
        dash.Input("ct-problem-slider", "value"),
        dash.Input("ct-selected-bins", "data"),
        dash.Input("ct-selected-color-spaces", "data"),
        dash.Input("ct-selected-solvers", "data"),
    ],
)
def _update_problem_view(
    prev_clicks,
    next_clicks,
    slider_value,
    selected_bins,
    selected_color_spaces,
    selected_solvers,
):
    _ = (prev_clicks, next_clicks)
    if df.empty:
        return _empty_problem_view("No color transfer data found.")

    filtered = filter_showcase_dataframe(df, selected_bins, selected_color_spaces, selected_solvers)
    if filtered.empty:
        return _empty_problem_view("No results for current filters.")

    problem_list = build_problem_list(filtered)
    if not problem_list:
        return _empty_problem_view("No results for current filters.")

    max_index = max(len(problem_list) - 1, 0)
    index = step_problem_index(slider_value, dash.ctx.triggered_id, max_index)
    source_name, target_name = problem_list[index]

    label = format_problem_label(index, len(problem_list), source_name, target_name)
    subset = filtered[
        (filtered["source_image_name"] == source_name)
        & (filtered["target_image_name"] == target_name)
    ]

    original_row = build_original_row(load_original_image, source_name, target_name, thumb_width=360)
    result_row = build_results_row(
        subset,
        load_result_image,
        source_name,
        target_name,
        selected_solvers,
        extract_image_params,
        thumb_width=300,
    )

    result_info = f"solver={selected_solvers}" if selected_solvers else None
    return index, max_index, label, original_row, result_row, result_info


@dash.callback(
    [
        dash.Output("ct-image-modal", "is_open"),
        dash.Output("ct-modal-image", "src"),
    ],
    [
        dash.Input(
            {
                "type": "ct-image",
                "kind": dash.ALL,
                "name": dash.ALL,
                "solver": dash.ALL,
                "run_folder": dash.ALL,
                "filename": dash.ALL,
                "source": dash.ALL,
                "target": dash.ALL,
            },
            "n_clicks",
        ),
        dash.Input("ct-image-modal-close", "n_clicks"),
    ],
    [
        dash.State(
            {
                "type": "ct-image",
                "kind": dash.ALL,
                "name": dash.ALL,
                "solver": dash.ALL,
                "run_folder": dash.ALL,
                "filename": dash.ALL,
                "source": dash.ALL,
                "target": dash.ALL,
            },
            "src",
        ),
    ],
)
def _toggle_image_modal(n_clicks_list, close_clicks, src_list):
    _ = (n_clicks_list, close_clicks, src_list)
    triggered = dash.ctx.triggered_id

    if triggered == "ct-image-modal-close":
        return False, None

    if not isinstance(triggered, dict) or triggered.get("type") != "ct-image":
        return False, None

    kind = triggered.get("kind")
    if kind == "original":
        image_name = triggered.get("name")
        full_image = load_original_image(image_name)
        encoded = encode_numpy_image(full_image, max_width=None)
        if not encoded:
            return False, None
        return True, encoded

    if kind == "result":
        source = triggered.get("source")
        target = triggered.get("target")
        solver = triggered.get("solver")
        params = {
            "run_folder": triggered.get("run_folder"),
            "result_image_filename": triggered.get("filename"),
        }
        full_image = load_result_image(f"{source} -> {target}", solver, params)
        encoded = encode_numpy_image(full_image, max_width=None)
        if not encoded:
            return False, None
        return True, encoded

    return False, None
