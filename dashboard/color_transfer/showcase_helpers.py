import base64
import io
from functools import lru_cache
from typing import Callable

import pandas as pd
from dash import html
import dash_bootstrap_components as dbc
from PIL import Image


def parse_pair(name: str) -> tuple[str | None, str | None]:
    if not isinstance(name, str):
        return None, None
    parts = [p.strip() for p in name.split("->")]
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def ensure_pair_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "source_image_name" in frame.columns and "target_image_name" in frame.columns:
        return frame
    source = []
    target = []
    name_col = "dataset" if "dataset" in frame.columns else ("name" if "name" in frame.columns else None)
    if name_col is None:
        frame["source_image_name"] = None
        frame["target_image_name"] = None
        return frame
    for value in frame[name_col]:
        src, tgt = parse_pair(value)
        source.append(src)
        target.append(tgt)
    frame["source_image_name"] = source
    frame["target_image_name"] = target
    return frame


@lru_cache(maxsize=256)
def _encode_image_bytes(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def encode_image_array(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _encode_image_bytes(buffer.getvalue())


def encode_numpy_image(image_arr, max_width: int | None = None) -> str | None:
    if image_arr is None:
        return None
    image = Image.fromarray(image_arr)
    if max_width is not None and image.width > max_width:
        ratio = max_width / float(image.width)
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return encode_image_array(image)


def build_option_buttons(options, group):
    return [
        dbc.Button(
            str(option),
            id={"type": "ct-option-btn", "group": group, "value": str(option)},
            color="secondary",
            outline=True,
            size="sm",
            className="me-2 mb-2",
        )
        for option in options
    ]


def build_original_row(
    load_original_image: Callable[[str], object],
    source_name: str,
    target_name: str,
    *,
    thumb_width: int = 360,
):
    originals = []
    for label, name in (("Source", source_name), ("Target", target_name)):
        img = load_original_image(name)
        encoded = encode_numpy_image(img, max_width=thumb_width)
        content = (
            html.Img(
                id={
                    "type": "ct-image",
                    "kind": "original",
                    "name": name,
                    "solver": "",
                    "run_folder": "",
                    "filename": "",
                    "source": source_name,
                    "target": target_name,
                },
                n_clicks=0,
                src=encoded,
                style={"maxWidth": "240px", "width": "100%", "cursor": "zoom-in"},
            )
            if encoded
            else html.Div("Image unavailable", className="ct-image-missing")
        )
        originals.append(
            dbc.Col(
                html.Div(
                    [
                        html.Div(label, className="text-muted small text-center"),
                        content,
                    ],
                    className="text-center",
                ),
                width="auto",
            )
        )
    return dbc.Row(
        originals,
        justify="center",
        className="g-4 ct-image-row",
        style={"flexWrap": "nowrap", "overflowX": "auto"},
    )


def build_results_row(
    subset: pd.DataFrame,
    load_result_image: Callable[[str, str, dict], object],
    source_name: str,
    target_name: str,
    solver_name: str | None,
    extract_image_params: Callable[[object], dict],
    *,
    thumb_width: int = 300,
):
    if subset.empty or solver_name is None:
        return dbc.Row([], justify="center")
    if "displacement_alpha" in subset.columns:
        subset = subset.sort_values("displacement_alpha")
        subset = subset.drop_duplicates(subset=["displacement_alpha"], keep="first")
    tiles = []
    for _, row in subset.iterrows():
        params = extract_image_params(row)
        img = load_result_image(f"{source_name} -> {target_name}", solver_name, params)
        encoded = encode_numpy_image(img, max_width=thumb_width)
        label = None
        if "displacement_alpha" in row:
            label = f"alpha={row['displacement_alpha']}"
        content = (
            html.Img(
                id={
                    "type": "ct-image",
                    "kind": "result",
                    "name": "",
                    "solver": solver_name,
                    "run_folder": params.get("run_folder"),
                    "filename": params.get("result_image_filename"),
                    "source": source_name,
                    "target": target_name,
                },
                n_clicks=0,
                src=encoded,
                style={"maxWidth": "200px", "width": "100%", "cursor": "zoom-in"},
            )
            if encoded
            else html.Div("Image unavailable", className="ct-image-missing")
        )
        tiles.append(
            dbc.Col(
                html.Div(
                    [
                        html.Div(label, className="text-muted small text-center") if label else None,
                        content,
                    ],
                    className="text-center",
                ),
                width="auto",
            )
        )
    return dbc.Row(
        tiles,
        justify="center",
        className="g-3 ct-image-row",
        style={"flexWrap": "nowrap", "overflowX": "auto"},
    )


def build_skeleton():
    def _skeleton_row(count, width=220, height=140):
        return dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        className="ct-skeleton-box",
                        style={"width": f"{width}px", "height": f"{height}px"},
                    ),
                    width="auto",
                )
                for _ in range(count)
            ],
            justify="center",
            className="g-3",
        )

    return html.Div(
        [
            html.Div(className="ct-skeleton-label"),
            _skeleton_row(2, width=240, height=150),
            html.Div(className="ct-skeleton-label mt-4"),
            _skeleton_row(5, width=200, height=130),
        ],
        id="ct-image-skeleton",
    )
