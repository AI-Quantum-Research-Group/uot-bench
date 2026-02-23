import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_COLORSCALE = "Cividis"
TEXT_FONT_SIZE = 16
SUBTEXT_FONT_SIZE = 13
LIGHT_TEXT = "rgba(255, 255, 255, 0.95)"
DARK_TEXT = "rgba(20, 20, 20, 0.95)"
MISSING_TEXT = "rgba(40, 40, 40, 0.7)"


def format_heatmap_value(val: float) -> str:
    if pd.isna(val):
        return "—"
    formatted = f"{val:.4f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _wrap_label(label: str, threshold: int = 10) -> str:
    if not label or len(label) <= threshold:
        return label
    mid = len(label) // 2
    for idx in range(mid, len(label)):
        if label[idx] == " ":
            return label[:idx] + "<br>" + label[idx + 1 :]
    return label


def _text_color(value, min_val, max_val):
    if (
        value is None
        or pd.isna(value)
        or min_val is None
        or max_val is None
        or not np.isfinite(value)
        or not np.isfinite(min_val)
        or not np.isfinite(max_val)
    ):
        return MISSING_TEXT
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    return LIGHT_TEXT if normalized <= 0.55 else DARK_TEXT


def prepare_heatmap_data(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return None

    metric_series = pd.to_numeric(df[metric], errors="coerce")
    stepsize_series = pd.to_numeric(df["stepsize"], errors="coerce")
    pushforward_series = df["pushforward_fn"]

    df_metric = (
        df.assign(metric=metric_series, stepsize=stepsize_series, pushforward_fn=pushforward_series)
        .dropna(subset=["metric", "stepsize", "pushforward_fn"])
    )
    if df_metric.empty:
        return None

    stepsizes = sorted(df_metric["stepsize"].dropna().unique().tolist())
    pushforwards = sorted(df_metric["pushforward_fn"].dropna().unique().tolist())
    if not stepsizes or not pushforwards:
        return None

    summary = (
        df_metric.groupby(["pushforward_fn", "stepsize"])["metric"]
        .agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    valid = summary["median"].dropna()
    min_val = valid.min() if not valid.empty else None
    max_val = valid.max() if not valid.empty else None
    lookup = {
        (row["pushforward_fn"], row["stepsize"]): (
            row["median"],
            row["q1"],
            row["q3"],
        )
        for _, row in summary.iterrows()
    }

    z_values = []
    text_values = []
    custom_values = []
    for pushforward in pushforwards:
        z_row = []
        text_row = []
        custom_row = []
        for step in stepsizes:
            stats = lookup.get((pushforward, step))
            if stats is None or any(pd.isna(v) for v in stats):
                z_row.append(np.nan)
                text_row.append("")
                custom_row.append([np.nan, np.nan, np.nan])
            else:
                median, q1, q3 = stats
                z_row.append(median)
                color = _text_color(median, min_val, max_val)
                median_text = format_heatmap_value(median)
                range_text = f"[{format_heatmap_value(q1)}, {format_heatmap_value(q3)}]"
                text_row.append(
                    "<span style='color:{color};font-weight:600;font-size:{size}px'>{median}</span>"
                    "<br>"
                    "<span style='color:{color};font-size:{sub}px'>{range}</span>".format(
                        color=color,
                        size=TEXT_FONT_SIZE,
                        sub=SUBTEXT_FONT_SIZE,
                        median=median_text,
                        range=range_text,
                    )
                )
                custom_row.append([median, q1, q3])
        z_values.append(z_row)
        text_values.append(text_row)
        custom_values.append(custom_row)

    x_labels = [f"{s:g}" if isinstance(s, (int, float)) else str(s).upper() for s in stepsizes]
    y_labels = [pf.upper() if isinstance(pf, str) else pf for pf in pushforwards]
    return {
        "z": z_values,
        "text": text_values,
        "custom": custom_values,
        "x": x_labels,
        "y": y_labels,
    }


def _normalize_metric_groups(metric_groups):
    if not metric_groups:
        return []
    first = metric_groups[0]
    if isinstance(first, (list, tuple)) and len(first) == 3 and isinstance(first[0], str):
        return [list(metric_groups)]
    return metric_groups


def build_heatmap_figure(df: pd.DataFrame, metric_groups, title: str):
    metric_groups = _normalize_metric_groups(metric_groups)
    prepared = [
        [
            (metric, label, colorscale, prepare_heatmap_data(df, metric))
            for metric, label, colorscale in group
        ]
        for group in metric_groups
    ]
    if all(all(data is None for _, _, _, data in group) for group in prepared):
        return None

    n_rows = len(metric_groups)
    n_cols = max(len(group) for group in metric_groups)
    titles = []
    for group in metric_groups:
        titles.extend([label for _, label, _ in group])
        if len(group) < n_cols:
            titles.extend([""] * (n_cols - len(group)))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=titles,
        horizontal_spacing=0.18,
        # vertical_spacing=0.18
        vertical_spacing=0.18,
    )

    coloraxis_idx = 1
    for row_idx, group in enumerate(prepared, start=1):
        for col_idx, (metric, label, colorscale, data) in enumerate(group, start=1):
            if data is None:
                continue
            coloraxis_name = f"coloraxis{coloraxis_idx}"
            heatmap = go.Heatmap(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                text=data["text"],
                texttemplate="%{text}",
                textfont=dict(size=TEXT_FONT_SIZE),
                customdata=data["custom"],
                hovertemplate=(
                    "Pushforward: %{y}<br>"
                    "Stepsize: %{x}<br>"
                    "Median: %{customdata[0]:.4g}<br>"
                    "P25: %{customdata[1]:.4g}<br>"
                    "P75: %{customdata[2]:.4g}<extra></extra>"
                ),
                coloraxis=coloraxis_name,
                xgap=1,
                ygap=1,
            )
            fig.add_trace(heatmap, row=row_idx, col=col_idx)
            fig.update_xaxes(title_text="Stepsize", row=row_idx, col=col_idx)
            # fig.update_xaxes(title_text="Stepsize", tickangle=-35, row=row_idx, col=col_idx)
            axis_name = "xaxis" if coloraxis_idx == 1 else f"xaxis{coloraxis_idx}"
            axis_obj = getattr(fig.layout, axis_name, None)
            domain = getattr(axis_obj, "domain", (0, 1))
            colorbar_x = min(1.02, domain[1] + 0.02)
            cell_height = len(data["y"]) if data and data["y"] else 1
            len_factor = max(0.50, min(0.7, 0.75 / cell_height))
            # len_factor = max(0.38, min(0.7, 0.75 / cell_height))
            fig.update_layout(
                **{
                    coloraxis_name: dict(
                        colorscale=colorscale or DEFAULT_COLORSCALE,
                        colorbar=dict(
                            # title=_wrap_label(label),
                            len=len_factor,
                            y=1 - (row_idx - 0.5) / n_rows,
                            x=colorbar_x,
                        ),
                    )
                }
            )
            coloraxis_idx += 1

    fig.update_yaxes(title_text="Pushforward", row=1, col=1)

    cell_height = 180 # 120
    n_pushforwards = max(
        len(data["y"]) if data and data["y"] else 0
        for group in prepared
        for _, _, _, data in group
    )
    figure_height = max(420, cell_height * max(n_pushforwards, 1) * n_rows)
    base_width = 350
    figure_width = max(850, base_width * n_cols)

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=figure_height,
        width=figure_width,
        margin=dict(t=90, l=60, r=80, b=60),
    )
    return fig
