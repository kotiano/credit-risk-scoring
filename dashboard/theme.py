"""Chart theme: validated reference palette (categorical order is the CVD-safety
mechanism - never cycled), recessive chrome, and shared Plotly layout."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"

# Categorical slots, fixed order (validated: adjacent CVD dE >= 9.1).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Sequential blue ramp (ordinal use starts at step 250 for >= 2:1 vs surface).
BLUE = {
    100: "#cde2fb",
    250: "#86b6ef",
    350: "#5598e7",
    450: "#2a78d6",
    550: "#1c5cab",
    650: "#104281",
    700: "#0d366b",
}
ORDINAL_5 = [BLUE[250], BLUE[350], BLUE[450], BLUE[550], BLUE[700]]
# Status palette - reserved for state, always paired with an icon/label.
STATUS = {"PASS": "#0ca30c", "WARN": "#fab219", "SERIOUS": "#ec835a", "FAIL": "#d03b3b"}
STATUS_ICON = {"PASS": "✓", "WARN": "▲", "FAIL": "✕"}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def layout(title: str | None = None, height: int = 320, **kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "height": height,
        "margin": {"l": 8, "r": 16, "t": 76 if title else 36, "b": 8},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT, "color": INK_2, "size": 12},
        "title": {
            "text": title,
            "x": 0.012,
            "xanchor": "left",
            "xref": "container",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 12},
            "font": {"size": 14, "color": INK, "weight": 600},
        }
        if title
        else None,
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#ffffff",
            "bordercolor": GRID,
            "font": {"family": FONT, "color": INK},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": INK_2},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "xaxis": {
            "automargin": True,
            "gridcolor": GRID,
            "linecolor": AXIS,
            "zeroline": False,
            "tickfont": {"color": MUTED},
            "showgrid": False,
        },
        "yaxis": {
            "automargin": True,
            "gridcolor": GRID,
            "linecolor": AXIS,
            "zeroline": False,
            "tickfont": {"color": MUTED},
        },
        "bargap": 0.25,
    }
    base.update(kwargs)
    return base


def figure(title: str | None = None, height: int = 320, **kwargs: Any) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**layout(title, height, **kwargs))
    return fig


def line(
    fig: go.Figure, x: Any, y: Any, name: str, slot: int, fmt: str = ",.2f", dash: str | None = None
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line={"color": SERIES[slot], "width": 2, "dash": dash},
            hovertemplate=f"%{{y:{fmt}}}<extra>{name}</extra>",
        )
    )


def bars(
    fig: go.Figure,
    x: Any,
    y: Any,
    name: str,
    color: str,
    fmt: str = ",.0f",
    orientation: str = "v",
    text: Any = None,
) -> None:
    fig.add_trace(
        go.Bar(
            x=x if orientation == "v" else y,
            y=y if orientation == "v" else x,
            name=name,
            orientation=orientation,
            marker={"color": color, "line": {"color": SURFACE, "width": 2}, "cornerradius": 4},
            text=text,
            textposition="outside",
            textfont={"color": INK_2},
            hovertemplate=(f"%{{y:{fmt}}}" if orientation == "v" else f"%{{x:{fmt}}}")
            + f"<extra>{name}</extra>",
        )
    )
