"""Plotly figures styled for the ForgeSight dark theme.

Series colours were validated for the dark surface (lightness band, CVD separation, contrast):
cyan = people in frame, amber = people in the restricted zone. Red is reserved for
safety-event status and is always paired with a diamond marker and a text label.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import events as evt
from .annotate import fmt_ts

SERIES_PEOPLE = "#0B9DB8"
SERIES_ZONE = "#D07A0A"
STATUS_CRITICAL = "#E5484D"
STATUS_WARNING = "#F5A524"
TEXT = "#E6EAF0"
TEXT_2 = "#9AA4B2"
TEXT_3 = "#64748B"
GRID = "rgba(148,163,184,0.10)"
SURFACE = "#111827"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def _base_layout(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=TEXT_2),
        hoverlabel=dict(bgcolor="#0B1220", bordercolor="#334155", font=dict(family=FONT, color=TEXT, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=TEXT_2, size=12), bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(148,163,184,0.25)",
                     tickfont=dict(color=TEXT_3, size=11), ticks="", title_font=dict(color=TEXT_3, size=11))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=TEXT_3, size=11),
                     title_font=dict(color=TEXT_3, size=11), rangemode="tozero")
    return fig


def _mmss_ticks(fig: go.Figure, max_s: float) -> None:
    step = next((s for s in (5, 10, 15, 30, 60, 120, 300, 600) if max_s / s <= 8), 900)
    vals = list(np.arange(0, max_s + step, step))
    fig.update_xaxes(tickvals=vals, ticktext=[f"{int(v // 60):02d}:{int(v % 60):02d}" for v in vals],
                     range=[0, max(max_s, 1)])


def occupancy_chart(frames: pd.DataFrame, events: pd.DataFrame, zone_enabled: bool, height: int = 320,
                    trend_window_s: float = 0.0, alerts: pd.DataFrame | None = None,
                    threshold: int | None = None, threshold_scope: str = "zone") -> go.Figure:
    """Raw per-frame counts (thin steps) with an optional rolling-mean trend, alert threshold and
    shaded sustained-occupancy periods."""
    from .analytics import trend

    fig = go.Figure()
    if frames.empty:
        return _base_layout(fig, height)
    t = frames["time_s"]
    hover_t = t.map(fmt_ts)
    smooth = trend_window_s > 0
    series = [("frame", "people", "People in frame", SERIES_PEOPLE, "11,157,184")]
    if zone_enabled:
        series.append(("zone", "people_in_zone", "People in restricted zone", SERIES_ZONE, "208,122,10"))
    for i, (scope, col, name, color, rgb) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=t, y=frames[col], name=f"{name} (per frame)" if smooth else name, mode="lines",
            line=dict(color=f"rgba({rgb},0.45)" if smooth else color, width=1 if smooth else 2, shape="hv"),
            fill="tozeroy" if i == 0 else None, fillcolor=f"rgba({rgb},0.08)", customdata=hover_t,
            showlegend=not smooth,
            hovertemplate=("%{customdata}<br>" if i == 0 else "") + f"{name}: <b>%{{y}}</b><extra></extra>",
        ))
        if smooth:
            fig.add_trace(go.Scatter(
                x=t, y=trend(frames, scope, trend_window_s), name=f"{name} ({trend_window_s:g} s avg)",
                mode="lines", line=dict(color=color, width=2, shape="spline", smoothing=0.6),
                hovertemplate=f"{trend_window_s:g} s avg: %{{y:.1f}}<extra></extra>",
            ))
    if alerts is not None and len(alerts):
        for a in alerts.itertuples():
            fig.add_vrect(x0=a.start_s, x1=a.end_s, fillcolor="rgba(245,165,36,0.10)", line_width=0, layer="below",
                          annotation_text=a.alert_id, annotation_position="top left",
                          annotation_font=dict(color=STATUS_WARNING, size=10))
    if threshold:
        fig.add_hline(y=threshold, line=dict(color=STATUS_WARNING, width=1, dash="dash"),
                      annotation_text=f"Alert threshold ≥ {threshold} ({'zone' if threshold_scope == 'zone' else 'view'})",
                      annotation_position="top right", annotation_font=dict(color=STATUS_WARNING, size=10))
    groups = [
        (evt.intrusions(events), "Zone intrusion", "diamond", STATUS_CRITICAL, "Restricted zone intrusion"),
        (evt.at_start(events), "Present at start", "circle-open", STATUS_WARNING, "Already in zone at start"),
    ] if len(events) and "event_type" in events else []
    for ev, name, symbol, color, title in groups:
        if ev.empty:
            continue
        ev_y = np.interp(ev["entry_time_s"], t, frames["people"])
        labels = [f"{r.event_id} · {'ID ' + str(r.track_id) if pd.notna(r.track_id) else 'untracked'} · "
                  f"{fmt_ts(r.entry_time_s)} · {r.entry_context}" for r in ev.itertuples()]
        fig.add_trace(go.Scatter(
            x=ev["entry_time_s"], y=ev_y, name=name, mode="markers",
            marker=dict(symbol=symbol, size=11, color=color, line=dict(color=color if "open" in symbol else SURFACE,
                                                                      width=2)),
            text=labels, hovertemplate=f"<b>{title}</b><br>%{{text}}<extra></extra>",
        ))
    peak_i = int(frames["people"].idxmax())
    fig.add_annotation(x=t.iloc[peak_i], y=frames["people"].iloc[peak_i], text=f"Peak {frames['people'].iloc[peak_i]}",
                       showarrow=True, arrowhead=0, arrowcolor=TEXT_3, ax=0, ay=-26,
                       font=dict(color=TEXT, size=11), bgcolor="#0B1220", bordercolor="#334155", borderpad=3)
    _base_layout(fig, height, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="People", dtick=1 if frames["people"].max() <= 8 else None)
    _mmss_ticks(fig, float(t.max()))
    return fig


def hazard_chart(checks: pd.DataFrame, hazards: pd.DataFrame | None, threshold: float, classes: list[str],
                 height: int = 260) -> go.Figure:
    """Fire (and optionally smoke) score per check, threshold line and shaded confirmed episodes."""
    fig = go.Figure()
    if checks is None or checks.empty:
        return _base_layout(fig, height)
    colors = {"fire": STATUS_CRITICAL, "smoke": TEXT_2}
    for c in classes:
        fig.add_trace(go.Scatter(
            x=checks["time_s"], y=checks[c], name=f"{c.capitalize()} score", mode="lines+markers",
            line=dict(color=colors[c], width=2), marker=dict(size=5), customdata=checks["time_s"].map(fmt_ts),
            hovertemplate=f"%{{customdata}}<br>{c.capitalize()} score: <b>%{{y:.2f}}</b><extra></extra>"))
    if hazards is not None and len(hazards):
        for h in hazards.itertuples():
            end = h.end_time_s if pd.notna(h.end_time_s) else h.start_time_s
            fig.add_vrect(x0=h.start_time_s, x1=max(end, h.start_time_s + 0.3), fillcolor="rgba(229,72,77,0.12)",
                          line_width=0, layer="below", annotation_text=h.event_id, annotation_position="top left",
                          annotation_font=dict(color=STATUS_CRITICAL, size=10))
    fig.add_hline(y=threshold, line=dict(color=STATUS_WARNING, width=1, dash="dash"),
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="bottom right",
                  annotation_font=dict(color=STATUS_WARNING, size=10))
    _base_layout(fig, height, legend=len(classes) > 1)
    fig.update_yaxes(range=[0, 1], title_text="Class probability")
    _mmss_ticks(fig, float(checks["time_s"].max()))
    return fig


def activity_chart(intervals: pd.DataFrame, bin_s: int, height: int = 280) -> go.Figure:
    fig = go.Figure()
    if intervals.empty:
        return _base_layout(fig, height)
    fig.add_trace(go.Bar(
        x=intervals["label"], y=intervals["detections"], marker=dict(color=SERIES_PEOPLE, cornerradius=4),
        customdata=np.stack([intervals["avg_occupancy"], intervals["peak_occupancy"]], axis=1),
        hovertemplate=(f"<b>%{{x}}</b> (+{bin_s}s)<br>Person detections: <b>%{{y}}</b>"
                       "<br>Avg occupancy: %{customdata[0]:.1f}<br>Peak occupancy: %{customdata[1]}<extra></extra>"),
    ))
    _base_layout(fig, height)
    fig.update_yaxes(title_text="Person detections")
    fig.update_xaxes(title_text=f"Interval start (mm:ss, {bin_s}s bins)", type="category")
    return fig


def events_chart(intervals: pd.DataFrame, bin_s: int, height: int = 280) -> go.Figure:
    fig = go.Figure()
    if intervals.empty:
        return _base_layout(fig, height)
    fig.add_trace(go.Bar(
        x=intervals["label"], y=intervals["zone_entries"], marker=dict(color=STATUS_CRITICAL, cornerradius=4),
        hovertemplate=f"<b>%{{x}}</b> (+{bin_s}s)<br>Restricted-zone entries: <b>%{{y}}</b><extra></extra>",
    ))
    _base_layout(fig, height)
    peak = int(intervals["zone_entries"].max()) if len(intervals) else 0
    fig.update_yaxes(title_text="Zone entries", dtick=1 if peak <= 6 else None, range=[0, max(peak, 1) * 1.15])
    fig.update_xaxes(title_text=f"Interval start (mm:ss, {bin_s}s bins)", type="category")
    return fig


def _jpeg_data_uri(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def normalized_heatmap(heat: np.ndarray) -> np.ndarray:
    if heat.max() <= 0:
        return heat
    blurred = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=1.6)
    # Square-root scaling keeps walking paths visible next to spots where people stood still.
    return np.sqrt(blurred / blurred.max())


def heatmap_figure(heat: np.ndarray, background_bgr: np.ndarray | None, zone_points: list | None,
                   frame_size: tuple[int, int], height: int = 380, measure: str = "presence") -> go.Figure:
    """Foot-position density over a median background frame (people removed by the median)."""
    w, h = frame_size
    if background_bgr is not None:
        background_bgr = cv2.resize(background_bgr, (min(960, w), int(min(960, w) * h / w)))
        dim = cv2.addWeighted(background_bgr, 0.55, np.zeros_like(background_bgr), 0.45, 0)
        w, h = dim.shape[1], dim.shape[0]
    norm = normalized_heatmap(heat)
    gh, gw = norm.shape
    fig = go.Figure()
    if background_bgr is not None:
        fig.add_trace(go.Image(source=_jpeg_data_uri(dim), x0=0, y0=0, dx=1, dy=1, hoverinfo="skip"))
    z = np.where(norm < 0.12, np.nan, norm)
    fig.add_trace(go.Heatmap(
        z=z, x0=w / gw / 2, dx=w / gw, y0=h / gh / 2, dy=h / gh, zmin=0, zmax=1, zsmooth="best",
        colorscale=[[0.0, "rgba(14,116,144,0.10)"], [0.3, "rgba(8,145,178,0.45)"],
                    [0.6, "rgba(34,211,238,0.70)"], [0.85, "rgba(165,243,252,0.85)"],
                    [1.0, "rgba(240,253,255,0.95)"]],
        colorbar=dict(title=dict(text=f"Relative<br>{measure}", font=dict(color=TEXT_3, size=11)),
                      tickvals=[0.05, 0.5, 1], ticktext=["Low", "Med", "High"], tickfont=dict(color=TEXT_3, size=11),
                      thickness=10, len=0.75, outlinewidth=0),
        hovertemplate=f"Relative {measure}: %{{z:.0%}}<extra></extra>",
    ))
    if zone_points:
        xs = [p[0] / 100 * w for p in zone_points] + [zone_points[0][0] / 100 * w]
        ys = [p[1] / 100 * h for p in zone_points] + [zone_points[0][1] / 100 * h]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=SERIES_ZONE, width=2, dash="dash"),
                                 name="Restricted zone", hoverinfo="name"))
    _base_layout(fig, height)
    fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", constrain="domain", gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig
