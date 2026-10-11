"""Restricted-zone editor: presets, quick rectangle, editable polygon points and a live preview."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from forgesight.annotate import CYAN, WHITE, draw_zone
from forgesight.config import ZONE_PRESETS
from forgesight.state import bound, get_settings, update_settings
from forgesight.zones import polygon_to_pixels, validate_polygon


def placeholder_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    img = np.full((h, w, 3), (26, 18, 11), dtype=np.uint8)
    cv2.putText(img, "Upload a video to preview the zone on a real frame", (w // 2 - 330, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 116, 100), 1, cv2.LINE_AA)
    return img


def zone_preview(frame: np.ndarray, points: list, name: str, enabled: bool) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(1280 * h / w)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    # 10% reference grid so coordinates are easy to read off the image
    grid = img.copy()
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(grid, (x, 0), (x, h), (200, 200, 200), 1)
        cv2.line(grid, (0, y), (w, y), (200, 200, 200), 1)
    cv2.addWeighted(grid, 0.22, img, 0.78, 0, dst=img)
    for i in range(1, 10):
        cv2.putText(img, f"{i * 10}", (int(w * i / 10) + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{i * 10}", (3, int(h * i / 10) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
    ok, _ = validate_polygon(points)
    if enabled and ok:
        poly = polygon_to_pixels(points, w, h)
        draw_zone(img, poly, name, occupied=False)
        for i, (x, y) in enumerate(poly, start=1):
            c = (int(x), int(y))
            cv2.circle(img, c, 11, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, c, 11, CYAN, 2, cv2.LINE_AA)
            cv2.putText(img, str(i), (c[0] - 5, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _apply(points: list) -> None:
    update_settings(zone_points=[[round(float(x), 1), round(float(y), 1)] for x, y in points])
    st.session_state.zone_ver = st.session_state.get("zone_ver", 0) + 1


def render_zone_editor(frame: np.ndarray | None, key: str, layout: str = "stacked") -> None:
    s = get_settings()
    ver = st.session_state.get("zone_ver", 0)
    frame = frame if frame is not None else placeholder_frame()

    if layout == "side":
        preview_col, ctrl_col = st.columns([3, 2], gap="large")
    else:
        preview_col = ctrl_col = st.container()
    preview_slot = preview_col.empty()

    with ctrl_col:
        c1, c2 = st.columns([1, 2])
        with c1:
            bound(st.toggle, "Zone enabled", "zone_enabled", key_suffix=key)
        with c2:
            bound(st.text_input, "Zone name", "zone_name", key_suffix=key, max_chars=40, label_visibility="collapsed",
                  placeholder="Zone name")

        tab_quick, tab_points = st.tabs(["Presets & rectangle", "Polygon points"])
        with tab_quick:
            p1, p2 = st.columns([3, 1], vertical_alignment="bottom")
            preset = p1.selectbox("Preset", list(ZONE_PRESETS), key=f"zone_preset_{key}")
            if p2.button("Apply", key=f"zone_apply_preset_{key}", width="stretch"):
                _apply(ZONE_PRESETS[preset])
                st.rerun()
            xs = [p[0] for p in s.zone_points] or [0, 100]
            ys = [p[1] for p in s.zone_points] or [0, 100]
            xr = st.slider("Horizontal extent (% of width)", 0.0, 100.0, (float(min(xs)), float(max(xs))), 0.5, format="%.0f%%",
                           key=f"zone_xr_{key}_{ver}")
            yr = st.slider("Vertical extent (% of height)", 0.0, 100.0, (float(min(ys)), float(max(ys))), 0.5, format="%.0f%%",
                           key=f"zone_yr_{key}_{ver}")
            if st.button("Use rectangle", key=f"zone_apply_rect_{key}", width="stretch",
                         help="Replace the polygon with this rectangle"):
                _apply([[xr[0], yr[0]], [xr[1], yr[0]], [xr[1], yr[1]], [xr[0], yr[1]]])
                st.rerun()

        with tab_points:
            st.caption("Points in % of frame (0–100), in order around the shape. Add or delete rows to change "
                       "the polygon. The grid on the preview is in 10% steps.")
            editor_key = f"zone_points_{key}_{ver}"
            if editor_key not in st.session_state:
                st.session_state[f"zone_base_{key}"] = [p[:] for p in s.zone_points]
            base = pd.DataFrame(st.session_state[f"zone_base_{key}"], columns=["x (%)", "y (%)"])
            edited = st.data_editor(
                base, key=editor_key, num_rows="dynamic", width="stretch",
                column_config={
                    "x (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                    "y (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
                },
            )
            pts = edited.dropna().values.tolist()
            ok_pts, _ = validate_polygon(pts)
            if ok_pts and pts != s.zone_points:
                update_settings(zone_points=[[round(float(x), 1), round(float(y), 1)] for x, y in pts])

        ok, msg = validate_polygon(s.zone_points)
        if not s.zone_enabled:
            st.caption("Zone monitoring is off. Detection and analytics still run.")
        elif ok:
            st.caption(f"✓ Valid zone · {len(s.zone_points)} points · {msg}")
        else:
            st.error(msg)

    preview_slot.image(zone_preview(frame, s.zone_points, s.zone_name, s.zone_enabled),
                       width="stretch", caption="Zone preview · the foot point (bottom-centre of each "
                       "person box) is tested against this polygon")
