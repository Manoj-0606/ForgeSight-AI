"""Derived manufacturing analytics: occupancy trend, sustained-occupancy alerts, heatmap statistics.

Everything here is computed from the per-frame detections saved with a run (frames.csv, heatmap grids),
so thresholds can be changed after the analysis without re-processing the video.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .annotate import AMBER_DARK, draw_alert_banner, fmt_ts
from .zones import point_in_polygon, polygon_to_pixels

# Dips below the threshold shorter than this do not split an alert (detection flicker, brief occlusion).
OCC_GAP_TOLERANCE_S = 0.5
SCOPES = {"zone": "people in the restricted zone", "frame": "people in the camera view"}

ALERT_COLUMNS = ["alert_id", "scope", "threshold", "start_s", "end_s", "duration_s", "peak", "mean",
                 "peak_time_s", "peak_sample"]


def occupancy_series(frames: pd.DataFrame, scope: str) -> pd.Series:
    return frames["people_in_zone" if scope == "zone" else "people"]


def trend(frames: pd.DataFrame, scope: str, window_s: float) -> pd.Series:
    """Centred rolling mean over roughly ``window_s`` seconds of analyzed frames."""
    counts = occupancy_series(frames, scope).astype(float)
    if len(frames) < 2 or window_s <= 0:
        return counts
    dt = float(frames["dt_s"].median()) if "dt_s" in frames else float(frames["time_s"].diff().median())
    k = max(1, int(round(window_s / max(dt, 1e-6))))
    return counts.rolling(window=k, center=True, min_periods=1).mean()


def sustained_occupancy(frames: pd.DataFrame, scope: str, threshold: int, min_duration_s: float,
                        gap_tolerance_s: float = OCC_GAP_TOLERANCE_S) -> pd.DataFrame:
    """Periods where occupancy stays ≥ threshold for at least ``min_duration_s``.

    A period runs from the first to the end of the last qualifying analyzed frame; dips shorter than
    ``gap_tolerance_s`` are bridged. Each frame represents ``dt_s`` seconds of video.
    """
    if frames.empty or threshold < 1:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    counts = occupancy_series(frames, scope).to_numpy()
    times = frames["time_s"].to_numpy(dtype=float)
    dts = frames["dt_s"].to_numpy(dtype=float)
    above = counts >= threshold

    periods, start, last, gap = [], None, None, 0.0
    for i, ok in enumerate(above):
        if ok:
            if start is None:
                start = i
            last, gap = i, 0.0
        elif start is not None:
            gap += dts[i]
            if gap > gap_tolerance_s:
                periods.append((start, last))
                start, last, gap = None, None, 0.0
    if start is not None:
        periods.append((start, last))

    rows = []
    for s, e in periods:
        duration = times[e] + dts[e] - times[s]
        if duration + 1e-9 < min_duration_s:
            continue
        seg = counts[s:e + 1]
        peak_i = s + int(np.argmax(seg))
        rows.append({"scope": scope, "threshold": int(threshold), "start_s": round(times[s], 2),
                     "end_s": round(times[e] + dts[e], 2), "duration_s": round(duration, 2),
                     "peak": int(seg.max()), "mean": round(float(seg.mean()), 2),
                     "peak_time_s": round(times[peak_i], 2), "peak_sample": int(peak_i)})
    df = pd.DataFrame(rows, columns=ALERT_COLUMNS[1:])
    df.insert(0, "alert_id", [f"OCC-{i + 1:03d}" for i in range(len(df))])
    return df


def read_annotated_frame(video: Path, sample: int) -> np.ndarray | None:
    """Frame ``sample`` of the annotated video (one frame per analyzed sample). Read sequentially for accuracy."""
    cap = cv2.VideoCapture(str(video))
    try:
        for _ in range(sample):
            if not cap.grab():
                return None
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def occupancy_alert_evidence(run_dir: Path, alert: dict, zone_name: str) -> str:
    """Relative path of an evidence frame for a sustained-occupancy alert (created once, then reused)."""
    rel = (f"evidence/OCC_{alert['scope']}_ge{alert['threshold']}_{alert['start_s']:07.2f}s"
           f"_s{alert['peak_sample']}.jpg")
    path = run_dir / rel
    if path.exists():
        return rel
    frame = read_annotated_frame(run_dir / "annotated.mp4", int(alert["peak_sample"]))
    if frame is None:
        return ""
    where = zone_name if alert["scope"] == "zone" else "camera view"
    draw_alert_banner(frame, "SUSTAINED OCCUPANCY",
                      f"{alert['peak']} PEOPLE | {where} | >= {alert['threshold']} FOR {alert['duration_s']:.1f} S | "
                      f"{fmt_ts(alert['start_s'])}-{fmt_ts(alert['end_s'])}", AMBER_DARK)
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rel


def zone_mask(grid_shape: tuple[int, int], zone_points: list) -> np.ndarray:
    """Boolean mask of heatmap cells whose centre lies inside the zone polygon."""
    gh, gw = grid_shape
    poly = polygon_to_pixels(zone_points, gw, gh)
    mask = np.zeros(grid_shape, dtype=bool)
    for y in range(gh):
        for x in range(gw):
            mask[y, x] = point_in_polygon(poly, x + 0.5, y + 0.5)
    return mask


def heatmap_stats(heat: np.ndarray, zone_points: list | None) -> dict:
    total = float(heat.sum())
    if total <= 0:
        return {"total": 0.0, "zone_share": None, "hotspot_pct": None}
    gy, gx = np.unravel_index(int(np.argmax(heat)), heat.shape)
    stats = {"total": total, "hotspot_pct": ((gx + 0.5) / heat.shape[1] * 100, (gy + 0.5) / heat.shape[0] * 100),
             "zone_share": None}
    if zone_points:
        stats["zone_share"] = float(heat[zone_mask(heat.shape, zone_points)].sum() / total)
    return stats
