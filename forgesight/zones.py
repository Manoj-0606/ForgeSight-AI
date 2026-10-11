"""Restricted-zone geometry and entry/exit state machine.

Entries are confirmed with hysteresis so that a single noisy detection does not raise an alarm,
and a person standing inside the zone produces exactly one event. Re-entry produces a new event
only after an exit has been confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import cv2
import numpy as np


# ---------------------------------------------------------------- geometry

def polygon_to_pixels(points_pct: list[list[float]], width: int, height: int) -> np.ndarray:
    pts = [[p[0] / 100.0 * width, p[1] / 100.0 * height] for p in points_pct]
    return np.array(pts, dtype=np.float32)


def point_in_polygon(polygon_px: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(polygon_px.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def foot_point(xyxy) -> tuple[float, float]:
    """Bottom-centre of a bounding box: the person's approximate floor contact point."""
    x1, _, x2, y2 = xyxy
    return (x1 + x2) / 2.0, float(y2)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def validate_polygon(points_pct: list[list[float]]) -> tuple[bool, str]:
    """Return (ok, message) for a polygon given in percent coordinates."""
    if len(points_pct) < 3:
        return False, "A zone needs at least 3 points."
    for i, p in enumerate(points_pct, start=1):
        if len(p) != 2 or any(v is None for v in p):
            return False, f"Point {i} is incomplete."
        if not (0 <= p[0] <= 100 and 0 <= p[1] <= 100):
            return False, f"Point {i} is outside the frame (values must be 0–100%)."
    xs = np.array([p[0] for p in points_pct], dtype=float)
    ys = np.array([p[1] for p in points_pct], dtype=float)
    area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    if area < 0.5:  # percent² of frame
        return False, "The zone has (almost) no area — spread the points out."
    n = len(points_pct)
    edges = [(points_pct[i], points_pct[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) in (1, n - 1):
                continue  # adjacent edges share a vertex
            if _segments_intersect(*edges[i], *edges[j]):
                return False, "The zone edges cross each other — reorder the points around the shape."
    return True, f"Zone covers {area / 100:.1f}% of the frame."


# ---------------------------------------------------------------- state machine

@dataclass
class ZoneVisit:
    key: Hashable  # track id, or "zone" when tracking is disabled
    entry_t: float  # first analyzed frame where the subject was inside
    confirmed_t: float  # frame at which the entry was confirmed (event raised)
    confirmed_frame: int
    exit_t: float | None = None
    exit_reason: str | None = None

    @property
    def dwell_s(self) -> float | None:
        return None if self.exit_t is None else max(0.0, self.exit_t - self.entry_t)


@dataclass
class _KeyState:
    inside_streak: int = 0
    outside_streak: int = 0
    missing: int = 0
    streak_start_t: float = 0.0
    last_inside_t: float = 0.0
    visit: ZoneVisit | None = None  # set while a confirmed visit is open


@dataclass
class ZoneMonitor:
    entry_frames: int = 2
    exit_frames: int = 3
    lost_frames: int = 12  # unseen this many analyzed frames while inside → visit closed
    visits: list[ZoneVisit] = field(default_factory=list)
    _states: dict = field(default_factory=dict)

    def update(self, t: float, frame_idx: int, observations: dict[Hashable, bool]) -> list[ZoneVisit]:
        """Feed one analyzed frame. Returns visits whose entry was confirmed on this frame."""
        new_visits: list[ZoneVisit] = []

        for key, inside in observations.items():
            st = self._states.setdefault(key, _KeyState())
            st.missing = 0
            if inside:
                if st.inside_streak == 0:
                    st.streak_start_t = t
                st.inside_streak += 1
                st.outside_streak = 0
                st.last_inside_t = t
                if st.visit is None and st.inside_streak >= self.entry_frames:
                    st.visit = ZoneVisit(key, st.streak_start_t, t, frame_idx)
                    self.visits.append(st.visit)
                    new_visits.append(st.visit)
            else:
                st.outside_streak += 1
                st.inside_streak = 0
                if st.visit is not None and st.outside_streak >= self.exit_frames:
                    self._close(st, "left zone")

        for key in list(self._states):
            if key in observations:
                continue
            st = self._states[key]
            st.missing += 1
            st.inside_streak = 0
            if st.missing >= self.lost_frames:
                if st.visit is not None:
                    self._close(st, "track lost")
                del self._states[key]

        return new_visits

    def is_confirmed_inside(self, key: Hashable) -> bool:
        st = self._states.get(key)
        return bool(st and st.visit is not None)

    def finish(self, t_end: float) -> None:
        """Close visits still open when the video ends."""
        for st in self._states.values():
            if st.visit is not None:
                # Unseen in the final frames: it disappeared before the end, not cut off by it.
                self._close(st, "track lost" if st.missing > 0 else "video ended")

    @staticmethod
    def _close(st: _KeyState, reason: str) -> None:
        st.visit.exit_t = st.last_inside_t
        st.visit.exit_reason = reason
        st.visit = None
