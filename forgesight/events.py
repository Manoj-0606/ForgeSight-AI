"""Safety-event vocabulary, entry classification and upgrade of runs saved by earlier versions.

Every zone event is one of:
- RESTRICTED_ZONE_ENTRY ("intrusion"), with context
    * "crossed into zone"            – the track was observed outside the zone before entering
    * "first detected inside zone"   – the track's first detections were already inside (walked into the
                                       camera view inside the zone, or reappeared from occlusion); the
                                       crossing itself was not observed
    * "zone became occupied"         – tracking disabled: zone went from empty to occupied
- ZONE_OCCUPIED_AT_START, context "present at start of footage" – already inside when the footage began.
  Not counted as an intrusion: nobody was observed entering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPE_ENTRY = "RESTRICTED_ZONE_ENTRY"
TYPE_AT_START = "ZONE_OCCUPIED_AT_START"

CTX_CROSSED = "crossed into zone"
CTX_FIRST_INSIDE = "first detected inside zone"
CTX_AT_START = "present at start of footage"
CTX_UNTRACKED = "zone became occupied"

# A tracked person first detected inside the zone within this many seconds of the first analyzed frame
# (and never seen outside before) is treated as already present when the footage started.
START_GRACE_S = 1.0

# Exit reasons for which the true exit time is unknown, so the measured dwell is a lower bound.
CENSORED_EXITS = ("video ended", "track lost")

TYPE_LABELS = {TYPE_ENTRY: "Zone intrusion", TYPE_AT_START: "Present at start"}

SCHEMA_VERSION = 2


def classify_tracked_entry(entry_t: float, first_seen_t: float, last_outside_t: float | None,
                           first_frame_t: float) -> str:
    if last_outside_t is not None and last_outside_t < entry_t:
        return CTX_CROSSED
    if first_seen_t <= first_frame_t + START_GRACE_S:
        return CTX_AT_START
    return CTX_FIRST_INSIDE


def classify_untracked_entry(entry_t: float, first_frame_t: float) -> str:
    return CTX_AT_START if entry_t <= first_frame_t + 1e-6 else CTX_UNTRACKED


def event_type_for(context: str) -> str:
    return TYPE_AT_START if context == CTX_AT_START else TYPE_ENTRY


def dwell_is_lower_bound(context: str, exit_reason) -> bool:
    return context == CTX_AT_START or (isinstance(exit_reason, str) and exit_reason in CENSORED_EXITS)


def format_dwell(seconds, lower_bound=False) -> str:
    """'4.2 s', or '≥ 4.2 s' when the true dwell is longer than what was observed."""
    if seconds is None or pd.isna(seconds):
        return "—"
    return f"{'≥ ' if bool(lower_bound) else ''}{float(seconds):.1f} s"


def sample_durations(times: np.ndarray, end_s: float | None, nominal_dt: float) -> np.ndarray:
    """Seconds of video represented by each analyzed frame.

    Each sample stands for the interval until the next sample; the last one for the remainder of the
    analyzed span (at most one nominal interval). Sum == analyzed duration, never longer than the clip.
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return times
    last = nominal_dt
    if end_s is not None and end_s > times[-1]:
        last = min(nominal_dt, end_s - times[-1])
    return np.append(np.diff(times), last)


def intrusions(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["event_type"] == TYPE_ENTRY] if len(events) else events


def at_start(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["event_type"] == TYPE_AT_START] if len(events) else events


def upgrade_legacy_run(summary: dict, frames: pd.DataFrame, events: pd.DataFrame,
                       tracks: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Bring runs saved before schema 2 up to the current event model (in memory only).

    Earlier versions counted people present at the start as entries and labelled every other event
    "entered zone". The track table lets us recover whether a crossing was actually observed.
    """
    if summary.get("schema_version", 1) >= SCHEMA_VERSION:
        return summary, frames, events
    summary = {**summary, "metrics": dict(summary["metrics"]), "processing": dict(summary["processing"])}
    events = events.copy()
    tracked = bool(summary.get("tracking", {}).get("active"))
    first_seen = dict(zip(tracks["track_id"], tracks["first_seen_s"])) if len(tracks) else {}

    first_frame_t = float(frames["time_s"].iloc[0]) if len(frames) else 0.0
    contexts = []
    for r in events.itertuples():
        old = getattr(r, "entry_context", "entered zone")
        if old == "in zone at start of footage":
            contexts.append(CTX_AT_START)
        elif not tracked or pd.isna(r.track_id):
            contexts.append(classify_untracked_entry(r.entry_time_s, first_frame_t))
        else:
            seen = first_seen.get(int(r.track_id), r.entry_time_s)
            # Seen before the entry streak began ⇒ it was outside then (legacy runs lack last_outside_t).
            outside = seen if seen < r.entry_time_s - 1e-6 else None
            contexts.append(classify_tracked_entry(r.entry_time_s, seen, outside, first_frame_t))
    if len(events):
        events["entry_context"] = contexts
        events["event_type"] = [event_type_for(c) for c in contexts]
        events["dwell_is_lower_bound"] = [dwell_is_lower_bound(c, e) for c, e in
                                          zip(contexts, events["exit_reason"])]
    else:
        events["entry_context"] = pd.Series(dtype=str)
        events["dwell_is_lower_bound"] = pd.Series(dtype=bool)

    fps = float(summary["processing"].get("analysis_fps") or 1.0)
    end = summary.get("video", {}).get("duration_s") or None
    if len(frames) and "dt_s" not in frames:
        frames = frames.assign(dt_s=sample_durations(frames["time_s"].to_numpy(), end, 1.0 / fps))
    m = summary["metrics"]
    m["zone_entries"] = int(len(intrusions(events)))
    m["zone_initial_occupants"] = int(len(at_start(events)))
    if len(frames):
        summary["processing"]["analyzed_duration_s"] = round(float(frames["dt_s"].sum()), 2)
        if m.get("zone_enabled"):
            m["zone_occupied_s"] = round(float(frames.loc[frames["zone_occupied"], "dt_s"].sum()), 2)
    summary["schema_version"] = SCHEMA_VERSION
    summary["upgraded_from_legacy"] = True
    return summary, frames, events
