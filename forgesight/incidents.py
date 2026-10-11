"""Unified incident list (zone events, sustained occupancy, PPE) with rule-based severity and review status.

Severity is derived at view time from transparent, configurable rules; each incident carries the reason.
Review status lives in <run>/review.json — a local file, no database.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import analytics
from . import events as evt
from . import ppe
from .config import Settings

SEVERITIES = ["Critical", "High", "Medium", "Low"]
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
OPEN, REVIEWED = "Open", "Reviewed"
SOURCES = {"zone": "Restricted zone", "occupancy": "Sustained occupancy", "ppe": "PPE (beta)",
           "hazard": "Fire / smoke (beta)"}

COLUMNS = ["incident_id", "review_key", "source", "type", "severity", "severity_reason", "status", "note",
           "reviewed_at", "time_s", "end_s", "track_id", "context", "dwell", "evidence_image", "evidence_crop",
           "video_time_s", "notification"]


def severity_rules(s: Settings) -> list[tuple[str, str]]:
    """Human-readable rule table, shown in Settings and in the report."""
    return [
        ("Zone intrusion, dwell ≥ %.1f s" % s.sev_critical_dwell_s, "Critical"),
        ("Zone intrusion, dwell between %.1f s and %.1f s, or dwell cut off by the end of footage / "
         "lost track" % (s.sev_brief_dwell_s, s.sev_critical_dwell_s), "High"),
        ("Zone intrusion, observed dwell < %.1f s (brief boundary incursion)" % s.sev_brief_dwell_s, "Medium"),
        ("Person already in zone when footage began (no entry observed)", "Medium"),
        ("Sustained occupancy ≥ 2× the minimum duration", "High"),
        ("Sustained occupancy", "Medium"),
        ("PPE item not detected in clear view (beta, unverified)", "Medium"),
        ("Potential fire confirmed (beta, verify on video)", "Critical"),
        ("Potential smoke confirmed (beta, verify on video)", "High"),
    ]


def zone_severity(row, s: Settings) -> tuple[str, str]:
    if row.event_type == evt.TYPE_AT_START:
        return "Medium", "Already inside when the footage began; no entry was observed"
    dwell, lb = row.dwell_s, bool(row.dwell_is_lower_bound)
    shown = evt.format_dwell(dwell, lb)
    if dwell is None or pd.isna(dwell):
        return "High", "Intrusion; dwell time unknown"
    if dwell >= s.sev_critical_dwell_s:
        return "Critical", f"Dwell {shown} ≥ {s.sev_critical_dwell_s:g} s critical limit"
    if lb:
        return "High", f"Dwell {shown}: true dwell unknown ({row.exit_reason})"
    if dwell < s.sev_brief_dwell_s:
        return "Medium", f"Brief incursion: dwell {shown} < {s.sev_brief_dwell_s:g} s (possible boundary flicker)"
    return "High", f"Intrusion with dwell {shown}"


def occupancy_severity(a, s: Settings) -> tuple[str, str]:
    where = "zone" if a.scope == "zone" else "camera view"
    base = f"≥ {a.threshold} people in {where} for {a.duration_s:.1f} s (peak {a.peak})"
    if a.duration_s >= 2 * s.occ_alert_min_s:
        return "High", f"{base}; ≥ 2× the {s.occ_alert_min_s:g} s minimum"
    return "Medium", base


# ------------------------------------------------------------------ review store

def _review_path(run_dir: Path) -> Path:
    return Path(run_dir) / "review.json"


def load_reviews(run_dir: Path) -> dict:
    try:
        return json.loads(_review_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_review(run_dir: Path, keys: list[str], status: str, note: str | None = None) -> None:
    reviews = load_reviews(run_dir)
    now = datetime.now().isoformat(timespec="seconds")
    for k in keys:
        entry = reviews.get(k, {})
        entry["status"] = status
        entry["reviewed_at"] = now if status == REVIEWED else None
        if note is not None:
            entry["note"] = note
        reviews[k] = entry
    path = _review_path(run_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ incidents

def occupancy_alerts(run, s: Settings) -> pd.DataFrame:
    if not s.occ_alert_enabled:
        return pd.DataFrame(columns=analytics.ALERT_COLUMNS)
    scope = s.occ_alert_scope if run.metrics.get("zone_enabled") else "frame"
    return analytics.sustained_occupancy(run.frames, scope, int(s.occ_alert_threshold), float(s.occ_alert_min_s))


def build_incidents(run, s: Settings, with_evidence: bool = True) -> pd.DataFrame:
    rows = []
    for r in run.events.itertuples():
        sev, why = zone_severity(r, s)
        rows.append({
            "incident_id": r.event_id, "review_key": r.event_id, "source": "zone",
            "type": evt.TYPE_LABELS[r.event_type], "severity": sev, "severity_reason": why,
            "time_s": r.entry_time_s, "end_s": r.exit_time_s, "track_id": r.track_id, "context": r.entry_context,
            "dwell": evt.format_dwell(r.dwell_s, r.dwell_is_lower_bound),
            "evidence_image": r.evidence_image, "evidence_crop": r.evidence_crop, "video_time_s": r.entry_time_s,
        })
    zone_name = run.summary["settings"].get("zone_name", "zone")
    for a in occupancy_alerts(run, s).itertuples():
        sev, why = occupancy_severity(a, s)
        evidence = analytics.occupancy_alert_evidence(run.run_dir, a._asdict(), zone_name) if with_evidence else ""
        rows.append({
            "incident_id": a.alert_id, "review_key": f"OCC|{a.scope}|{a.threshold}|{a.start_s:.2f}",
            "source": "occupancy", "type": "Sustained occupancy", "severity": sev, "severity_reason": why,
            "time_s": a.start_s, "end_s": a.end_s, "track_id": None,
            "context": f"{SCOPES_LABEL[a.scope]} ≥ {a.threshold} for ≥ {s.occ_alert_min_s:g} s",
            "dwell": f"{a.duration_s:.1f} s", "evidence_image": evidence, "evidence_crop": evidence,
            "video_time_s": a.peak_time_s,
        })
    if run.ppe_active and len(run.ppe):
        # One incident per person, listing every PPE item that was not detected on them.
        missing = run.ppe[run.ppe["status"] == ppe.NOT_DETECTED].sort_values(["evidence_time_s", "event_id"])
        for tid, g in missing.groupby("track_id", sort=False):
            first = g.iloc[0]
            items = " & ".join(g["item_label"])
            checks = "; ".join(f"{r.item_label.lower()} found in {r.hits}/{r.checks}" for r in g.itertuples())
            rows.append({
                "incident_id": first.event_id, "review_key": f"PPE|{tid}", "source": "ppe",
                "type": "PPE not detected", "severity": "Medium",
                "severity_reason": f"{checks} clear-view checks (beta, unverified: confirm on video)",
                "time_s": first.evidence_time_s, "end_s": g["last_check_s"].max(), "track_id": tid,
                "context": f"{items} not detected", "dwell": "—",
                "evidence_image": first.evidence_image, "evidence_crop": first.evidence_crop,
                "video_time_s": first.evidence_time_s,
            })
    hz = getattr(run, "hazards", None)
    if getattr(run, "hazard_active", False) and hz is not None and len(hz):
        thr = run.summary["hazard"]["threshold"]
        k = run.summary["hazard"]["confirm_checks"]
        for h in hz.itertuples():
            label = "fire" if h.hazard == "fire" else "smoke"
            dur = (h.end_time_s - h.start_time_s) if pd.notna(h.end_time_s) else None
            rows.append({
                "incident_id": h.event_id, "review_key": h.event_id, "source": "hazard",
                "type": f"Potential {label}", "severity": "Critical" if label == "fire" else "High",
                "severity_reason": f"{k} consecutive checks with {label} score ≥ {thr:.2f} (max {h.max_score:.2f}); "
                                   "beta classifier, verify on video",
                "time_s": h.start_time_s, "end_s": h.end_time_s, "track_id": None,
                "context": f"Confirmed at {h.confirmed_time_s:.1f} s · email: {h.email_status}",
                "dwell": "—" if dur is None else f"{dur:.1f} s",
                "evidence_image": h.evidence_image, "evidence_crop": h.evidence_image,
                "video_time_s": h.start_time_s, "notification": h.email_status,
            })
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("status", "note", "reviewed_at")])
    df["notification"] = df["notification"].fillna("")
    reviews = load_reviews(run.run_dir)
    df["status"] = [reviews.get(k, {}).get("status", OPEN) for k in df["review_key"]]
    df["note"] = [reviews.get(k, {}).get("note", "") or "" for k in df["review_key"]]
    df["reviewed_at"] = [reviews.get(k, {}).get("reviewed_at") for k in df["review_key"]]
    df["track_id"] = df["track_id"].astype("Int64")
    df["sev_rank"] = df["severity"].map(SEV_RANK)
    return df.sort_values(["time_s", "sev_rank"], kind="stable").reset_index(drop=True)[COLUMNS + ["sev_rank"]]


SCOPES_LABEL = {"zone": "People in zone", "frame": "People in view"}


def filter_incidents(df: pd.DataFrame, types=None, severities=None, statuses=None, time_range=None,
                     track: str = "") -> pd.DataFrame:
    out = df
    if types:
        out = out[out["type"].isin(types)]
    if severities:
        out = out[out["severity"].isin(severities)]
    if statuses:
        out = out[out["status"].isin(statuses)]
    if time_range:
        out = out[(out["time_s"] >= time_range[0]) & (out["time_s"] <= time_range[1])]
    if track.strip():
        ids = {int(x) for x in track.replace(",", " ").split() if x.strip().isdigit()}
        out = out[out["track_id"].isin(ids)] if ids else out.iloc[0:0]
    return out
