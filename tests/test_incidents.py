from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgesight import events as evt
from forgesight import incidents as inc
from forgesight.config import Settings

S = Settings(sev_critical_dwell_s=5.0, sev_brief_dwell_s=0.5, occ_alert_threshold=2, occ_alert_min_s=1.0)


def zrow(event_type=evt.TYPE_ENTRY, dwell=2.0, lb=False, exit_reason="left zone"):
    return SimpleNamespace(event_type=event_type, dwell_s=dwell, dwell_is_lower_bound=lb, exit_reason=exit_reason)


@pytest.mark.parametrize("row,expected", [
    (zrow(dwell=6.0), "Critical"),
    (zrow(dwell=5.0), "Critical"),  # boundary inclusive
    (zrow(dwell=2.0), "High"),
    (zrow(dwell=0.2), "Medium"),  # brief incursion
    (zrow(dwell=0.2, lb=True, exit_reason="video ended"), "High"),  # cut off: cannot call it brief
    (zrow(dwell=7.0, lb=True, exit_reason="video ended"), "Critical"),  # lower bound already critical
    (zrow(event_type=evt.TYPE_AT_START, dwell=9.0, lb=True), "Medium"),  # no entry observed
])
def test_zone_severity_rules(row, expected):
    sev, reason = inc.zone_severity(row, S)
    assert sev == expected and reason


def _fake_run(tmp_path):
    events = pd.DataFrame({
        "event_id": ["EVT-001", "EVT-002"], "event_type": [evt.TYPE_AT_START, evt.TYPE_ENTRY],
        "entry_context": [evt.CTX_AT_START, evt.CTX_CROSSED], "track_id": pd.array([1, 2], dtype="Int64"),
        "entry_time_s": [0.0, 3.0], "exit_time_s": [2.0, 9.0], "dwell_s": [2.0, 6.0],
        "dwell_is_lower_bound": [True, False], "exit_reason": ["left zone", "left zone"],
        "evidence_image": ["a.jpg", "b.jpg"], "evidence_crop": ["a_c.jpg", "b_c.jpg"]})
    n = 50
    frames = pd.DataFrame({"time_s": np.arange(n) * 0.2, "dt_s": 0.2, "people": 3,
                           "people_in_zone": [0] * 20 + [2] * 15 + [0] * 15})
    return SimpleNamespace(run_dir=tmp_path, events=events, frames=frames, ppe=pd.DataFrame(), ppe_active=False,
                           metrics={"zone_enabled": True}, summary={"settings": {"zone_name": "Cell"}})


def test_build_incidents_and_review_roundtrip(tmp_path):
    run = _fake_run(tmp_path)
    df = inc.build_incidents(run, S, with_evidence=False)
    assert list(df["incident_id"]) == ["EVT-001", "EVT-002", "OCC-001"]
    assert list(df["severity"]) == ["Medium", "Critical", "High"]  # 3.0 s ≥ 2 × 1.0 s
    assert (df["status"] == inc.OPEN).all()

    inc.save_review(tmp_path, ["EVT-002"], inc.REVIEWED, "Maintenance with permit")
    df = inc.build_incidents(run, S, with_evidence=False)
    r = df.set_index("incident_id").loc["EVT-002"]
    assert r.status == inc.REVIEWED and r.note == "Maintenance with permit" and r.reviewed_at
    inc.save_review(tmp_path, ["EVT-002"], inc.OPEN)
    r = inc.build_incidents(run, S, with_evidence=False).set_index("incident_id").loc["EVT-002"]
    assert r.status == inc.OPEN and r.note == "Maintenance with permit" and r.reviewed_at is None


def test_occupancy_review_does_not_follow_a_different_alert(tmp_path):
    """Changing the threshold yields a different period; its review status must not carry over."""
    run = _fake_run(tmp_path)
    df = inc.build_incidents(run, S, with_evidence=False)
    key = df.set_index("incident_id").loc["OCC-001", "review_key"]
    inc.save_review(tmp_path, [key], inc.REVIEWED)
    other = Settings(**{**S.to_dict(), "occ_alert_threshold": 1})
    df2 = inc.build_incidents(run, other, with_evidence=False)
    assert df2.set_index("incident_id").loc["OCC-001", "status"] == inc.OPEN


def test_filters(tmp_path):
    df = inc.build_incidents(_fake_run(tmp_path), S, with_evidence=False)
    assert list(inc.filter_incidents(df, severities=["Critical"])["incident_id"]) == ["EVT-002"]
    assert list(inc.filter_incidents(df, types=["Sustained occupancy"])["incident_id"]) == ["OCC-001"]
    assert list(inc.filter_incidents(df, track="1")["incident_id"]) == ["EVT-001"]
    assert inc.filter_incidents(df, track="abc").empty
    assert list(inc.filter_incidents(df, time_range=(2.5, 5.0))["incident_id"]) == ["EVT-002", "OCC-001"]


def test_ppe_incidents_grouped_per_person(tmp_path):
    from forgesight import ppe
    run = _fake_run(tmp_path)
    run.ppe_active = True
    run.ppe = pd.DataFrame({
        "track_id": [7, 7, 9, 9], "item": ["hardhat", "vest", "hardhat", "vest"],
        "item_label": ["Hard hat", "Hi-vis vest", "Hard hat", "Hi-vis vest"],
        "status": [ppe.NOT_DETECTED, ppe.NOT_DETECTED, ppe.DETECTED, ppe.NOT_DETECTED],
        "checks": [6, 6, 5, 5], "hits": [0, 0, 5, 0], "last_check_s": [3.0, 3.0, 4.0, 4.0],
        "evidence_time_s": [1.0, 1.2, None, 2.0], "event_id": ["PPE-001", "PPE-002", "", "PPE-003"],
        "evidence_image": ["p1.jpg", "p2.jpg", "", "p3.jpg"], "evidence_crop": ["c1", "c2", "", "c3"]})
    df = inc.build_incidents(run, S, with_evidence=False)
    p = df[df["source"] == "ppe"].set_index("incident_id")
    assert list(p.index) == ["PPE-001", "PPE-003"]
    assert p.loc["PPE-001", "context"] == "Hard hat & Hi-vis vest not detected"
    assert p.loc["PPE-003", "context"] == "Hi-vis vest not detected"  # hard hat was detected on ID 9
    assert (p["severity"] == "Medium").all()
