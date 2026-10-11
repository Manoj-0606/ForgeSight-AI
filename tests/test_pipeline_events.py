"""Regression tests: run the real pipeline (ByteTrack, zone logic, annotation, persistence) on a synthetic
video with a scripted detector, so every expected event is known exactly."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

import forgesight.pipeline as pipeline
from forgesight import events as evt
from forgesight.config import Settings
from forgesight.results import events_export, load_run

W, H, FPS, N = 320, 240, 10, 60
ZONE = [[50, 0], [100, 0], [100, 100], [50, 100]]  # right half: foot x >= 160 px


def box(cx, foot_y, w=20, h=60):
    return [cx - w / 2, foot_y - h, cx + w / 2, foot_y]


def script():
    frames = []
    for f in range(N):
        dets = []
        # A: inside at the start, walks out at 4 px/frame (last inside at f=36, x=162), keeps walking outside
        ax = 250 if f < 15 else 250 - (f - 14) * 4
        dets.append(box(ax, 200))
        # B: starts outside, crosses into the zone at f=30, still inside when the video ends
        dets.append(box(40 + 4 * f, 120))
        # C: appears inside the zone at f=40 (first detected inside), vanishes after f=49
        if 40 <= f <= 49:
            dets.append(box(290, 230))
        # E: outside near the boundary, a single analyzed frame inside (f=12) must NOT raise an event
        if 5 <= f <= 20:
            dets.append(box(165 if f == 12 else 145, 60))
        frames.append(dets)
    return frames


class ScriptedModel:
    name, version = "scripted", "test"

    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def detect_people(self, frame, conf):
        dets = self.frames[self.i] if self.i < len(self.frames) else []
        self.i += 1
        xyxy = np.array(dets, dtype=float).reshape(-1, 4)
        return xyxy, np.full(len(xyxy), 0.9)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    video = tmp / "synthetic.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(img, str(f), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        w.write(img)
    w.release()
    runs = tmp / "runs"
    runs.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "RUNS_DIR", runs)
    settings = Settings(analysis_fps=FPS, zone_points=ZONE, entry_frames=2, exit_frames=3, zone_name="Test Cell")
    run_dir = pipeline.analyze_video(video, settings, ScriptedModel(script()))
    mp.undo()
    return load_run(run_dir)


def _by_context(run, ctx):
    rows = run.events[run.events["entry_context"] == ctx]
    assert len(rows) == 1, f"expected exactly one '{ctx}' event, got {len(rows)}"
    return rows.iloc[0]


def test_exactly_three_events_and_no_flicker_event(run):
    assert len(run.events) == 3
    assert run.events["track_id"].nunique() == 3
    assert list(run.events["event_id"]) == ["EVT-001", "EVT-002", "EVT-003"]


def test_initial_occupant_is_not_an_intrusion(run):
    a = _by_context(run, evt.CTX_AT_START)
    assert a.event_type == evt.TYPE_AT_START
    assert a.entry_time_s == 0.0
    assert a.exit_time_s == pytest.approx(3.6) and a.exit_reason == "left zone"
    assert a.dwell_is_lower_bound  # it was already inside before the footage began
    assert run.metrics["zone_entries"] == 2
    assert run.metrics["zone_initial_occupants"] == 1


def test_observed_crossing(run):
    b = _by_context(run, evt.CTX_CROSSED)
    assert b.event_type == evt.TYPE_ENTRY
    assert b.entry_time_s == pytest.approx(3.0)
    assert b.confirmed_time_s == pytest.approx(3.1)  # entry_frames=2
    assert b.exit_reason == "video ended" and b.dwell_is_lower_bound
    assert b.dwell_s == pytest.approx(2.9)


def test_first_detected_inside(run):
    c = _by_context(run, evt.CTX_FIRST_INSIDE)
    assert c.event_type == evt.TYPE_ENTRY
    assert 4.0 <= c.entry_time_s <= 4.1  # ByteTrack activates new tracks on their 1st or 2nd frame
    assert c.exit_reason == "track lost" and c.dwell_is_lower_bound
    assert c.exit_time_s == pytest.approx(4.9)


def test_durations_never_exceed_clip(run):
    p, m = run.summary["processing"], run.metrics
    assert run.frames["dt_s"].sum() == pytest.approx(N / FPS)
    assert p["analyzed_duration_s"] == pytest.approx(N / FPS)
    assert m["zone_occupied_s"] <= p["analyzed_duration_s"] + 1e-9
    assert run.summary["schema_version"] == evt.SCHEMA_VERSION


def test_evidence_banners_match_event_type(run):
    """Present-at-start evidence uses the amber 'PERSON IN ZONE AT START' banner, intrusions the red one."""
    for e in run.events.itertuples():
        img = cv2.imread(str(run.path(e.evidence_image)))
        assert img is not None and cv2.imread(str(run.path(e.evidence_crop))) is not None
        b, g, r = img[4, W // 2].astype(int)
        if e.event_type == evt.TYPE_AT_START:
            assert r < 200 and b < 60, (b, g, r)  # dark amber
        else:
            assert r > 200 and b > 40, (b, g, r)  # red


def test_annotated_video_and_frames_consistent(run):
    cap = cv2.VideoCapture(str(run.video_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(run.frames) == N
    cap.release()
    assert run.frames["people"].sum() == run.metrics["total_detections"]


def test_csv_export_roundtrip(run, tmp_path):
    df = events_export(run)
    path = tmp_path / "events.csv"
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert len(back) == 3
    assert set(back["event_type"]) == {evt.TYPE_ENTRY, evt.TYPE_AT_START}
    assert back["dwell_is_lower_bound"].all()
    for p in back["evidence_image"]:
        assert cv2.imread(p) is not None


def _write_legacy_run(d):
    """A run in the format saved before schema 2 (contexts 'entered zone' / 'in zone at start')."""
    d.mkdir()
    times = np.round(np.arange(0, 70) * 0.2, 2)  # 70 samples at 5 fps = 14.0 s
    pd.DataFrame({"sample": range(70), "frame_index": range(0, 350, 5), "time_s": times,
                  "people": 3, "people_in_zone": 1, "zone_occupied": True,
                  "mean_confidence": 0.7}).to_csv(d / "frames.csv", index=False)
    pd.DataFrame({
        "event_id": ["EVT-001", "EVT-002", "EVT-003"], "event_type": "RESTRICTED_ZONE_ENTRY",
        "entry_context": ["in zone at start of footage", "entered zone", "entered zone"],
        "zone_name": "Z", "track_id": [1, 2, 3], "entry_time_s": [0.0, 4.0, 9.0],
        "confirmed_time_s": [0.2, 4.2, 9.2], "exit_time_s": [3.0, 6.0, 13.8], "dwell_s": [3.0, 2.0, 4.8],
        "exit_reason": ["left zone", "left zone", "video ended"], "frame_index": [5, 105, 230],
        "confidence": 0.8, "foot_x_pct": 70, "foot_y_pct": 70, "people_in_zone": 1,
        "evidence_image": "", "evidence_crop": "", "video_reference": "",
    }).to_csv(d / "events.csv", index=False)
    pd.DataFrame({"track_id": [1, 2, 3], "first_seen_s": [0.0, 1.0, 9.0], "last_seen_s": [5, 8, 13.8],
                  "frames": 20, "zone_frames": 10}).to_csv(d / "tracks.csv", index=False)
    np.save(d / "heatmap.npy", np.zeros((10, 10)))
    (d / "summary.json").write_text(json.dumps({
        "run_id": "legacy", "original_name": "x.mp4", "camera_label": "CAM",
        "video": {"duration_s": 14.0}, "settings": {}, "tracking": {"active": True},
        "processing": {"analysis_fps": 5.0, "analyzed_duration_s": 13.8},
        "metrics": {"zone_entries": 3, "zone_occupied_s": 14.0, "zone_enabled": True},
    }))


def test_legacy_runs_are_reclassified(tmp_path):
    _write_legacy_run(tmp_path / "legacy")
    run = load_run(tmp_path / "legacy")
    ctx = dict(zip(run.events["event_id"], run.events["entry_context"]))
    assert ctx == {"EVT-001": evt.CTX_AT_START, "EVT-002": evt.CTX_CROSSED, "EVT-003": evt.CTX_FIRST_INSIDE}
    assert run.metrics["zone_entries"] == 2 and run.metrics["zone_initial_occupants"] == 1
    assert run.summary["processing"]["analyzed_duration_s"] == pytest.approx(14.0)
    assert run.metrics["zone_occupied_s"] <= run.summary["processing"]["analyzed_duration_s"]
    assert list(run.events["dwell_is_lower_bound"]) == [True, False, True]


def test_finalize_retries_transient_windows_lock(tmp_path, monkeypatch):
    """Antivirus/indexer can hold a new file briefly; the run folder must still be moved into place."""
    from pathlib import Path
    work, final = tmp_path / "run.partial", tmp_path / "run"
    work.mkdir()
    (work / "summary.json").write_text("{}")
    real_rename, calls = Path.rename, {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("[WinError 5] Access is denied")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    pipeline._finalize_run_dir(work, final, delay_s=0)
    assert (final / "summary.json").exists() and not work.exists() and calls["n"] == 3


def test_finalize_falls_back_to_copy(tmp_path, monkeypatch):
    from pathlib import Path
    work, final = tmp_path / "run.partial", tmp_path / "run"
    (work / "evidence").mkdir(parents=True)
    (work / "evidence" / "a.jpg").write_bytes(b"x")

    def always_locked(self, target):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "rename", always_locked)
    pipeline._finalize_run_dir(work, final, attempts=2, delay_s=0)
    assert (final / "evidence" / "a.jpg").read_bytes() == b"x" and not work.exists()


def test_no_people_with_tracking_reports_zero_not_unknown(tmp_path, monkeypatch):
    """Footage without people (e.g. a fire clip) must show 0 unique people, not 'tracking disabled'."""
    video = tmp_path / "empty.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(10):
        w.write(np.full((H, W, 3), 60, np.uint8))
    w.release()
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    run = load_run(pipeline.analyze_video(video, Settings(analysis_fps=FPS, zone_points=ZONE), ScriptedModel([])))
    assert run.tracking_active and run.metrics["unique_people_est"] == 0
