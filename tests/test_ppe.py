"""PPE beta: association rules, status classification, and an end-to-end pipeline run with a scripted
PPE detector (no model download needed)."""
import cv2
import numpy as np
import pytest

import forgesight.pipeline as pipeline
from forgesight import ppe
from forgesight.config import Settings
from forgesight.results import load_run
from test_pipeline_events import FPS, N, W, H, ZONE, ScriptedModel, box, script

PERSON = np.array([100, 100, 140, 220])  # 40×120 px person


def test_hardhat_must_be_in_head_region_and_on_this_person():
    head = np.array([[108, 95, 132, 112]])
    assert ppe.associate(PERSON, head, np.array([0.8]), ppe.ITEMS["hardhat"].region) == 0
    feet = np.array([[108, 200, 132, 215]])  # right size, wrong place
    assert ppe.associate(PERSON, feet, np.array([0.8]), ppe.ITEMS["hardhat"].region) == -1
    neighbour = np.array([[150, 95, 174, 112]])  # next person's hat
    assert ppe.associate(PERSON, neighbour, np.array([0.8]), ppe.ITEMS["hardhat"].region) == -1


def test_vest_on_torso_picks_most_confident():
    vests = np.array([[104, 130, 136, 175], [105, 132, 135, 170]])
    assert ppe.associate(PERSON, vests, np.array([0.6, 0.9]), ppe.ITEMS["vest"].region) == 1


def test_small_or_cut_off_people_are_not_judged():
    assert not ppe.evaluable([10, 10, 20, 20], 720, "vest")  # too small
    assert not ppe.evaluable([10, 0, 60, 300], 720, "hardhat")  # head outside the frame
    assert ppe.evaluable([10, 0, 60, 300], 720, "vest")


@pytest.mark.parametrize("evaluated,hits,expected", [
    (2, 0, ppe.UNCERTAIN), (2, 2, ppe.UNCERTAIN),  # too few checks either way
    (3, 0, ppe.NOT_DETECTED), (10, 1, ppe.NOT_DETECTED),
    (10, 5, ppe.DETECTED), (3, 2, ppe.DETECTED),
    (10, 3, ppe.UNCERTAIN),  # mixed evidence
])
def test_status_rules(evaluated, hits, expected):
    assert ppe.classify(evaluated, hits) == expected


class ScriptedPpe:
    """Returns PPE boxes for the persons scripted in test_pipeline_events, keyed by the current frame."""

    def __init__(self, person_model):
        self.pm = person_model

    def detect(self, frame, conf):
        f = self.pm.i - 1  # frame index the person model just served
        hats, vests = [], []
        ax = 250 if f < 15 else 250 - (f - 14) * 4
        hats.append([ax - 8, 200 - 62, ax + 8, 200 - 50])  # A: always wears a hard hat, never a vest
        bx = 40 + 4 * f
        vests.append([bx - 9, 120 - 45, bx + 9, 120 - 25])  # B: always a vest...
        if (f // 5) % 3 == 0:
            hats.append([bx - 8, 120 - 62, bx + 8, 120 - 50])  # ...hard hat in only 1 of 3 checks
        out = {}
        for key, boxes in (("hardhat", hats), ("vest", vests)):
            b = np.array(boxes, float).reshape(-1, 4)
            out[key] = (b, np.full(len(b), 0.8))
        return out


@pytest.fixture(scope="module")
def ppe_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ppe")
    video = tmp / "synthetic.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        w.write(np.full((H, W, 3), 60, np.uint8))
    w.release()
    (tmp / "runs").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "RUNS_DIR", tmp / "runs")
    settings = Settings(analysis_fps=FPS, zone_points=ZONE, ppe_enabled=True, ppe_rate_fps=2.0)
    person_model = ScriptedModel(script())
    run_dir = pipeline.analyze_video(video, settings, person_model, ppe_model=ScriptedPpe(person_model))
    mp.undo()
    return load_run(run_dir)


def _status(run, track, item):
    row = run.ppe[(run.ppe["track_id"] == track) & (run.ppe["item"] == item)]
    return None if row.empty else row.iloc[0]


def test_ppe_statuses_end_to_end(ppe_run):
    run = ppe_run
    assert run.ppe_active
    tid = {ctx: t for ctx, t in zip(run.events["entry_context"], run.events["track_id"])}
    a, b = tid["present at start of footage"], tid["crossed into zone"]
    assert _status(run, a, "hardhat").status == ppe.DETECTED
    assert _status(run, a, "vest").status == ppe.NOT_DETECTED
    assert _status(run, b, "vest").status == ppe.DETECTED
    hb = _status(run, b, "hardhat")
    assert hb.status == ppe.UNCERTAIN and 0.1 < hb.hit_ratio < 0.5
    # E's head is cut off by the top of the frame: hard hat is never judged, vest is
    e_rows = run.ppe[~run.ppe["track_id"].isin(list(tid.values()))]
    assert "hardhat" not in set(e_rows["item"])


def test_only_not_detected_rows_become_events_with_evidence(ppe_run):
    run = ppe_run
    missing = run.ppe[run.ppe["status"] == ppe.NOT_DETECTED].sort_values("event_id")
    assert list(missing["event_id"]) == [f"PPE-{i + 1:03d}" for i in range(len(missing))]
    for r in missing.itertuples():
        assert cv2.imread(str(run.path(r.evidence_image))) is not None
    others = run.ppe[run.ppe["status"] != ppe.NOT_DETECTED]
    assert (others["event_id"].fillna("") == "").all()
    assert others["evidence_image"].fillna("").eq("").all()
    # provisional evidence of tallies that recovered must not be left on disk
    on_disk = {p.name for p in (run.run_dir / "evidence").glob("PPE_*") if not p.name.endswith("_crop.jpg")}
    assert on_disk == {p.split("/")[-1] for p in missing["evidence_image"]}


def test_ppe_summary_counts_match_table(ppe_run):
    counts = ppe_run.summary["ppe"]["counts"]
    for item, c in counts.items():
        rows = ppe_run.ppe[ppe_run.ppe["item"] == item]
        assert sum(c.values()) == len(rows)
        assert c["not detected"] == int((rows["status"] == ppe.NOT_DETECTED).sum())


def test_ppe_requires_tracking(tmp_path):
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(10):
        w.write(np.full((H, W, 3), 60, np.uint8))
    w.release()
    (tmp_path / "runs").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "RUNS_DIR", tmp_path / "runs")
    pm = ScriptedModel(script())
    run = load_run(pipeline.analyze_video(video, Settings(analysis_fps=FPS, zone_points=ZONE, ppe_enabled=True,
                                                          tracking_enabled=False), pm, ppe_model=ScriptedPpe(pm)))
    mp.undo()
    assert not run.ppe_active and "tracking" in run.summary["ppe"]["note"]
    assert run.ppe.empty
