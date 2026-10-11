import cv2
import numpy as np
import pandas as pd
import pytest

from forgesight import analytics


def frames_from(counts, dt=0.2, zone=None):
    n = len(counts)
    return pd.DataFrame({"time_s": np.round(np.arange(n) * dt, 3), "dt_s": dt, "people": counts,
                         "people_in_zone": zone if zone is not None else counts})


def test_brief_dip_is_bridged():
    f = frames_from([0, 3, 3, 3, 2, 3, 3, 3, 3, 0, 0])
    a = analytics.sustained_occupancy(f, "frame", 3, 1.0, gap_tolerance_s=0.5)
    assert len(a) == 1
    r = a.iloc[0]
    assert (r.start_s, r.end_s, r.duration_s, r.peak) == (0.2, 1.8, 1.6, 3)
    assert r.alert_id == "OCC-001"


def test_without_gap_tolerance_period_splits_and_is_too_short():
    f = frames_from([0, 3, 3, 3, 2, 3, 3, 3, 3, 0, 0])
    assert analytics.sustained_occupancy(f, "frame", 3, 1.0, gap_tolerance_s=0.0).empty
    assert len(analytics.sustained_occupancy(f, "frame", 3, 0.6, gap_tolerance_s=0.0)) == 2


def test_long_dip_ends_alert_and_min_duration_is_inclusive():
    f = frames_from([4] * 5 + [0] * 5 + [4] * 5)
    a = analytics.sustained_occupancy(f, "frame", 4, 1.0)
    assert list(a["duration_s"]) == [1.0, 1.0]
    assert list(a["alert_id"]) == ["OCC-001", "OCC-002"]


def test_scope_selects_zone_counts():
    f = frames_from([5] * 20, zone=[0] * 10 + [2] * 10)
    assert analytics.sustained_occupancy(f, "zone", 2, 1.0).iloc[0].start_s == 2.0
    assert analytics.sustained_occupancy(f, "frame", 2, 1.0).iloc[0].start_s == 0.0


def test_trend_is_rolling_mean():
    f = frames_from([0, 0, 6, 0, 0], dt=1.0)
    t = analytics.trend(f, "frame", 3.0)
    assert t.tolist() == [0.0, 2.0, 2.0, 2.0, 0.0]


def test_annotated_frame_lookup_and_alert_evidence(tmp_path):
    video = tmp_path / "annotated.mp4"
    from forgesight.video_io import AnnotatedVideoWriter
    w = AnnotatedVideoWriter(video, (160, 120), 5)
    for i in range(12):
        w.write(np.full((120, 160, 3), 20 * i, np.uint8))
    w.close()
    assert analytics.read_annotated_frame(video, 7).mean() == pytest.approx(140, abs=6)
    assert analytics.read_annotated_frame(video, 99) is None
    alert = {"scope": "zone", "threshold": 2, "start_s": 1.0, "end_s": 2.0, "duration_s": 1.0, "peak": 3,
             "peak_sample": 7}
    rel = analytics.occupancy_alert_evidence(tmp_path, alert, "Cell")
    img = cv2.imread(str(tmp_path / rel))
    assert img is not None
    assert analytics.occupancy_alert_evidence(tmp_path, alert, "Cell") == rel  # cached


def test_heatmap_zone_share_and_hotspot():
    heat = np.zeros((10, 20))
    heat[5, 15] = 3  # right half
    heat[5, 2] = 1  # left half
    s = analytics.heatmap_stats(heat, [[50, 0], [100, 0], [100, 100], [50, 100]])
    assert s["zone_share"] == pytest.approx(0.75)
    assert s["hotspot_pct"] == pytest.approx((77.5, 55.0))
    assert analytics.heatmap_stats(np.zeros((4, 4)), None)["zone_share"] is None
