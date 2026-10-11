import cv2
import numpy as np
import pandas as pd
import pytest

from forgesight.config import Settings
from forgesight.results import choose_bin_seconds, interval_activity
from forgesight.video_io import (
    AnnotatedVideoWriter,
    VideoError,
    plan_sampling,
    probe_video,
    processing_size,
)


def _write_video(path, frames=30, fps=10, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.rectangle(img, (10 + i * 5, 50), (60 + i * 5, 150), (200, 200, 200), -1)
        w.write(img)
    w.release()


def test_probe_valid_video(tmp_path):
    p = tmp_path / "ok.mp4"
    _write_video(p)
    info = probe_video(p)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10, abs=0.5)
    assert info.duration_s == pytest.approx(3.0, abs=0.2)


def test_probe_rejects_non_video(tmp_path):
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoError):
        probe_video(p)


def test_probe_rejects_empty_and_missing(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoError):
        probe_video(empty)
    with pytest.raises(VideoError):
        probe_video(tmp_path / "missing.mp4")


def test_probe_rejects_truncated_video(tmp_path):
    p = tmp_path / "ok.mp4"
    _write_video(p, frames=60)
    bad = tmp_path / "truncated.mp4"
    bad.write_bytes(p.read_bytes()[:300])
    with pytest.raises(VideoError):
        probe_video(bad)


def test_sampling_plan(tmp_path):
    p = tmp_path / "ok.mp4"
    _write_video(p, frames=50, fps=25)
    info = probe_video(p)
    plan = plan_sampling(info, analysis_fps=5)
    assert plan.step == 5 and plan.frames_to_analyze == 10
    limited = plan_sampling(info, analysis_fps=5, max_seconds=1)
    assert limited.frames_to_analyze == 5


def test_processing_size_even_and_capped():
    assert processing_size(1920, 1080, 1280) == (1280, 720)
    w, h = processing_size(1001, 563, 0)
    assert w % 2 == 0 and h % 2 == 0


def test_h264_writer_produces_readable_file(tmp_path):
    out = tmp_path / "out.mp4"
    writer = AnnotatedVideoWriter(out, (320, 240), 10)
    for _ in range(12):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.close()
    assert writer.browser_playable
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    cap.release()


def test_interval_activity_and_bins():
    frames = pd.DataFrame({"time_s": [0, 0.5, 1.0, 2.5, 4.9], "people": [1, 3, 2, 0, 4]})
    events = pd.DataFrame({"entry_time_s": [0.5, 4.0, 4.5]})
    g = interval_activity(frames, events, 2)
    assert g["detections"].tolist() == [6, 0, 4]
    assert g["peak_occupancy"].tolist() == [3, 0, 4]
    assert g["zone_entries"].tolist() == [1, 0, 2]
    assert choose_bin_seconds(14) == 1 and choose_bin_seconds(300) == 15


def test_settings_roundtrip_ignores_unknown_keys():
    s = Settings.from_dict({"confidence": 0.5, "bogus": 1, "model_name": "yolo11n.pt"})
    assert s.confidence == 0.5
    assert s.model_name == "yolo11n.onnx"  # unknown/legacy model falls back to default
