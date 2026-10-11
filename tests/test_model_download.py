"""Person-model download: the file is only kept when its SHA-256 matches the pinned digest."""
import hashlib

import pytest

from forgesight import detector


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "MODELS_DIR", tmp_path)
    return tmp_path


def fake_download(content: bytes):
    def urlretrieve(url, dest):
        open(dest, "wb").write(content)
    return urlretrieve


def test_tampered_download_is_rejected_and_removed(models_dir, monkeypatch):
    monkeypatch.setattr(detector.urllib.request, "urlretrieve", fake_download(b"not the real model" * 1000))
    with pytest.raises(detector.ModelError, match="integrity"):
        detector.ensure_model("yolo11n.onnx")
    assert not list(models_dir.iterdir())  # neither the model nor a partial download is left behind


def test_matching_download_is_kept(models_dir, monkeypatch):
    content = b"x" * 2_000_000
    monkeypatch.setitem(detector.MODEL_SHA256, "yolo11n.onnx", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(detector.urllib.request, "urlretrieve", fake_download(content))
    path = detector.ensure_model("yolo11n.onnx")
    assert path.read_bytes() == content and path.name == "yolo11n.onnx"


def test_every_downloadable_model_has_a_pinned_digest():
    assert set(detector.MODEL_URLS) == set(detector.MODEL_SHA256)
    assert all(len(h) == 64 for h in detector.MODEL_SHA256.values())
