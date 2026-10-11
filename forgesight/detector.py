"""YOLO11 person detection (ONNX Runtime, CPU) and ByteTrack tracking.

The official Ultralytics YOLO11 weights are used in their ONNX export, executed with ONNX Runtime.
This avoids a PyTorch runtime dependency: it keeps the install small, is fast on CPU, and works on
Windows machines where Smart App Control blocks PyTorch's unsigned DLLs.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

log = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO "person"
TRACKER_LOW_CONF = 0.10  # ByteTrack uses low-confidence boxes to keep existing tracks alive
NMS_IOU = 0.5
MODEL_URLS = {
    "yolo11n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx",
    "yolo11s.onnx": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.onnx",
}
# SHA-256 digests published by GitHub for the v8.4.0 release assets.
MODEL_SHA256 = {
    "yolo11n.onnx": "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
    "yolo11s.onnx": "21d6650c5097610c92c76ce5e4b717976059169eaea4962035b90c6a92c07a8f",
}


class ModelError(Exception):
    pass


@dataclass
class FrameDetections:
    xyxy: np.ndarray  # (N, 4) float pixel boxes
    confidence: np.ndarray  # (N,)
    track_ids: np.ndarray | None  # (N,) int, or None when tracking is disabled

    def __len__(self) -> int:
        return len(self.xyxy)


def model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name


def ensure_model(model_name: str) -> Path:
    """Return the local model path, downloading the official Ultralytics export if missing."""
    path = model_path(model_name)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    url = MODEL_URLS.get(model_name)
    if url is None:
        raise ModelError(f"Unknown model '{model_name}'. Place the ONNX file in {MODELS_DIR}.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    try:
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"Could not download {model_name} ({exc}). Check your internet connection, or download "
            f"{url} manually into the 'models' folder."
        ) from exc
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != MODEL_SHA256[model_name]:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"Downloaded {model_name} failed its integrity check (SHA-256 mismatch).")
    tmp.replace(path)
    return path


class YoloOnnx:
    """Minimal YOLO11 detector: letterbox → ONNX Runtime → person filter → NMS."""

    def __init__(self, path: str | Path, imgsz: int = 640, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.name = Path(path).stem
        meta = self.session.get_modelmeta().custom_metadata_map
        self.version = meta.get("version", "unknown")
        if meta.get("task", "detect") != "detect":
            raise ModelError(f"{path} is not a detection model.")

    def _letterbox(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        # minimal padding to a multiple of the model stride (the export has dynamic input size)
        pw, ph = (32 - nw % 32) % 32, (32 - nh % 32) % 32
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame
        left, top = pw // 2, ph // 2
        img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, r, left, top

    def detect_people(self, frame: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray]:
        blob, r, left, top = self._letterbox(frame)
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (84, N)
        scores = out[4 + PERSON_CLASS_ID]
        keep = scores >= conf
        if not np.any(keep):
            return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=float)
        cx, cy, bw, bh = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
        scores = scores[keep]
        x1 = (cx - bw / 2 - left) / r
        y1 = (cy - bh / 2 - top) / r
        boxes_xywh = np.stack([x1, y1, bw / r, bh / r], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf, NMS_IOU)
        idx = np.array(idx, dtype=int).reshape(-1)
        h, w = frame.shape[:2]
        xyxy = np.stack([boxes_xywh[idx, 0], boxes_xywh[idx, 1],
                         boxes_xywh[idx, 0] + boxes_xywh[idx, 2], boxes_xywh[idx, 1] + boxes_xywh[idx, 3]], axis=1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h - 1)
        return xyxy.astype(float), scores[idx].astype(float)


def load_model(model_name: str) -> YoloOnnx:
    model = YoloOnnx(ensure_model(model_name))
    model.detect_people(np.zeros((360, 640, 3), dtype=np.uint8), 0.5)  # warm-up
    return model


class PersonDetector:
    """Runs YOLO on a frame and optionally associates boxes across frames with ByteTrack."""

    def __init__(self, model: YoloOnnx, confidence: float, tracking: bool, fps: float):
        self.model = model
        self.confidence = float(confidence)
        self.tracker = None
        self.tracking_error: str | None = None
        if tracking:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    import supervision as sv

                    # supervision scales lost_track_buffer by frame_rate/30; 60 → ~2 s at any rate.
                    self.tracker = sv.ByteTrack(
                        track_activation_threshold=self.confidence,
                        lost_track_buffer=60,
                        minimum_matching_threshold=0.8,
                        frame_rate=max(1, int(round(fps))),
                    )
                self._sv = sv
            except Exception as exc:  # pragma: no cover - only when supervision is missing/broken
                self.tracking_error = f"ByteTrack unavailable ({exc}); running detection only."
                log.warning(self.tracking_error)

    @property
    def tracking_active(self) -> bool:
        return self.tracker is not None

    def process(self, frame: np.ndarray) -> FrameDetections:
        if self.tracker is None:
            xyxy, conf = self.model.detect_people(frame, self.confidence)
            return FrameDetections(xyxy, conf, None)

        xyxy, conf = self.model.detect_people(frame, min(self.confidence, TRACKER_LOW_CONF))
        dets = self._sv.Detections(xyxy=xyxy.reshape(-1, 4), confidence=conf,
                                   class_id=np.zeros(len(conf), dtype=int))
        tracked = self.tracker.update_with_detections(dets)
        ids = tracked.tracker_id if tracked.tracker_id is not None else np.zeros(0, dtype=int)
        conf_arr = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
        return FrameDetections(np.asarray(tracked.xyxy, dtype=float).reshape(-1, 4),
                               np.asarray(conf_arr, dtype=float), np.asarray(ids, dtype=int))
