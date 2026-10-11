"""Optional PPE (hard hat / hi-vis vest) detection — BETA.

People come from the YOLO11 person tracks; PPE items come only from a dedicated PPE detector
(YOLOv8n fine-tuned on the Roboflow "Construction Site Safety" dataset, ONNX). An item is linked to a
person when its box sits in the expected body region (hard hat: head, vest: torso).

Per tracked person and item, single-frame results are aggregated over several PPE checks:
  DETECTED      – item linked in ≥ DETECT_RATIO of ≥ MIN_EVALS checks
  NOT_DETECTED  – person clearly visible in ≥ MIN_EVALS checks and item linked in ≤ MISSING_RATIO of them.
                  Reported as "not detected — verify", never as a confirmed violation.
  UNCERTAIN     – too few usable checks, or mixed results.
The model's own "NO-Hardhat" class is deliberately ignored: on held-out data its precision was 0.20.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

log = logging.getLogger(__name__)

PPE_MODEL_FILE = "ppe_construction_yolov8n.onnx"
PPE_MODEL_REPO = "mancalazure/construction-ppe-safety"
PPE_MODEL_REVISION = "70cc801710f246765f70ada7248d9560ec749bb2"
PPE_MODEL_URL = f"https://huggingface.co/{PPE_MODEL_REPO}/resolve/{PPE_MODEL_REVISION}/best.onnx"
PPE_MODEL_SHA256 = "1d5f3cf69883e89ba6e2b3836a0b21623044e9a682f1f00dfbacac97b99baf94"
PPE_MODEL_NOTE = ("YOLOv8n fine-tuned on Roboflow 'Construction Site Safety' v28 (CC BY 4.0 data); "
                  "weights exported with Ultralytics (AGPL-3.0)")

DETECTED, NOT_DETECTED, UNCERTAIN = "detected", "not detected", "uncertain"
STATUS_LABELS = {DETECTED: "Detected", NOT_DETECTED: "Not detected (verify)", UNCERTAIN: "Uncertain"}

MIN_EVALS = 3
DETECT_RATIO = 0.5
MISSING_RATIO = 0.1
MIN_PERSON_HEIGHT_FRAC = 0.08  # person box must be ≥ 8% of frame height to be judged at all
NMS_IOU = 0.5


@dataclass(frozen=True)
class Item:
    key: str
    label: str
    class_name: str  # name in the model's metadata
    region: tuple[float, float]  # expected vertical position of the item's centre, as fraction of person height


ITEMS = {
    "hardhat": Item("hardhat", "Hard hat", "Hardhat", (0.0, 0.30)),
    "vest": Item("vest", "Hi-vis vest", "Safety Vest", (0.20, 0.75)),
}

# Measured by scripts/validate_ppe.py (single images, IoU ≥ 0.5, confidence ≥ 0.40) on the 141-image test split
# of the independent Ultralytics Construction-PPE dataset. Not measured on automotive footage.
VALIDATION = {
    "dataset": "Ultralytics Construction-PPE, test split (141 images)",
    "hardhat": {"precision": 0.83, "recall": 0.72},
    "vest": {"precision": 0.86, "recall": 0.60},
    "no_hardhat_class": {"precision": 0.20, "recall": 0.23, "used": False},
    "negative_clip": "people-walking sample (no PPE worn): hard-hat false detections 27 at conf≥0.40, "
                     "4 at conf≥0.55, over ~560 person-frames; vest false detections ≤1",
    # Person-level, full pipeline (YOLO11 + ByteTrack + PPE voting), 133 test images as a slideshow video.
    "person_level": {
        "hardhat": {"detected_precision": 115 / 117, "not_detected_precision": 32 / 50,
                    "wearing_judged_detected": 115 / 133, "no_result_share": 32 / 199},
        "vest": {"detected_precision": 108 / 114, "not_detected_precision": 46 / 69,
                 "wearing_judged_detected": 108 / 132, "no_result_share": 52 / 236},
    },
}


class PpeModelError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_ppe_model() -> Path:
    """Local path of the PPE model; downloads the pinned revision and verifies its SHA-256."""
    path = MODELS_DIR / PPE_MODEL_FILE
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(PPE_MODEL_URL, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise PpeModelError(f"Could not download the PPE model ({exc}). Download {PPE_MODEL_URL} manually and "
                            f"save it as models/{PPE_MODEL_FILE}.") from exc
    if _sha256(tmp) != PPE_MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise PpeModelError("Downloaded PPE model failed its integrity check (SHA-256 mismatch).")
    tmp.replace(path)
    return path


class PpeModel:
    """YOLOv8-format ONNX detector with a fixed 640×640 input; returns boxes for the PPE items only."""

    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta["names"])
        except (KeyError, ValueError, SyntaxError) as exc:
            raise PpeModelError("PPE model has no class-name metadata.") from exc
        by_name = {v: int(k) for k, v in names.items()}
        missing = [i.class_name for i in ITEMS.values() if i.class_name not in by_name]
        if missing:  # the class order is never assumed: a wrong model fails loudly instead of mislabelling
            raise PpeModelError(f"PPE model lacks required classes: {missing}")
        self.class_ids = {k: by_name[i.class_name] for k, i in ITEMS.items()}
        self.num_classes = len(names)
        self.name = Path(path).stem

    def detect(self, frame: np.ndarray, conf: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        img = cv2.copyMakeBorder(cv2.resize(frame, (nw, nh)), top, self.size - nh - top, left,
                                 self.size - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0][0]  # (4 + classes, anchors)
        boxes = out[:4].T
        result = {}
        for key, cid in self.class_ids.items():
            sc = out[4 + cid]
            k = sc >= conf
            if not np.any(k):
                result[key] = (np.zeros((0, 4)), np.zeros(0))
                continue
            b, s = boxes[k], sc[k]
            xywh = np.c_[(b[:, 0] - b[:, 2] / 2 - left) / r, (b[:, 1] - b[:, 3] / 2 - top) / r, b[:, 2] / r, b[:, 3] / r]
            idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), conf, NMS_IOU), dtype=int).reshape(-1)
            xyxy = np.c_[xywh[idx, :2], xywh[idx, :2] + xywh[idx, 2:]]
            result[key] = (xyxy, s[idx].astype(float))
        return result


def load_ppe_model() -> PpeModel:
    return PpeModel(ensure_ppe_model())


def associate(person_xyxy, boxes: np.ndarray, confs: np.ndarray, region: tuple[float, float]) -> int:
    """Index of the best item box belonging to this person, or -1.

    The item's centre must lie within the person's box horizontally and in the expected vertical band,
    and at least half of the item box must overlap the person box (rejects a neighbour's equipment).
    """
    x1, y1, x2, y2 = person_xyxy
    ph = max(y2 - y1, 1e-6)
    best, best_conf = -1, -1.0
    for i, (bx1, by1, bx2, by2) in enumerate(boxes):
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (x1 <= cx <= x2 and y1 + region[0] * ph <= cy <= y1 + region[1] * ph):
            continue
        area = max((bx2 - bx1) * (by2 - by1), 1e-6)
        inter = max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1))
        if inter / area >= 0.5 and confs[i] > best_conf:
            best, best_conf = i, float(confs[i])
    return best


def evaluable(person_xyxy, frame_h: int, item: str) -> bool:
    """Is the person large and complete enough for this item to be judged?"""
    x1, y1, x2, y2 = person_xyxy
    if (y2 - y1) < MIN_PERSON_HEIGHT_FRAC * frame_h:
        return False
    if item == "hardhat" and y1 <= 2:  # head cut off by the top of the frame
        return False
    return True


@dataclass
class Tally:
    evaluated: int = 0
    hits: int = 0
    max_conf: float = 0.0
    first_eval_t: float | None = None
    last_eval_t: float | None = None
    box_sum: np.ndarray = field(default_factory=lambda: np.zeros(4))
    evidence_image: str = ""
    evidence_crop: str = ""
    evidence_t: float | None = None

    @property
    def ratio(self) -> float:
        return self.hits / self.evaluated if self.evaluated else 0.0

    @property
    def status(self) -> str:
        return classify(self.evaluated, self.hits)


def classify(evaluated: int, hits: int) -> str:
    if evaluated < MIN_EVALS:
        return UNCERTAIN
    ratio = hits / evaluated
    if ratio >= DETECT_RATIO:
        return DETECTED
    if ratio <= MISSING_RATIO:
        return NOT_DETECTED
    return UNCERTAIN


class PpeMonitor:
    """Accumulates per-track, per-item PPE evidence across PPE-check frames."""

    def __init__(self, items: list[str]):
        self.items = [i for i in items if i in ITEMS]
        self.tallies: dict[tuple[int, str], Tally] = {}

    def update(self, t: float, frame_h: int, persons: list[tuple[int, np.ndarray]],
               detections: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[list[tuple[int, str]], dict]:
        """Feed one PPE-check frame. Returns (tallies that just became NOT_DETECTED, linked item boxes)."""
        newly_missing, linked = [], {}
        for tid, xyxy in persons:
            for item in self.items:
                if not evaluable(xyxy, frame_h, item):
                    continue
                tl = self.tallies.setdefault((tid, item), Tally())
                before = tl.status
                boxes, confs = detections.get(item, (np.zeros((0, 4)), np.zeros(0)))
                j = associate(xyxy, boxes, confs, ITEMS[item].region)
                tl.evaluated += 1
                tl.box_sum += np.asarray(xyxy, dtype=float)
                tl.first_eval_t = t if tl.first_eval_t is None else tl.first_eval_t
                tl.last_eval_t = t
                if j >= 0:
                    tl.hits += 1
                    tl.max_conf = max(tl.max_conf, float(confs[j]))
                    linked[(tid, item)] = (boxes[j], float(confs[j]))
                if tl.status == NOT_DETECTED and before != NOT_DETECTED:
                    newly_missing.append((tid, item))
        return newly_missing, linked

    def status(self, tid: int, item: str) -> str | None:
        tl = self.tallies.get((tid, item))
        return tl.status if tl else None
