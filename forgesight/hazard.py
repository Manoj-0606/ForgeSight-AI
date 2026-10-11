"""Optional fire / smoke detection — BETA.

A dedicated fire/smoke image classifier (ViT-B/16, ONNX int8, Apache-2.0) scores whole frames a few times per
second; nothing is inferred from the person model. A class is *confirmed* only after ``confirm_checks``
consecutive positive checks; one confirmed episode produces one event and stays open until ``clear_checks``
consecutive negative checks, so a fire that persists for minutes is not reported again on every frame.

Model choice (see docs/FIRE_SMOKE_VALIDATION.md): a YOLOv8n fire/smoke *detector* was evaluated first but missed
a real house fire and dense indoor smoke; this classifier detected the fire with no false alarms on the
automotive-factory clips. It gives no boxes, so evidence is the full frame with the scores.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

HAZARD_MODEL_FILE = "fire_smoke_vit_int8.onnx"
HAZARD_MODEL_REPO = "prithivMLmods/Fire-Detection-Engine-ONNX"
HAZARD_MODEL_REVISION = "02bd7f981aac3e27a75f83e0a3b97dfadaffc228"
HAZARD_MODEL_URL = (f"https://huggingface.co/{HAZARD_MODEL_REPO}/resolve/{HAZARD_MODEL_REVISION}"
                    "/onnx/model_int8.onnx")
HAZARD_MODEL_SHA256 = "4917442f15fd7014ccf9cf9509b7dd6692db2dcc3aace56f03ae18808ced0785"
HAZARD_MODEL_NOTE = ("ViT-B/16 image classifier 'Fire-Detection-Engine' (int8 ONNX, Apache-2.0); classes "
                     "Fire Needed Action / Normal Conditions / Smoky Environment; frame-level, no boxes")
# Output order from the model's config.json at the pinned revision (id2label).
OUTPUT_CLASSES = ("fire", "normal", "smoke")
INPUT_SIZE = 224

CLASSES = {"fire": "Fire", "smoke": "Smoke"}


class HazardModelError(Exception):
    pass


def ensure_hazard_model() -> Path:
    path = MODELS_DIR / HAZARD_MODEL_FILE
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(HAZARD_MODEL_URL, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HazardModelError(f"Could not download the fire/smoke model ({exc}). Download {HAZARD_MODEL_URL} "
                               f"manually and save it as models/{HAZARD_MODEL_FILE}.") from exc
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != HAZARD_MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise HazardModelError("Downloaded fire/smoke model failed its integrity check (SHA-256 mismatch).")
    tmp.replace(path)
    return path


class HazardModel:
    """ViT fire/smoke classifier. ``scores(frame)`` returns class probabilities for the whole frame."""

    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) != 2 or out_shape[-1] != len(OUTPUT_CLASSES):
            raise HazardModelError(f"Unexpected fire/smoke model output shape {out_shape}; expected [batch, 3].")
        self.name = Path(path).stem
        self.version = HAZARD_MODEL_REVISION[:12]

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        """Softmax probabilities {'fire', 'normal', 'smoke'} (ViT preprocessing: 224², RGB, mean/std 0.5)."""
        x = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        z = self.session.run(None, {self.input_name: x})[0][0].astype(np.float64)
        e = np.exp(z - z.max())
        p = e / e.sum()
        return {c: float(v) for c, v in zip(OUTPUT_CLASSES, p)}


def load_hazard_model() -> HazardModel:
    return HazardModel(ensure_hazard_model())


@dataclass
class Episode:
    cls: str
    start_t: float  # first positive check of the confirming streak
    confirmed_t: float
    max_conf: float
    conf_sum: float = 0.0
    positive_checks: int = 0
    end_t: float | None = None  # last positive check, set when the episode closes
    last_positive_t: float = 0.0

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(self.positive_checks, 1)


@dataclass
class _ClassState:
    streak: int = 0
    streak_start: float = 0.0
    streak_conf: list = field(default_factory=list)
    negatives: int = 0
    episode: Episode | None = None


class HazardMonitor:
    """Consecutive-check confirmation and episode tracking per hazard class."""

    def __init__(self, confirm_checks: int = 3, clear_checks: int = 5, classes=tuple(CLASSES)):
        self.confirm_checks = max(1, int(confirm_checks))
        self.clear_checks = max(1, int(clear_checks))
        self.episodes: list[Episode] = []
        self._state = {c: _ClassState() for c in CLASSES if c in classes}

    def update(self, t: float, best_conf: dict[str, float]) -> list[Episode]:
        """Feed one check: the best confidence per class (0 when nothing above threshold).

        Returns episodes confirmed on this check.
        """
        new = []
        for cls, st in self._state.items():
            c = float(best_conf.get(cls, 0.0))
            if c > 0:
                st.negatives = 0
                if st.episode is not None:  # ongoing episode: extend it, no new event
                    ep = st.episode
                    ep.positive_checks += 1
                    ep.conf_sum += c
                    ep.max_conf = max(ep.max_conf, c)
                    ep.last_positive_t = t
                    continue
                if st.streak == 0:
                    st.streak_start = t
                    st.streak_conf = []
                st.streak += 1
                st.streak_conf.append(c)
                if st.streak >= self.confirm_checks:
                    ep = Episode(cls, st.streak_start, t, max(st.streak_conf), sum(st.streak_conf),
                                 len(st.streak_conf), last_positive_t=t)
                    st.episode, st.streak = ep, 0
                    self.episodes.append(ep)
                    new.append(ep)
            else:
                st.streak = 0
                if st.episode is not None:
                    st.negatives += 1
                    if st.negatives >= self.clear_checks:
                        st.episode.end_t = st.episode.last_positive_t
                        st.episode, st.negatives = None, 0
        return new

    def active(self, cls: str) -> bool:
        st = self._state.get(cls)
        return st is not None and st.episode is not None

    def finish(self) -> None:
        for st in self._state.values():
            if st.episode is not None:
                st.episode.end_t = st.episode.last_positive_t
                st.episode = None
