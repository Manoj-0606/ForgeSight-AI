"""Video validation, sampled frame reading and browser-playable video writing."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoError(Exception):
    """Raised with a user-facing message when a video cannot be used."""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int  # may be 0 when the container does not report it
    duration_s: float
    codec: str
    fps_estimated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return chars.strip("\x00 ").upper() or "unknown"


def probe_video(path: str | Path) -> VideoInfo:
    """Open the video, read its metadata and one frame. Raises VideoError if unusable."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise VideoError("The video file is empty or missing.")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(
                "OpenCV could not open this video. It may be corrupted or use an unsupported "
                "codec — try re-exporting it as H.264 MP4."
            )
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("The video opened but no frames could be decoded. The file may be corrupted.")

        height, width = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_estimated = False
        if not (1.0 <= fps <= 240.0):
            fps, fps_estimated = 25.0, True
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = max(frame_count, 0)
        duration = frame_count / fps if frame_count else 0.0
        codec = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()

    if width < 64 or height < 64:
        raise VideoError(f"Video resolution {width}×{height} is too small for person detection.")

    return VideoInfo(str(path), width, height, fps, frame_count, duration, codec, fps_estimated)


def read_first_frame(path: str | Path, at_seconds: float = 0.0) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        if at_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def processing_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Output size capped at max_width, preserving aspect ratio, with even dimensions (H.264)."""
    scale = min(1.0, max_width / width) if max_width else 1.0
    w = int(round(width * scale)) // 2 * 2
    h = int(round(height * scale)) // 2 * 2
    return max(w, 2), max(h, 2)


@dataclass
class SamplingPlan:
    step: int  # analyze every Nth source frame
    effective_fps: float
    frames_to_analyze: int  # 0 when unknown
    max_frame_index: int  # exclusive; 0 = no limit


def plan_sampling(info: VideoInfo, analysis_fps: float, max_seconds: float = 0) -> SamplingPlan:
    step = max(1, int(round(info.fps / max(analysis_fps, 0.1))))
    max_frame = int(max_seconds * info.fps) if max_seconds and max_seconds > 0 else 0
    total = info.frame_count
    if max_frame:
        total = min(total, max_frame) if total else max_frame
    frames = (total + step - 1) // step if total else 0
    return SamplingPlan(step, info.fps / step, frames, max_frame)


def iter_sampled_frames(
    path: str | Path, info: VideoInfo, plan: SamplingPlan, size: tuple[int, int]
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (source_frame_index, timestamp_seconds, resized_bgr_frame) for sampled frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Could not reopen the video for analysis.")
    try:
        idx = 0
        while True:
            if plan.max_frame_index and idx >= plan.max_frame_index:
                break
            if not cap.grab():
                break
            if idx % plan.step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    yield idx, idx / info.fps, frame
            idx += 1
    finally:
        cap.release()


class AnnotatedVideoWriter:
    """Writes H.264 MP4 via the ffmpeg binary bundled with imageio-ffmpeg.

    OpenCV's default 'mp4v' codec does not play in browsers, so it is only a fallback.
    """

    def __init__(self, path: str | Path, size: tuple[int, int], fps: float):
        self.path = Path(path)
        self.size = size
        self.fps = fps
        self.browser_playable = False
        self._gen = None
        self._cv_writer = None
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(
                str(self.path),
                size,
                fps=fps,
                codec="libx264",
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                quality=None,
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=["-preset", "veryfast", "-crf", "24", "-movflags", "+faststart"],
            )
            self._gen.send(None)
            self.browser_playable = True
        except Exception as exc:  # pragma: no cover - depends on local ffmpeg availability
            log.warning("ffmpeg writer unavailable (%s); falling back to OpenCV mp4v", exc)
            self._gen = None
            self._cv_writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )
            if not self._cv_writer.isOpened():
                raise VideoError("Could not create the annotated output video.") from exc

    def write(self, frame: np.ndarray) -> None:
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame))
        else:
            self._cv_writer.write(frame)

    def close(self) -> None:
        if self._gen is not None:
            try:
                self._gen.close()
            finally:
                self._gen = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
