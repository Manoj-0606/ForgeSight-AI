"""OpenCV drawing for annotated output frames (BGR colour order)."""
from __future__ import annotations

import cv2
import numpy as np

CYAN = (238, 211, 34)
RED = (56, 68, 240)
AMBER = (36, 165, 245)
AMBER_DARK = (9, 83, 180)  # #B45309, readable under white text
PANEL = (28, 18, 12)
WHITE = (245, 240, 235)
MUTED = (178, 164, 154)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_PLAIN = cv2.FONT_HERSHEY_SIMPLEX


def _scale(frame: np.ndarray) -> float:
    return max(0.55, frame.shape[1] / 1280.0)


def _blend_rect(frame, x1, y1, x2, y2, color, alpha):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _label(frame, text, x, y, color, s, text_color=WHITE, anchor_bottom=True):
    """Filled label pill whose bottom-left corner sits at (x, y)."""
    fs = 0.5 * s
    th = max(1, int(round(1.2 * s)))
    (tw, tht), base = cv2.getTextSize(text, FONT_PLAIN, fs, th)
    pad = int(5 * s)
    h, w = frame.shape[:2]
    x = int(min(max(0, x), w - tw - 2 * pad))
    top = int(y - tht - base - 2 * pad) if anchor_bottom else int(y)
    top = max(0, top)
    cv2.rectangle(frame, (x, top), (x + tw + 2 * pad, top + tht + base + 2 * pad), color, -1)
    cv2.putText(frame, text, (x + pad, top + pad + tht), FONT_PLAIN, fs, text_color, th, cv2.LINE_AA)


def draw_zone(frame: np.ndarray, polygon_px: np.ndarray, name: str, occupied: bool) -> None:
    s = _scale(frame)
    pts = polygon_px.astype(np.int32).reshape(-1, 1, 2)
    color = RED if occupied else AMBER
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.28 if occupied else 0.14, frame, 0.72 if occupied else 0.86, 0, dst=frame)
    cv2.polylines(frame, [pts], True, color, max(2, int(2 * s)), cv2.LINE_AA)
    # label at the top-most vertex
    top = polygon_px[np.argmin(polygon_px[:, 1])]
    status = "OCCUPIED" if occupied else "CLEAR"
    _label(frame, f"RESTRICTED ZONE | {ascii_text(name).upper()} | {status}", int(top[0]), int(top[1]) - int(4 * s),
           color, s, text_color=(20, 20, 20) if not occupied else WHITE)


def draw_person(frame, xyxy, track_id, conf, in_zone: bool, trail: list | None = None) -> None:
    s = _scale(frame)
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    color = RED if in_zone else CYAN
    if trail and len(trail) > 1:
        pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, max(1, int(1.5 * s)), cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(1 * s)), cv2.LINE_AA)
    # corner brackets for a "targeting" look
    L = int(max(8, 0.18 * min(x2 - x1, y2 - y1)))
    t = max(2, int(2.5 * s))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)

    foot = ((x1 + x2) // 2, y2)
    cv2.circle(frame, foot, max(3, int(4 * s)), color, -1, cv2.LINE_AA)
    cv2.circle(frame, foot, max(5, int(7 * s)), WHITE, 1, cv2.LINE_AA)

    text = f"ID {track_id}  {conf:.2f}" if track_id is not None else f"PERSON  {conf:.2f}"
    if in_zone:
        text = "! " + text
    _label(frame, text, x1, y1 - 2, color, s, text_color=(15, 15, 15) if not in_zone else WHITE)


def draw_alert_banner(frame, title: str, subtitle: str, color=RED) -> None:
    s = _scale(frame)
    h, w = frame.shape[:2]
    bh = int(52 * s)
    _blend_rect(frame, 0, 0, w, bh, color, 0.88)
    cv2.rectangle(frame, (0, bh - max(2, int(3 * s))), (w, bh), WHITE, -1)
    fs = 0.8 * s
    th = max(1, int(2 * s))
    (tw, tht), _ = cv2.getTextSize(title, FONT, fs, th)
    x = int(18 * s)
    cv2.putText(frame, ascii_text(title), (x, int(bh / 2 + tht / 2) - 2), FONT, fs, WHITE, th, cv2.LINE_AA)
    cv2.putText(frame, ascii_text(subtitle), (x + tw + int(18 * s), int(bh / 2 + tht / 2) - 2), FONT_PLAIN,
                0.55 * s, WHITE, max(1, int(1.2 * s)), cv2.LINE_AA)


def draw_hud(frame, camera_label: str, t: float, occupancy: int, in_zone: int, events: int,
             tracking: bool, ppe: bool = False) -> None:
    s = _scale(frame)
    h, w = frame.shape[:2]
    lines = [
        ("FORGESIGHT AI", CYAN),
        (ascii_text(camera_label), WHITE),
        (f"T {fmt_ts(t)}   PEOPLE {occupancy}   IN ZONE {in_zone}   ENTRIES {events}", WHITE),
        ("YOLO11 + BYTETRACK" if tracking else "YOLO11 (NO TRACKING)", MUTED),
    ]
    if ppe:
        lines.append(("PPE BETA  H=HARD HAT V=VEST  + DET  - NOT DET  ? UNCERTAIN", MUTED))
    fs = 0.48 * s
    th = max(1, int(1.1 * s))
    lh = int(20 * s)
    width = max(cv2.getTextSize(t_, FONT_PLAIN, fs, th)[0][0] for t_, _ in lines) + int(24 * s)
    x0, y0 = int(12 * s), h - int(12 * s) - lh * len(lines) - int(10 * s)
    _blend_rect(frame, x0, y0, x0 + width, h - int(12 * s), PANEL, 0.72)
    cv2.line(frame, (x0, y0), (x0, h - int(12 * s)), CYAN, max(2, int(3 * s)))
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x0 + int(12 * s), y0 + int(8 * s) + lh * (i + 1) - int(4 * s)),
                    FONT_PLAIN, fs, color, th, cv2.LINE_AA)


GREEN = (106, 183, 18)  # #12B76A


def draw_ppe_box(frame, xyxy, label: str) -> None:
    """A PPE item detected by the PPE model and linked to a person."""
    s = _scale(frame)
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, max(1, int(1.5 * s)), cv2.LINE_AA)
    fs = 0.38 * s
    (tw, th), _ = cv2.getTextSize(label, FONT_PLAIN, fs, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 6), GREEN, -1)
    cv2.putText(frame, label, (x1 + 3, y2 + th + 3), FONT_PLAIN, fs, (15, 15, 15), 1, cv2.LINE_AA)


HAZARD = (0, 84, 230)  # #E65400 red-orange, used only for fire/smoke


def draw_hazard_panel(frame, scores: dict | None, active: dict[str, bool], threshold: float) -> None:
    """Top-right panel with the latest fire/smoke scores; turns red-orange while an episode is confirmed."""
    s = _scale(frame)
    h, w = frame.shape[:2]
    pw, ph = int(250 * s), int(74 * s)
    x0, y0 = w - pw - int(12 * s), int(60 * s)
    any_active = any(active.values())
    _blend_rect(frame, x0, y0, x0 + pw, y0 + ph, HAZARD if any_active else PANEL, 0.82)
    title = ("POTENTIAL " + " + ".join(c.upper() for c, a in active.items() if a)) if any_active else "FIRE/SMOKE BETA"
    fs, th = 0.45 * s, max(1, int(1.1 * s))
    cv2.putText(frame, title, (x0 + int(8 * s), y0 + int(18 * s)), FONT_PLAIN, fs, WHITE, th, cv2.LINE_AA)
    for i, cls in enumerate(active):  # only the monitored classes
        y = y0 + int((34 + 20 * i) * s)
        v = None if scores is None else scores.get(cls)
        cv2.putText(frame, cls.upper(), (x0 + int(8 * s), y + int(4 * s)), FONT_PLAIN, 0.4 * s, WHITE, th, cv2.LINE_AA)
        bx0, bx1 = x0 + int(62 * s), x0 + pw - int(52 * s)
        cv2.rectangle(frame, (bx0, y - int(5 * s)), (bx1, y + int(4 * s)), MUTED, 1)
        if v is not None:
            fill = bx0 + int((bx1 - bx0) * min(1.0, v))
            cv2.rectangle(frame, (bx0, y - int(5 * s)), (fill, y + int(4 * s)), AMBER if v >= threshold else MUTED, -1)
        tx = bx0 + int((bx1 - bx0) * threshold)
        cv2.line(frame, (tx, y - int(8 * s)), (tx, y + int(7 * s)), WHITE, 1)
        cv2.putText(frame, "--" if v is None else f"{v:.2f}", (bx1 + int(6 * s), y + int(4 * s)), FONT_PLAIN,
                    0.4 * s, WHITE, th, cv2.LINE_AA)


PPE_SYMBOL = {"detected": "+", "not detected": "-", "uncertain": "?"}


def draw_ppe_tag(frame, person_xyxy, statuses: dict[str, str | None]) -> None:
    """Running PPE status under a person: H/V = hard hat / vest, + detected, - not detected, ? uncertain."""
    parts = [f"{'H' if k == 'hardhat' else 'V'}{PPE_SYMBOL.get(v, '?')}" for k, v in statuses.items()]
    if not parts:
        return
    vals = [v for v in statuses.values()]
    color = (RED if "not detected" in vals else GREEN if all(v == "detected" for v in vals) else AMBER)
    s = _scale(frame)
    x1, _, _, y2 = (int(v) for v in person_xyxy)
    _label(frame, "PPE " + " ".join(parts), x1, y2 + int(10 * s), color, s * 0.85,
           text_color=(15, 15, 15) if color != RED else WHITE, anchor_bottom=False)


def draw_evidence_focus(frame, xyxy, label: str) -> None:
    """Extra emphasis on the person who triggered an event (evidence images only)."""
    s = _scale(frame)
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    pad = int(8 * s)
    cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), WHITE, max(2, int(2 * s)), cv2.LINE_AA)
    cv2.rectangle(frame, (x1 - pad - 3, y1 - pad - 3), (x2 + pad + 3, y2 + pad + 3), RED, max(2, int(2 * s)), cv2.LINE_AA)
    _label(frame, label, x1 - pad - 3, y2 + pad + 4, RED, s, anchor_bottom=False)


_ASCII_MAP = str.maketrans({"·": "|", "—": "-", "–": "-", "’": "'", "“": '"', "”": '"'})


def ascii_text(text: str) -> str:
    """OpenCV's Hershey fonts only render ASCII."""
    return str(text).translate(_ASCII_MAP).encode("ascii", "replace").decode()


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, sec = divmod(seconds, 60)
    return f"{int(m):02d}:{sec:04.1f}"
