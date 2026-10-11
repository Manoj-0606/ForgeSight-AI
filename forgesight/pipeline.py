"""End-to-end video analysis: detect → track → zone monitoring → annotate → persist.

This module has no Streamlit dependency so it can be tested and run headless.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import __version__
from . import events as evt
from .annotate import (AMBER_DARK, HAZARD, RED, draw_alert_banner, draw_evidence_focus, draw_hazard_panel,
                       draw_hud, draw_person, draw_ppe_box, draw_ppe_tag,
                       draw_zone, fmt_ts)
from .config import RUNS_DIR, Settings
from .detector import PersonDetector
from .hazard import CLASSES as HZ_CLASSES
from .hazard import HAZARD_MODEL_NOTE, HAZARD_MODEL_REVISION, HazardMonitor
from .notify import AlertDispatcher
from .ppe import ITEMS as PPE_ITEMS
from .ppe import NOT_DETECTED, PPE_MODEL_NOTE, PPE_MODEL_REVISION, PpeMonitor
from .video_io import (
    AnnotatedVideoWriter,
    VideoError,
    iter_sampled_frames,
    plan_sampling,
    probe_video,
    processing_size,
)
from .zones import ZoneMonitor, foot_point, point_in_polygon, polygon_to_pixels, validate_polygon

HEATMAP_COLS = 96
# Below this foot speed (fraction of frame width per second) a person counts as stationary for the dwell heatmap.
STATIONARY_SPEED_FRAC = 0.03
ALERT_BANNER_SECONDS = 2.0
BANNER_ENTRY = "RESTRICTED ZONE ENTRY"
BANNER_AT_START = "PERSON IN ZONE AT START"
HAZARD_COLUMNS = ["event_id", "hazard", "start_time_s", "confirmed_time_s", "end_time_s", "max_score",
                  "mean_score", "positive_checks", "frame_index", "detected_at", "evidence_image",
                  "email_status", "email_detail"]


@dataclass
class Progress:
    done: int
    total: int  # 0 if unknown
    t: float
    duration: float
    people: int
    events: int
    elapsed: float

    @property
    def fraction(self) -> float:
        if self.total:
            return min(1.0, self.done / self.total)
        if self.duration:
            return min(1.0, self.t / self.duration)
        return 0.0


class AnalysisError(Exception):
    pass


def _slug(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")[:40] or "video"


def _crop(frame: np.ndarray, xyxy, pad: float = 0.25) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh * 0.5)), int(min(h, y2 + pad * bh * 0.3))
    return frame[y1:y2, x1:x2]


def _outcome_counts(dispatcher) -> dict:
    counts: dict[str, int] = {}
    for row in (dispatcher.log if dispatcher else []):
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def _finalize_run_dir(work_dir: Path, final_dir: Path, attempts: int = 20, delay_s: float = 0.25) -> None:
    """Rename the finished run folder into place.

    On Windows a just-written file can be held briefly by antivirus or the search indexer, which makes the
    rename fail with PermissionError. Retry, then fall back to copy + delete.
    """
    for _ in range(attempts):
        try:
            work_dir.rename(final_dir)
            return
        except OSError:
            time.sleep(delay_s)
    shutil.copytree(work_dir, final_dir)
    shutil.rmtree(work_dir, ignore_errors=True)


def analyze_video(
    video_path: str | Path,
    settings: Settings,
    model,
    original_name: str | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    on_preview: Callable[[np.ndarray], None] | None = None,
    preview_every: int = 8,
    ppe_model=None,
    hazard_model=None,
    alert_dispatcher: AlertDispatcher | None = None,
) -> Path:
    """Analyze a video and write all outputs to a new run folder. Returns the run folder.

    ``ppe_model`` (optional) enables the PPE beta when ``settings.ppe_enabled`` is set and tracking is active.
    ``hazard_model`` (optional) enables fire/smoke checks when ``settings.hazard_enabled`` is set; confirmed
    events are passed to ``alert_dispatcher`` (email), whose outcome is recorded per event.
    """
    video_path = Path(video_path)
    original_name = original_name or video_path.name
    info = probe_video(video_path)
    plan = plan_sampling(info, settings.analysis_fps, settings.max_seconds)
    width, height = processing_size(info.width, info.height, settings.max_width)
    duration = (plan.max_frame_index / info.fps) if plan.max_frame_index else info.duration_s
    if info.duration_s:
        duration = min(duration, info.duration_s) if duration else info.duration_s

    zone_active = bool(settings.zone_enabled)
    polygon_px = None
    if zone_active:
        ok, msg = validate_polygon(settings.zone_points)
        if not ok:
            raise AnalysisError(f"Restricted zone is invalid: {msg}")
        polygon_px = polygon_to_pixels(settings.zone_points, width, height)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{_slug(original_name)}"
    final_dir = RUNS_DIR / run_id
    work_dir = RUNS_DIR / f"{run_id}.partial"
    (work_dir / "evidence").mkdir(parents=True, exist_ok=True)

    detector = PersonDetector(model, settings.confidence, settings.tracking_enabled, plan.effective_fps)
    tracking = detector.tracking_active
    monitor = ZoneMonitor(
        entry_frames=max(1, int(settings.entry_frames)),
        exit_frames=max(1, int(settings.exit_frames)),
        lost_frames=int(round(plan.effective_fps * 2.5)) + 1,
    )

    grid_h = max(1, int(round(HEATMAP_COLS * height / width)))
    heat = np.zeros((grid_h, HEATMAP_COLS), dtype=np.float32)
    heat_dwell = np.zeros_like(heat)
    last_foot: dict[int, tuple[float, float, float]] = {}
    trails: dict = defaultdict(lambda: deque(maxlen=max(4, int(plan.effective_fps * 2))))
    tracks: dict[int, dict] = {}
    frame_rows: list[dict] = []
    events: list[dict] = []
    visit_to_event: dict[int, dict] = {}
    bg_samples: list[np.ndarray] = []
    bg_every = max(1, plan.frames_to_analyze // 30) if plan.frames_to_analyze else 10
    alert_until, alert_sub, alert_title, alert_color = -1.0, "", BANNER_ENTRY, RED
    n_intrusions = 0

    ppe_note = None
    if settings.ppe_enabled and not tracking:
        ppe_note = "PPE checks need tracking to aggregate results per person; they were skipped."
    elif settings.ppe_enabled and ppe_model is None:
        ppe_note = "PPE model was not loaded; PPE checks were skipped."
    ppe_active = bool(settings.ppe_enabled and tracking and ppe_model is not None and settings.ppe_items)
    ppe_mon = PpeMonitor(list(settings.ppe_items)) if ppe_active else None
    ppe_every = max(1, int(round(plan.effective_fps / max(settings.ppe_rate_fps, 0.1))))
    ppe_checks = 0

    hz_active = bool(settings.hazard_enabled and hazard_model is not None)
    hz_note = ("Fire/smoke model was not loaded; checks were skipped."
               if settings.hazard_enabled and hazard_model is None else None)
    hz_classes = [c for c in HZ_CLASSES if c in settings.hazard_classes]
    hz_mon = HazardMonitor(settings.hazard_confirm_checks, settings.hazard_clear_checks, hz_classes)
    hz_every = max(1, int(round(plan.effective_fps / max(settings.hazard_rate_fps, 0.1))))
    hz_rows: list[dict] = []
    hz_events: list[dict] = []
    hz_by_episode: dict[int, dict] = {}
    hz_last = None
    if hz_active and alert_dispatcher is None:
        alert_dispatcher = AlertDispatcher(None, enabled=settings.email_enabled)
    alert_priority = 0  # 3 fire/smoke > 2 intrusion > 1 present at start

    writer = AnnotatedVideoWriter(work_dir / "annotated.mp4", (width, height), plan.effective_fps)
    started = time.perf_counter()
    last_t = first_t = 0.0
    sample_idx = -1

    try:
        for sample_idx, (frame_idx, t, frame) in enumerate(
            iter_sampled_frames(video_path, info, plan, (width, height))
        ):
            last_t = t
            if sample_idx == 0:
                first_t = t
            if sample_idx % bg_every == 0 and len(bg_samples) < 40:
                bg_samples.append(cv2.resize(frame, (min(640, width), int(min(640, width) * height / width))))

            dets = detector.process(frame)
            annotated = frame.copy()
            n = len(dets)
            feet = [foot_point(b) for b in dets.xyxy]
            inside = [bool(zone_active and point_in_polygon(polygon_px, fx, fy)) for fx, fy in feet]

            for i, (fx, fy) in enumerate(feet):
                gx = int(np.clip(fx / width * HEATMAP_COLS, 0, HEATMAP_COLS - 1))
                gy = int(np.clip(fy / height * grid_h, 0, grid_h - 1))
                heat[gy, gx] += 1.0
                if tracking:  # dwell layer: people standing still or moving slowly
                    tid = int(dets.track_ids[i])
                    prev = last_foot.get(tid)
                    if prev is not None and t > prev[2]:
                        speed = np.hypot(fx - prev[0], fy - prev[1]) / (t - prev[2])
                        if speed < STATIONARY_SPEED_FRAC * width:
                            heat_dwell[gy, gx] += 1.0
                    last_foot[tid] = (fx, fy, t)

            if tracking:
                for i in range(n):
                    tid = int(dets.track_ids[i])
                    trails[tid].append((int(feet[i][0]), int(feet[i][1])))
                    ts = tracks.setdefault(tid, {"track_id": tid, "first_seen_s": t, "last_seen_s": t,
                                                 "frames": 0, "conf_sum": 0.0, "zone_frames": 0,
                                                 "last_outside_t": None})
                    ts["last_seen_s"] = t
                    ts["frames"] += 1
                    ts["conf_sum"] += float(dets.confidence[i])
                    ts["zone_frames"] += int(inside[i])
                    if not inside[i]:
                        ts["last_outside_t"] = t

            new_visits = []
            if zone_active:
                if tracking:
                    obs = {int(dets.track_ids[i]): inside[i] for i in range(n)}
                else:
                    obs = {"zone": any(inside)}
                new_visits = monitor.update(t, frame_idx, obs)

            zone_count = sum(inside)
            if zone_active:
                draw_zone(annotated, polygon_px, settings.zone_name, zone_count > 0)
            for i in range(n):
                tid = int(dets.track_ids[i]) if tracking else None
                draw_person(annotated, dets.xyxy[i], tid, float(dets.confidence[i]), inside[i],
                            list(trails[tid]) if tracking else None)

            ppe_missing_now = []
            if ppe_active:
                if sample_idx % ppe_every == 0:
                    ppe_checks += 1
                    ppe_dets = ppe_model.detect(frame, settings.ppe_confidence)
                    persons = [(int(dets.track_ids[i]), dets.xyxy[i]) for i in range(n)]
                    ppe_missing_now, linked = ppe_mon.update(t, height, persons, ppe_dets)
                    for (_, item), (b, c) in linked.items():
                        draw_ppe_box(annotated, b, f"{PPE_ITEMS[item].label.upper()} {c:.2f}")
                for i in range(n):
                    tid = int(dets.track_ids[i])
                    st_ = {item: ppe_mon.status(tid, item) for item in ppe_mon.items}
                    if any(v is not None for v in st_.values()):
                        draw_ppe_tag(annotated, dets.xyxy[i], {k: v or "uncertain" for k, v in st_.items()})

            new_hazards = []
            if hz_active:
                if sample_idx % hz_every == 0:
                    hz_last = hazard_model.scores(frame)
                    hz_rows.append({"time_s": round(t, 3), "fire": round(hz_last["fire"], 4),
                                    "smoke": round(hz_last["smoke"], 4)})
                    new_hazards = hz_mon.update(t, {c: (hz_last[c] if hz_last[c] >= settings.hazard_threshold
                                                        else 0.0) for c in HZ_CLASSES})
                draw_hazard_panel(annotated, hz_last, {c: hz_mon.active(c) for c in hz_classes},
                                  settings.hazard_threshold)
            for ep in new_hazards:
                n_same = sum(1 for e in hz_events if e["hazard"] == ep.cls) + 1
                hz_event = {
                    "event_id": f"{ep.cls.upper()}-{n_same:03d}", "hazard": ep.cls,
                    "start_time_s": round(ep.start_t, 2), "confirmed_time_s": round(ep.confirmed_t, 2),
                    "end_time_s": None, "max_score": round(ep.max_conf, 3), "mean_score": None,
                    "positive_checks": None, "frame_index": frame_idx,
                    "detected_at": datetime.now().isoformat(timespec="seconds"),
                    "evidence_image": "", "email_status": "", "email_detail": "",
                }
                hz_events.append(hz_event)
                hz_by_episode[id(ep)] = hz_event
                alert_title, alert_color, alert_priority = f"POTENTIAL {ep.cls.upper()} DETECTED", HAZARD, 3
                alert_sub = f"{hz_event['event_id']} | score {ep.max_conf:.2f} | BETA - VERIFY ON VIDEO"
                alert_until = t + ALERT_BANNER_SECONDS

            for visit in new_visits:
                # The person who triggered the entry (tracking) or the most confident person inside.
                if tracking:
                    idx = next(i for i in range(n) if int(dets.track_ids[i]) == visit.key)
                else:
                    idx = max((i for i in range(n) if inside[i]), key=lambda i: dets.confidence[i])
                tid = int(visit.key) if tracking else None
                event_id = f"EVT-{len(events) + 1:03d}"
                if tracking:
                    ts = tracks[tid]
                    context = evt.classify_tracked_entry(visit.entry_t, ts["first_seen_s"],
                                                         ts["last_outside_t"], first_t)
                else:
                    context = evt.classify_untracked_entry(visit.entry_t, first_t)
                n_intrusions += int(context != evt.CTX_AT_START)
                events.append({
                    "event_id": event_id,
                    "event_type": evt.event_type_for(context),
                    "entry_context": context,
                    "zone_name": settings.zone_name,
                    "track_id": tid,
                    "track_first_seen_s": round(tracks[tid]["first_seen_s"], 2) if tracking else None,
                    "entry_time_s": round(visit.entry_t, 2),
                    "confirmed_time_s": round(visit.confirmed_t, 2),
                    "exit_time_s": None,
                    "dwell_s": None,
                    "dwell_is_lower_bound": None,
                    "exit_reason": None,
                    "frame_index": frame_idx,
                    "confidence": round(float(dets.confidence[idx]), 3),
                    "foot_x_pct": round(feet[idx][0] / width * 100, 1),
                    "foot_y_pct": round(feet[idx][1] / height * 100, 1),
                    "people_in_zone": zone_count,
                    "_xyxy": dets.xyxy[idx].tolist(),
                })
                visit_to_event[id(visit)] = events[-1]

            if new_visits:
                new_events = [visit_to_event[id(v)] for v in new_visits]
                shown = [e for e in new_events if e["event_type"] == evt.TYPE_ENTRY] or new_events
                whos = [f"ID {e['track_id']}" if tracking else "PERSON" for e in shown]
                who = ", ".join(whos[:3]) + (f" +{len(whos) - 3}" if len(whos) > 3 else "")
                intrusion = shown[0]["event_type"] == evt.TYPE_ENTRY
                prio = 2 if intrusion else 1
                # A banner is only replaced by one of equal or higher priority while it is still showing.
                if t > alert_until or prio >= alert_priority:
                    alert_title = BANNER_ENTRY if intrusion else BANNER_AT_START
                    alert_color, alert_priority = (RED if intrusion else AMBER_DARK), prio
                    alert_sub = f"{who} | {settings.zone_name} | {fmt_ts(shown[0]['entry_time_s'])}"
                    alert_until = t + ALERT_BANNER_SECONDS

            # Evidence images get their own banner naming exactly the person of that event.
            scene = annotated.copy() if (new_visits or ppe_missing_now or new_hazards) else None
            if t <= alert_until:
                draw_alert_banner(annotated, alert_title, alert_sub, alert_color)
            draw_hud(annotated, settings.camera_label, t, n, zone_count, n_intrusions, tracking, ppe_active)

            for tid, item in ppe_missing_now:
                tl = ppe_mon.tallies[(tid, item)]
                if tl.evidence_image:
                    continue
                xyxy = next(dets.xyxy[i] for i in range(n) if int(dets.track_ids[i]) == tid)
                label = PPE_ITEMS[item].label
                img = scene.copy()  # clean frame: no zone banner underneath
                draw_alert_banner(img, f"{label.upper()} NOT DETECTED",
                                  f"VERIFY | ID {tid} | {fmt_ts(t)} | PPE BETA", AMBER_DARK)
                draw_hud(img, settings.camera_label, t, n, zone_count, n_intrusions, tracking, ppe_active)
                draw_evidence_focus(img, xyxy, f"ID {tid} | {label.upper()} NOT DETECTED")
                base = f"evidence/PPE_id{tid}_{item}"
                cv2.imwrite(str(work_dir / f"{base}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                crop = _crop(scene, xyxy)
                if crop.size:
                    cv2.imwrite(str(work_dir / f"{base}_crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                tl.evidence_image, tl.evidence_crop, tl.evidence_t = f"{base}.jpg", f"{base}_crop.jpg", t

            for visit in new_visits:
                ev = visit_to_event[id(visit)]
                base = f"{ev['event_id']}_t{visit.entry_t:07.2f}s" + (f"_id{ev['track_id']}" if tracking else "")
                full_rel = f"evidence/{base}.jpg"
                crop_rel = f"evidence/{base}_crop.jpg"
                xyxy = ev.pop("_xyxy")
                evidence = scene.copy()
                who = f"ID {ev['track_id']}" if tracking else "PERSON"
                is_entry = ev["event_type"] == evt.TYPE_ENTRY
                draw_alert_banner(evidence, BANNER_ENTRY if is_entry else BANNER_AT_START,
                                  f"{ev['event_id']} | {who} | {settings.zone_name} | {fmt_ts(visit.entry_t)}",
                                  RED if is_entry else AMBER_DARK)
                draw_hud(evidence, settings.camera_label, t, n, zone_count, n_intrusions, tracking, ppe_active)
                draw_evidence_focus(evidence, xyxy, f"{ev['event_id']} | {who}")
                cv2.imwrite(str(work_dir / full_rel), evidence, [cv2.IMWRITE_JPEG_QUALITY, 90])
                crop = _crop(scene, xyxy)
                if crop.size:
                    cv2.imwrite(str(work_dir / crop_rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                ev["evidence_image"] = full_rel
                ev["evidence_crop"] = crop_rel if crop.size else ""
                ev["video_reference"] = f"annotated.mp4 @ {fmt_ts(visit.entry_t)}"

            for ep in new_hazards:
                hz_event = hz_by_episode[id(ep)]
                img = scene.copy()
                draw_alert_banner(img, f"POTENTIAL {ep.cls.upper()} DETECTED",
                                  f"{hz_event['event_id']} | score {ep.max_conf:.2f} | {fmt_ts(ep.start_t)} | BETA",
                                  HAZARD)
                draw_hud(img, settings.camera_label, t, n, zone_count, n_intrusions, tracking, ppe_active)
                rel = f"evidence/{hz_event['event_id']}_t{ep.start_t:07.2f}s.jpg"
                cv2.imwrite(str(work_dir / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                hz_event["evidence_image"] = rel
                row = alert_dispatcher.notify({
                    "event_id": hz_event["event_id"], "hazard": ep.cls, "video_time_s": ep.start_t,
                    "confirmed_time_s": ep.confirmed_t, "confidence": ep.max_conf,
                    "camera": settings.camera_label, "source_video": original_name, "run_id": run_id,
                    "detected_at": hz_event["detected_at"]}, work_dir / rel)
                hz_event["email_status"], hz_event["email_detail"] = row["status"], row["detail"]

            writer.write(annotated)
            frame_rows.append({
                "sample": sample_idx,
                "frame_index": frame_idx,
                "time_s": round(t, 3),
                "people": n,
                "people_in_zone": zone_count,
                "zone_occupied": zone_count > 0,
                "mean_confidence": round(float(dets.confidence.mean()), 3) if n else None,
            })

            if on_progress:
                on_progress(Progress(sample_idx + 1, plan.frames_to_analyze, t, duration, n, n_intrusions,
                                     time.perf_counter() - started))
            if on_preview and sample_idx % preview_every == 0:
                on_preview(annotated)

        if sample_idx < 0:
            raise AnalysisError("No frames could be read from the video.")

        monitor.finish(last_t)
        hz_mon.finish()
        for ep in hz_mon.episodes:
            e = hz_by_episode.get(id(ep))
            if e is not None:
                e.update(end_time_s=round(ep.end_t, 2), mean_score=round(ep.mean_conf, 3),
                         max_score=round(ep.max_conf, 3), positive_checks=ep.positive_checks)
        for visit in monitor.visits:
            ev = visit_to_event.get(id(visit))
            if ev is not None:
                ev["exit_time_s"] = None if visit.exit_t is None else round(visit.exit_t, 2)
                ev["dwell_s"] = None if visit.dwell_s is None else round(visit.dwell_s, 2)
                ev["exit_reason"] = visit.exit_reason
                ev["dwell_is_lower_bound"] = evt.dwell_is_lower_bound(ev["entry_context"], visit.exit_reason)
    except BaseException:
        writer.close()
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    writer.close()

    try:  # post-processing: a failure here must not leave a half-written run behind
        elapsed = time.perf_counter() - started
        frames_df = pd.DataFrame(frame_rows)
        if len(frames_df):
            end_s = duration if duration else None
            frames_df["dt_s"] = np.round(evt.sample_durations(frames_df["time_s"].to_numpy(), end_s,
                                                              1.0 / plan.effective_fps), 4)
        events_df = pd.DataFrame(events, columns=[
            "event_id", "event_type", "entry_context", "zone_name", "track_id", "track_first_seen_s",
            "entry_time_s", "confirmed_time_s", "exit_time_s", "dwell_s", "dwell_is_lower_bound", "exit_reason",
            "frame_index", "confidence", "foot_x_pct", "foot_y_pct", "people_in_zone", "evidence_image",
            "evidence_crop", "video_reference",
        ])
        tracks_df = pd.DataFrame([
            {**{k: v for k, v in ts.items() if k != "conf_sum"},
             "duration_s": round(ts["last_seen_s"] - ts["first_seen_s"], 2),
             "mean_confidence": round(ts["conf_sum"] / ts["frames"], 3),
             "counted_as_person": ts["frames"] >= settings.min_track_frames,
             "entered_zone": ts["zone_frames"] > 0}
            for ts in tracks.values()
        ], columns=["track_id", "first_seen_s", "last_seen_s", "frames", "zone_frames", "duration_s",
                    "mean_confidence", "counted_as_person", "entered_zone"])

        ppe_rows = []
        if ppe_active:
            for (tid, item), tl in sorted(ppe_mon.tallies.items(), key=lambda kv: (kv[1].first_eval_t, kv[0])):
                counted = tracks.get(tid, {}).get("frames", 0) >= settings.min_track_frames
                keep = counted and tl.status == NOT_DETECTED
                for rel in (tl.evidence_image, tl.evidence_crop):  # provisional evidence of resolved tallies
                    if rel and not keep:
                        (work_dir / rel).unlink(missing_ok=True)
                if not counted:
                    continue
                mb = tl.box_sum / max(tl.evaluated, 1)
                ppe_rows.append({
                    "track_id": tid, "item": item, "item_label": PPE_ITEMS[item].label, "status": tl.status,
                    "checks": tl.evaluated, "hits": tl.hits, "hit_ratio": round(tl.ratio, 3),
                    "max_confidence": round(tl.max_conf, 3), "first_check_s": round(tl.first_eval_t, 2),
                    "last_check_s": round(tl.last_eval_t, 2),
                    "box_x1": round(mb[0] / width, 4), "box_y1": round(mb[1] / height, 4),
                    "box_x2": round(mb[2] / width, 4), "box_y2": round(mb[3] / height, 4),
                    "evidence_image": tl.evidence_image if keep else "",
                    "evidence_crop": tl.evidence_crop if keep else "",
                    "evidence_time_s": round(tl.evidence_t, 2) if keep and tl.evidence_t is not None else None,
                })
        ppe_df = pd.DataFrame(ppe_rows, columns=[
            "track_id", "item", "item_label", "status", "checks", "hits", "hit_ratio", "max_confidence",
            "first_check_s", "last_check_s", "box_x1", "box_y1", "box_x2", "box_y2", "evidence_image",
            "evidence_crop", "evidence_time_s"])
        if len(ppe_df):
            missing = ppe_df["status"] == NOT_DETECTED
            order = ppe_df[missing].sort_values(["evidence_time_s", "track_id"]).index
            ppe_df["event_id"] = ""
            ppe_df.loc[order, "event_id"] = [f"PPE-{i + 1:03d}" for i in range(len(order))]
        if settings.ppe_enabled:
            ppe_df.to_csv(work_dir / "ppe.csv", index=False)
        if hz_active:
            pd.DataFrame(hz_events, columns=HAZARD_COLUMNS).to_csv(work_dir / "hazards.csv", index=False)
            pd.DataFrame(hz_rows, columns=["time_s", "fire", "smoke"]).to_csv(work_dir / "hazard_checks.csv",
                                                                           index=False)
            alert_dispatcher.write_log(work_dir / "alerts.csv")

        frames_df.to_csv(work_dir / "frames.csv", index=False)
        events_df.to_csv(work_dir / "events.csv", index=False)
        tracks_df.to_csv(work_dir / "tracks.csv", index=False)
        np.save(work_dir / "heatmap.npy", heat)
        if tracking:
            np.save(work_dir / "heatmap_dwell.npy", heat_dwell)
        if bg_samples:
            background = np.median(np.stack(bg_samples), axis=0).astype(np.uint8)
            cv2.imwrite(str(work_dir / "background.jpg"), background, [cv2.IMWRITE_JPEG_QUALITY, 90])

        people = frames_df["people"] if len(frames_df) else pd.Series([0])
        peak_idx = int(people.idxmax()) if len(frames_df) else 0
        unique = int(tracks_df["counted_as_person"].sum()) if tracking else None  # 0 when nobody was seen
        analyzed_s = float(frames_df["dt_s"].sum()) if len(frames_df) else 0.0
        summary = {
            "run_id": run_id,
            "schema_version": evt.SCHEMA_VERSION,
            "app_version": __version__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "original_name": original_name,
            "camera_label": settings.camera_label,
            "video": info.to_dict() | {"path": str(video_path)},
            "settings": settings.to_dict(),
            "model": {"name": getattr(model, "name", settings.model_name),
                      "version": getattr(model, "version", "unknown"),
                      "runtime": "ONNX Runtime (CPU)"},
            "processing": {
                "frames_analyzed": int(len(frames_df)),
                "sampling_step": plan.step,
                "analysis_fps": round(plan.effective_fps, 3),
                "analyzed_duration_s": round(analyzed_s, 2),
                "first_frame_s": round(first_t, 3),
                "output_size": [width, height],
                "elapsed_s": round(elapsed, 2),
                "processing_fps": round(len(frames_df) / elapsed, 2) if elapsed > 0 else None,
                "browser_playable_video": writer.browser_playable,
            },
            "tracking": {
                "enabled": settings.tracking_enabled,
                "active": tracking,
                "note": detector.tracking_error,
                "raw_track_ids": int(len(tracks_df)),
            },
            "ppe": {
                "enabled": bool(settings.ppe_enabled),
                "active": ppe_active,
                "note": ppe_note,
                "model": PPE_MODEL_NOTE if ppe_active else None,
                "model_revision": PPE_MODEL_REVISION if ppe_active else None,
                "items": list(settings.ppe_items),
                "confidence": settings.ppe_confidence,
                "checks_per_second": round(plan.effective_fps / ppe_every, 2) if ppe_active else None,
                "check_frames": ppe_checks,
                "counts": {item: {s: int(((ppe_df["item"] == item) & (ppe_df["status"] == s)).sum())
                                  for s in ("detected", "not detected", "uncertain")}
                           for item in settings.ppe_items} if ppe_active else {},
            },
            "hazard": {
                "enabled": bool(settings.hazard_enabled),
                "active": hz_active,
                "note": hz_note,
                "model": HAZARD_MODEL_NOTE if hz_active else None,
                "model_revision": HAZARD_MODEL_REVISION if hz_active else None,
                "threshold": settings.hazard_threshold,
                "classes": hz_classes,
                "checks_per_second": round(plan.effective_fps / hz_every, 2) if hz_active else None,
                "confirm_checks": settings.hazard_confirm_checks,
                "clear_checks": settings.hazard_clear_checks,
                "checks": len(hz_rows),
                "events": {c: sum(1 for e in hz_events if e["hazard"] == c) for c in HZ_CLASSES},
                "email": {
                    "enabled": bool(settings.email_enabled),
                    "cooldown_s": settings.email_cooldown_s,
                    "max_per_run": settings.email_max_per_run,
                    "outcomes": _outcome_counts(alert_dispatcher),
                },
            },
            "metrics": {
                "total_detections": int(people.sum()),
                "unique_people_est": unique,
                "peak_occupancy": int(people.max()),
                "peak_occupancy_time_s": float(frames_df["time_s"].iloc[peak_idx]) if len(frames_df) else 0.0,
                "avg_occupancy": round(float(people.mean()), 2),
                "zone_entries": int(len(evt.intrusions(events_df))),
                "zone_initial_occupants": int(len(evt.at_start(events_df))),
                "zone_occupied_s": round(float(frames_df.loc[frames_df["zone_occupied"], "dt_s"].sum()), 2)
                if zone_active and len(frames_df) else 0.0,
                "zone_enabled": zone_active,
            },
        }
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

        _finalize_run_dir(work_dir, final_dir)
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    return final_dir


__all__ = ["analyze_video", "AnalysisError", "VideoError", "Progress"]
