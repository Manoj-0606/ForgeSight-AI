"""Video Intelligence: upload footage, configure the zone, run analysis, review annotated output."""
from __future__ import annotations

import hashlib
import time
import traceback
from pathlib import Path

import cv2
import streamlit as st

from forgesight import ui
from forgesight import events as evt
from forgesight.annotate import fmt_ts
from forgesight.config import MODEL_CHOICES, SUPPORTED_EXTENSIONS, UPLOADS_DIR, MAX_RECOMMENDED_SECONDS, ensure_dirs
from forgesight.detector import ModelError
from forgesight.pipeline import AnalysisError, Progress, analyze_video
from forgesight.hazard import HazardModelError
from forgesight.notify import AlertDispatcher
from forgesight.ppe import PpeModelError
from forgesight.state import (PAGES, email_config, get_hazard_model, get_model, get_ppe_model, get_run,
                              get_settings, set_run)
from forgesight.video_io import VideoError, plan_sampling, probe_video, read_first_frame
from forgesight.zones import validate_polygon
from views.zone_editor import render_zone_editor

BROWSER_CODECS = {"AVC1", "H264", "X264", "VP80", "VP90", "AV01"}


def _handle_upload(uploaded) -> None:
    data = uploaded.getvalue()
    sha = hashlib.sha1(data).hexdigest()[:12]
    current = st.session_state.get("upload")
    if current and current["sha"] == sha:
        return
    ext = Path(uploaded.name).suffix.lower()
    ensure_dirs()
    path = UPLOADS_DIR / f"{sha}{ext}"
    if not path.exists():
        path.write_bytes(data)
    entry = {"sha": sha, "name": uploaded.name, "path": str(path), "info": None, "error": None,
             "size_mb": len(data) / 1e6}
    try:
        entry["info"] = probe_video(path)
    except VideoError as exc:
        entry["error"] = str(exc)
    st.session_state.upload = entry
    st.session_state.pop("first_frame", None)


def _first_frame(upload):
    cache = st.session_state.get("first_frame")
    if cache and cache[0] == upload["sha"]:
        return cache[1]
    info = upload["info"]
    frame = read_first_frame(upload["path"], at_seconds=min(1.0, info.duration_s / 2 if info.duration_s else 0))
    if frame is None:
        frame = read_first_frame(upload["path"])
    st.session_state.first_frame = (upload["sha"], frame)
    return frame


def _run_analysis(upload) -> None:
    s = get_settings()
    info = upload["info"]
    plan = plan_sampling(info, s.analysis_fps, s.max_seconds)

    with ui.card("progress"):
        ui.section("Analysis in progress", "Do not navigate away until processing completes")
        bar = st.progress(0.0, text="Loading YOLO11 model…")
        stats = st.empty()
        preview = st.empty()

        try:
            t_model = time.perf_counter()
            model = get_model(s.model_name)
            load_s = time.perf_counter() - t_model
            ppe_model = None
            if s.ppe_enabled and s.tracking_enabled and s.ppe_items:
                bar.progress(0.0, text="Loading PPE model (beta)…")
                ppe_model = get_ppe_model()
            hazard_model, dispatcher = None, None
            if s.hazard_enabled and s.hazard_classes:
                bar.progress(0.0, text="Loading fire/smoke model (beta)…")
                hazard_model = get_hazard_model()
                cfg, _ = email_config()
                dispatcher = AlertDispatcher(cfg, enabled=s.email_enabled, cooldown_s=s.email_cooldown_s,
                                             max_per_run=s.email_max_per_run)
            bar.progress(0.0, text=f"Model ready ({load_s:.1f}s) · decoding video…")

            def on_progress(p: Progress) -> None:
                rate = p.done / p.elapsed if p.elapsed > 0 else 0
                remaining = (p.total - p.done) / rate if rate and p.total else None
                eta = f" · ETA {remaining:.0f}s" if remaining is not None else ""
                bar.progress(p.fraction, text=f"Analyzing frame {p.done}/{p.total or '?'}"
                                             f" · {fmt_ts(p.t)} of {fmt_ts(p.duration)}{eta}")
                stats.markdown(
                    '<div class="fs-chips">'
                    + ui.chip(f"{p.people} people in frame", "live")
                    + ui.chip(f"{p.events} zone entries", "crit" if p.events else "ok")
                    + ui.chip(f"{rate:.1f} frames/s")
                    + ui.chip(f"{p.elapsed:.0f}s elapsed") + "</div>",
                    unsafe_allow_html=True,
                )

            def on_preview(frame) -> None:
                preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch",
                              caption="Live annotated preview")

            run_dir = analyze_video(upload["path"], s, model, original_name=upload["name"],
                                    on_progress=on_progress, on_preview=on_preview, ppe_model=ppe_model,
                                    hazard_model=hazard_model, alert_dispatcher=dispatcher)
        except (VideoError, AnalysisError, ModelError, PpeModelError, HazardModelError) as exc:
            bar.empty(); stats.empty(); preview.empty()
            st.error(f"**Analysis failed.** {exc}", icon=":material/error:")
            return
        except Exception as exc:  # unexpected: keep the app usable and show details
            bar.empty(); stats.empty(); preview.empty()
            st.error(f"**Analysis failed unexpectedly:** {exc}", icon=":material/error:")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            return

    set_run(run_dir)
    st.session_state.analysis_done = True
    st.rerun()


def _ppe_summary(run) -> str:
    parts = []
    for item, c in run.summary["ppe"]["counts"].items():
        short = "Hat" if item == "hardhat" else "Vest"
        parts.append(f"{short} {c['detected']}✓ {c['not detected']}✗ {c['uncertain']}?")
    return " · ".join(parts) or "no people judged"


def _hazard_summary(run) -> str:
    hz = run.summary["hazard"]
    ev = ", ".join(f"{n} {c}" for c, n in hz["events"].items()) or "0"
    mails = ", ".join(f"{n} {st}" for st, n in hz["email"]["outcomes"].items()) or "none"
    return f"{ev} event(s) · emails: {mails}"


def _results(run) -> None:
    s, m, p = run.summary, run.metrics, run.summary["processing"]
    st.markdown("")
    ui.section(f"Analysis results · {s['original_name']}", f"Run {run.run_id}")
    at_start = m.get("zone_initial_occupants", 0)
    ui.hazard_alert(run)
    if m["zone_entries"]:
        last = evt.intrusions(run.events).sort_values("entry_time_s").iloc[-1]
        who = f"ID {last.track_id}" if run.tracking_active else "untracked person"
        extra = f" · {at_start} already inside at start (not counted)" if at_start else ""
        st.markdown(f'<div class="fs-alert"><b>RESTRICTED ZONE ENTRY</b> {m["zone_entries"]} intrusion'
                    f'{"" if m["zone_entries"] == 1 else "s"} into {ui.esc(s["settings"]["zone_name"])} '
                    f'· latest {ui.esc(who)} at {fmt_ts(last.entry_time_s)}{extra}</div>', unsafe_allow_html=True)
    elif at_start:
        st.markdown(f'<div class="fs-alert amber"><b>ZONE OCCUPIED AT START</b> No intrusions observed · '
                    f'{at_start} already inside when the footage began</div>', unsafe_allow_html=True)

    left, right = st.columns([7, 4], gap="medium")
    with left:
        with ui.card("result-video"):
            if p["browser_playable_video"]:
                st.video(str(run.video_path))
            else:
                st.warning("The annotated video was saved with a fallback codec that browsers may not play. "
                           "Download it to view it locally.")
            st.download_button("Download annotated video (MP4)", data=run.video_path.read_bytes(),
                               file_name=f"{run.run_id}_annotated.mp4", mime="video/mp4",
                               icon=":material/download:", width="stretch")
    with right:
        with ui.card("result-summary"):
            ui.section("Detection summary")
            unique = m["unique_people_est"]
            st.markdown(ui.key_values([
                ("Status", "Completed"),
                ("Frames analyzed", f"{p['frames_analyzed']:,}"),
                ("Analysis rate", f"{p['analysis_fps']:.2f} fps (every {p['sampling_step']} frames)"),
                ("Person detections", f"{m['total_detections']:,}"),
                ("Est. unique people", "n/a (tracking off)" if unique is None else f"{unique}"),
                ("Peak occupancy", f"{m['peak_occupancy']} at {fmt_ts(m['peak_occupancy_time_s'])}"),
                ("Average occupancy", f"{m['avg_occupancy']:.1f}"),
                ("Zone intrusions", f"{m['zone_entries']}" if m["zone_enabled"] else "zone disabled"),
                ("Present at start", f"{at_start}" if m["zone_enabled"] else "—"),
                ("Processing time", f"{p['elapsed_s']:.1f}s · {p['processing_fps']:.1f} frames/s"),
                ("Model", f"{s['model']['name']} · {s['model']['runtime']}"),
                ("Tracking", "ByteTrack" if run.tracking_active else "off"),
                ("PPE (beta)", _ppe_summary(run) if run.ppe_active else "off"),
                ("Fire/smoke (beta)", _hazard_summary(run) if run.hazard_active else "off"),
            ]), unsafe_allow_html=True)
            st.markdown("")
            c1, c2 = st.columns(2)
            if "command" in PAGES:
                c1.page_link(PAGES["command"], label="Command Center", icon=":material/space_dashboard:")
            if "events" in PAGES:
                c2.page_link(PAGES["events"], label="Safety events", icon=":material/warning:")
        if s["tracking"].get("note"):
            st.warning(s["tracking"]["note"])
        if s.get("ppe", {}).get("note"):
            st.warning(s["ppe"]["note"])


def render() -> None:
    s = get_settings()
    upload = st.session_state.get("upload")
    chips = [ui.chip(MODEL_CHOICES[s.model_name].split(" —")[0] + " · CPU", "live"),
             ui.chip(f"{s.analysis_fps:g} fps sampling"),
             ui.chip(f"conf ≥ {s.confidence:.2f}"),
             ui.chip("ByteTrack on" if s.tracking_enabled else "Tracking off", "ok" if s.tracking_enabled else "warn")]
    if s.ppe_enabled:
        chips.append(ui.chip("PPE beta on" if s.tracking_enabled else "PPE needs tracking", "warn"))
    if s.hazard_enabled:
        chips.append(ui.chip("Fire/smoke beta on" + (" · email" if s.email_enabled else ""), "warn"))
    ui.page_header("Video Intelligence", "Analyze factory footage",
                   "Upload recorded footage, define the restricted zone, then run YOLO11 person detection, "
                   "tracking and zone monitoring.", chips)

    if st.session_state.pop("analysis_done", False):
        st.toast("Analysis complete: dashboard and events updated.", icon=":material/check_circle:")

    left, right = st.columns([7, 5], gap="medium")
    with left:
        with ui.card("upload"):
            ui.section("1 · Source footage", "MP4 recommended · max 200 MB")
            uploaded = st.file_uploader("Upload factory video", type=list(SUPPORTED_EXTENSIONS),
                                        label_visibility="collapsed", key="uploader")
            if uploaded is not None:
                try:
                    _handle_upload(uploaded)
                except OSError as exc:
                    st.error(f"Could not save the uploaded file: {exc}")
                upload = st.session_state.get("upload")
            elif upload is None:
                st.caption("Use a fixed-camera clip where people's full bodies are visible. A moving camera "
                           "makes a fixed zone meaningless.")

            if upload:
                if upload["error"]:
                    st.error(f"**{upload['name']}** can't be analyzed. {upload['error']}", icon=":material/error:")
                else:
                    info = upload["info"]
                    if info.codec in BROWSER_CODECS:
                        st.video(upload["path"])
                    else:
                        frame = _first_frame(upload)
                        if frame is not None:
                            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
                        st.caption(f"Codec {info.codec} may not play in the browser. Showing a still frame; "
                                   "analysis is unaffected.")
                    dur = fmt_ts(info.duration_s) if info.duration_s else "unknown"
                    st.markdown('<div class="fs-chips">'
                                + ui.chip(upload["name"], "ok")
                                + ui.chip(f"{info.width}×{info.height}")
                                + ui.chip(f"{info.fps:.1f} fps" + (" (est.)" if info.fps_estimated else ""))
                                + ui.chip(f"duration {dur}")
                                + ui.chip(f"codec {info.codec}")
                                + ui.chip(f"{upload['size_mb']:.1f} MB") + "</div>", unsafe_allow_html=True)
                    if info.duration_s > MAX_RECOMMENDED_SECONDS and not s.max_seconds:
                        st.warning(f"This video is longer than {MAX_RECOMMENDED_SECONDS // 60} minutes. CPU "
                                   "analysis may take a while. Consider limiting the analyzed duration in Settings.")

    with right:
        with ui.card("zone"):
            ui.section("2 · Restricted zone", "Robot cell / hazardous machinery area")
            frame = _first_frame(upload) if upload and not upload["error"] else None
            if frame is None and get_run() and get_run().background_path:
                frame = cv2.imread(str(get_run().background_path))
            render_zone_editor(frame, key="vi")

    with ui.card("run"):
        ui.section("3 · Run AI analysis", "Settings can be tuned on the Settings page")
        ready = bool(upload and not upload["error"])
        zone_ok = (not s.zone_enabled) or validate_polygon(s.zone_points)[0]
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c1:
            if ready:
                plan = plan_sampling(upload["info"], s.analysis_fps, s.max_seconds)
                limit = f"first {s.max_seconds}s" if s.max_seconds else "full video"
                st.markdown(
                    '<div class="fs-chips">'
                    + ui.chip(f"{plan.frames_to_analyze or '?'} frames to analyze ({limit})")
                    + ui.chip(f"every {plan.step} source frames → {plan.effective_fps:.1f} fps")
                    + ui.chip(f"zone: {s.zone_name}" if s.zone_enabled else "zone disabled",
                              "warn" if s.zone_enabled else "") + "</div>", unsafe_allow_html=True)
            else:
                st.caption("Upload a valid video to enable analysis.")
            if not zone_ok:
                st.error("Fix the restricted zone polygon before running the analysis.")
        with c2:
            go = st.button("Run AI Analysis", type="primary", icon=":material/play_arrow:",
                           disabled=not (ready and zone_ok), width="stretch")

    if go and ready and zone_ok:
        _run_analysis(upload)

    run = get_run()
    if run is not None:
        _results(run)
