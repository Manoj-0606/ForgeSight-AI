"""Factory Command Center: KPIs, occupancy, heatmap and recent safety events for the active analysis."""
from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from forgesight import analytics, charts, ui
from forgesight import events as evt
from forgesight import incidents as inc
from forgesight.annotate import fmt_ts
from forgesight.results import choose_bin_seconds, interval_activity
from forgesight.state import PAGES, get_run, get_settings, thumb

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


SEV_TONE = {"Critical": "", "High": "", "Medium": " amber", "Low": " muted"}


def _recent_incidents(run, incidents, limit: int = 5) -> None:
    rec = incidents.sort_values(["time_s", "sev_rank"], ascending=[False, True]).head(limit)
    if rec.empty:
        st.markdown('<div class="fs-note">No incidents: no zone events, occupancy alerts or PPE flags in this '
                    'footage.</div>', unsafe_allow_html=True)
        return
    rows = []
    for r in rec.itertuples():
        img = thumb(run.path(r.evidence_crop)) if isinstance(r.evidence_crop, str) and r.evidence_crop else ""
        img_html = '<img src="' + img + '" alt="evidence" />' if img else ""
        who = f" · ID {r.track_id}" if pd.notna(r.track_id) else ""
        status = " · reviewed" if r.status == inc.REVIEWED else ""
        rows.append(
            f'<div class="fs-event{SEV_TONE[r.severity]}">{img_html}<div>'
            f'<div class="t">{ui.sev_badge(r.severity)} {ui.esc(r.type)} · {ui.esc(r.incident_id)}</div>'
            f'<div class="m">{fmt_ts(r.time_s)}{ui.esc(who)} · {ui.esc(r.dwell)}{status}</div>'
            f'<div class="m">{ui.esc(r.context)}</div></div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render() -> None:
    run = get_run()
    if run is None:
        ui.page_header("Factory Command Center", "Workforce & safety overview",
                       "Real-time-style overview of people activity and restricted-zone safety for analyzed footage.",
                       [ui.chip("No analysis loaded", "warn")])
        ui.empty_state(
            "No footage analyzed yet",
            "ForgeSight AI turns ordinary factory video into workforce insights and restricted-zone safety alerts. "
            "All metrics on this page come from a real analysis, so nothing is shown until a video has been processed.",
            [("Upload footage", "Fixed-camera MP4 of a factory floor"),
             ("Define the zone", "Mark the robot cell or hazard area"),
             ("Run AI analysis", "YOLO11 + ByteTrack on your CPU"),
             ("Review insights", "KPIs, heatmap and safety events")],
        )
        st.markdown("")
        _, c, _ = st.columns([2, 1, 2])
        if "video" in PAGES:
            c.page_link(PAGES["video"], label="Open Video Intelligence", icon=":material/videocam:", width="stretch")
        return

    s, m, p = run.summary, run.metrics, run.summary["processing"]
    zone_name = s["settings"]["zone_name"]
    chips = [ui.chip(s["original_name"], "ok"), ui.chip(s["camera_label"]),
             ui.chip("ByteTrack tracking" if run.tracking_active else "Tracking off",
                     "live" if run.tracking_active else "warn"),
             ui.chip(f"Analyzed {s['created_at'].replace('T', ' ')[:16]}")]
    ui.page_header("Factory Command Center", "Workforce & safety overview",
                   f"Insights from {fmt_ts(p['analyzed_duration_s'])} of footage · {p['frames_analyzed']:,} "
                   f"analyzed frames at {p['analysis_fps']:.1f} fps", chips)

    unique = m["unique_people_est"]
    min_frames = s["settings"]["min_track_frames"]
    cfg = get_settings()
    incidents = inc.build_incidents(run, cfg)
    alerts = inc.occupancy_alerts(run, cfg)
    occ_scope = cfg.occ_alert_scope if m["zone_enabled"] else "frame"
    at_start = m.get("zone_initial_occupants", 0)
    n = m["zone_entries"]
    start_note = (f" · {at_start} person{'s were' if at_start != 1 else ' was'} already inside when the footage "
                  f"began (not counted)") if at_start else ""
    ui.hazard_alert(run)
    if n:
        last = evt.intrusions(run.events).sort_values("entry_time_s").iloc[-1]
        who = f"ID {last.track_id}" if run.tracking_active else "an untracked person"
        st.markdown(f'<div class="fs-alert"><b>RESTRICTED ZONE ENTRY</b> {n} intrusion{"" if n == 1 else "s"} '
                    f'into {ui.esc(zone_name)} · latest by {ui.esc(who)} at {fmt_ts(last.entry_time_s)}'
                    f'{ui.esc(start_note)}</div>', unsafe_allow_html=True)
    elif at_start:
        st.markdown(f'<div class="fs-alert amber"><b>ZONE OCCUPIED AT START</b> No intrusions observed'
                    f'{ui.esc(start_note)}</div>', unsafe_allow_html=True)
    ui.kpi_row([
        ui.kpi("Person detections", f"{m['total_detections']:,}",
               f"Boxes summed over {p['frames_analyzed']:,} frames",
               help_text="Every person box in every analyzed frame. One person visible in 50 frames counts 50 "
                         "times. This measures activity, not headcount."),
        ui.kpi("Est. unique people", "—" if unique is None else unique,
               "Tracking disabled" if unique is None else f"Track IDs lasting ≥ {min_frames} frames",
               tone="muted" if unique is None else "",
               help_text="Distinct ByteTrack IDs, excluding very short tracks. Occlusions can split one person into "
                         "several IDs, so treat this as an estimate (an upper bound)."),
        ui.kpi("Peak occupancy", m["peak_occupancy"], f"at {fmt_ts(m['peak_occupancy_time_s'])} · avg "
               f"{m['avg_occupancy']:.1f}", unit="people",
               help_text="Most people visible in a single analyzed frame (observed occupancy)."),
        ui.kpi("Zone intrusions", n if m["zone_enabled"] else "—",
               (f"+{at_start} present at start · occupied {m['zone_occupied_s']:.0f}s" if at_start
                else f"{zone_name} · occupied {m['zone_occupied_s']:.0f}s")
               if m["zone_enabled"] else "Zone disabled",
               tone=("red" if n else "green") if m["zone_enabled"] else "muted",
               help_text="Confirmed entries during the footage: a person's foot point is inside the zone for "
                         "consecutive analyzed frames. Staying inside produces one event; re-entry needs a confirmed "
                         "exit first. People already inside when the footage began are listed separately."),
        ui.kpi("Occupancy alerts", len(alerts) if cfg.occ_alert_enabled else "—",
               f"≥ {cfg.occ_alert_threshold} {'in zone' if occ_scope == 'zone' else 'in view'} for ≥ "
               f"{cfg.occ_alert_min_s:g} s" if cfg.occ_alert_enabled else "Alerts disabled in Settings",
               tone=("amber" if len(alerts) else "green") if cfg.occ_alert_enabled else "muted",
               help_text="Periods where occupancy stayed at or above the threshold for the minimum time. Computed "
                         "from the per-frame detections; change the rule in Settings without re-analyzing."),
        ui.kpi("Processing status", "Complete", f"{p['elapsed_s']:.0f}s · {p['processing_fps']:.1f} frames/s on CPU",
               tone="green", help_text="Analysis ran locally with YOLO11 (ONNX Runtime, CPU)."),
    ])
    st.markdown("")

    left, right = st.columns([8, 4], gap="medium")
    with left:
        with ui.card("occupancy"):
            ui.section("Observed occupancy over time",
                       f"Per analyzed frame + {cfg.trend_window_s:g} s trend · ◆ intrusion · ○ present at start"
                       + (" · shaded = occupancy alert" if len(alerts) else ""))
            st.plotly_chart(charts.occupancy_chart(
                run.frames, run.events, m["zone_enabled"], height=362, trend_window_s=cfg.trend_window_s,
                alerts=alerts, threshold=cfg.occ_alert_threshold if cfg.occ_alert_enabled else None,
                threshold_scope=occ_scope), config=PLOT_CONFIG, key="cc_occ")
    with right:
        with ui.card("recent", height=430):
            n_open = int((incidents["status"] == inc.OPEN).sum())
            ui.section("Recent incidents", f"{len(incidents)} total · {n_open} open")
            _recent_incidents(run, incidents)
            if len(incidents) and "events" in PAGES:
                st.page_link(PAGES["events"], label="Review incidents & evidence", icon=":material/arrow_forward:")

    left, right = st.columns([6, 6], gap="medium")
    with left:
        with ui.card("heatmap"):
            modes = {"Presence": "where feet were detected", "Dwell hotspots": "where people stood still"}
            c1, c2 = st.columns([3, 2], vertical_alignment="center")
            with c1:
                ui.section("Movement heatmap", "")
            with c2:
                mode = st.segmented_control("Heatmap layer", list(modes), default="Presence", key="cc_heat_mode",
                                            label_visibility="collapsed") or "Presence"
            dwell_ok = run.heatmap_dwell is not None and run.heatmap_dwell.sum() > 0
            heat = run.heatmap_dwell if mode == "Dwell hotspots" and dwell_ok else run.heatmap
            if mode == "Dwell hotspots" and not dwell_ok:
                st.caption("Dwell layer not available for this run (analyzed before v1.1 or without tracking, or "
                           "nobody stood still). Showing presence.")
                mode = "Presence"
            bg = cv2.imread(str(run.background_path)) if run.background_path else None
            zone_pts = s["settings"]["zone_points"] if m["zone_enabled"] else None
            st.plotly_chart(charts.heatmap_figure(heat, bg, zone_pts, tuple(p["output_size"]), height=320,
                                                  measure="dwell" if mode == "Dwell hotspots" else "presence"),
                            config=PLOT_CONFIG, key="cc_heat")
            hs = analytics.heatmap_stats(heat, zone_pts)
            facts = [f"Layer: {modes[mode]}"]
            if hs["hotspot_pct"]:
                facts.append(f"hotspot at x≈{hs['hotspot_pct'][0]:.0f}%, y≈{hs['hotspot_pct'][1]:.0f}% of frame")
            if hs["zone_share"] is not None:
                facts.append(f"{hs['zone_share']:.0%} of {'dwell' if mode == 'Dwell hotspots' else 'presence'} "
                             f"inside {zone_name}")
            st.caption(" · ".join(facts))
    with right:
        with ui.card("activity"):
            bin_s = choose_bin_seconds(p["analyzed_duration_s"])
            intervals = interval_activity(run.frames, run.events, bin_s)
            ui.section("Activity by time interval", f"{bin_s}-second intervals")
            t1, t2 = st.tabs(["Detection activity", "Zone entries over time"])
            with t1:
                st.plotly_chart(charts.activity_chart(intervals, bin_s, height=330), config=PLOT_CONFIG, key="cc_act")
            with t2:
                st.plotly_chart(charts.events_chart(intervals, bin_s, height=330), config=PLOT_CONFIG, key="cc_evt")

    if run.hazard_active:
        _hazard_card(run)
    if run.ppe_active:
        _ppe_card(run)

    with ui.card("status"):
        ui.section("Video processing status", f"Run {run.run_id}")
        c1, c2, c3 = st.columns(3, gap="large")
        c1.markdown(ui.key_values([
            ("Status", "Completed"),
            ("Source video", s["original_name"]),
            ("Resolution", f"{s['video']['width']}×{s['video']['height']} @ {s['video']['fps']:.1f} fps"),
            ("Processed at", f"{p['output_size'][0]}×{p['output_size'][1]}"),
        ]), unsafe_allow_html=True)
        c2.markdown(ui.key_values([
            ("Model", f"{s['model']['name']} (Ultralytics {s['model']['version']})"),
            ("Runtime", s["model"]["runtime"]),
            ("Confidence threshold", f"{s['settings']['confidence']:.2f}"),
            ("Tracker", "ByteTrack" if run.tracking_active else "Disabled"),
        ]), unsafe_allow_html=True)
        c3.markdown(ui.key_values([
            ("Frames analyzed", f"{p['frames_analyzed']:,} (every {p['sampling_step']} frames)"),
            ("Analysis rate", f"{p['analysis_fps']:.2f} fps"),
            ("Processing time", f"{p['elapsed_s']:.1f}s"),
            ("Raw track IDs", s["tracking"]["raw_track_ids"] if run.tracking_active else "—"),
        ]), unsafe_allow_html=True)
        with st.expander("How these metrics are calculated"):
            st.markdown(
                f"""
- **Person detections**: sum of person boxes across all analyzed frames. It scales with time and sampling rate,
  so it measures *activity*, not *people*.
- **Estimated unique people**: number of ByteTrack IDs seen in at least {min_frames} analyzed frames. ID switches
  after occlusion can inflate it, so read it as an upper-bound estimate.
- **Observed occupancy**: number of people detected in a single analyzed frame. *Peak* is the maximum, *average*
  the mean over all analyzed frames.
- **Restricted-zone entry**: the bottom-centre of a person's box (≈ foot position) is inside the zone polygon for
  {s['settings']['entry_frames']} consecutive analyzed frames. The person must be outside for
  {s['settings']['exit_frames']} consecutive frames (or the track must be lost) before a re-entry can be counted.
- **Zone intrusion vs present at start**: people already inside when the footage began are not counted as
  intrusions. Intrusions are labelled *crossed into zone* (seen outside first) or *first detected inside zone*
  (walked into view inside the zone or reappeared from occlusion). A dwell shown as "≥" was cut off by the start
  or end of the footage or a lost track.
- **Occupancy alert**: occupancy ≥ threshold continuously for the minimum time, bridging dips shorter than
  {analytics.OCC_GAP_TOLERANCE_S:g} s. Each analyzed frame represents the time until the next one.
- **Movement heatmap**: foot positions accumulated on a {run.heatmap.shape[1]}×{run.heatmap.shape[0]} grid, lightly
  smoothed, normalised to the busiest cell, drawn over a median background frame.

*Prototype for video-based safety monitoring. Not a certified industrial safety system.*
"""
            )


def _hazard_card(run) -> None:
    hz = run.summary["hazard"]
    with ui.card("hazard"):
        ui.section("Fire & smoke checks · BETA",
                   f"{hz['checks']} checks · {hz['checks_per_second']} /s · confirm after {hz['confirm_checks']} "
                   f"consecutive ≥ {hz['threshold']:.2f}")
        st.plotly_chart(charts.hazard_chart(run.hazard_checks, run.hazards, hz["threshold"], hz.get("classes", ["fire"])),
                        config=PLOT_CONFIG, key="cc_hazard")
        st.caption("Frame-level classifier probabilities. Shaded = confirmed episode (one event and at most one "
                   "email each). Process flames such as furnaces or welding also score as fire, so verify on video.")


def _ppe_card(run) -> None:
    from forgesight import ppe

    with ui.card("ppe"):
        ui.section("PPE checks · BETA", f"{run.summary['ppe']['check_frames']} check frames · "
                                       f"{run.summary['ppe']['checks_per_second']} checks/s · per tracked person")
        cols = st.columns(len(run.summary["ppe"]["counts"]) or 1, gap="medium")
        for col, (item, c) in zip(cols, run.summary["ppe"]["counts"].items()):
            col.markdown(ui.kpi(ppe.ITEMS[item].label, c["detected"], unit="detected",
                                sub=f"{c['not detected']} not detected (verify) · {c['uncertain']} uncertain",
                                tone="amber" if c["not detected"] else "green"), unsafe_allow_html=True)
        st.caption("Detected = item found in most clear-view checks of that person. Not detected = person clearly "
                   "visible but item found in almost none. A lead to verify on video, not a confirmed violation. "
                   "Validated on construction imagery only (see docs/PPE_VALIDATION.md).")
