"""Safety Events: incident review (filters, severity, review status), evidence, PPE results and exports."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from forgesight import charts, ui
from forgesight import events as evt
from forgesight import incidents as inc
from forgesight import ppe
from forgesight import report
from forgesight.annotate import fmt_ts
from forgesight.results import choose_bin_seconds, events_export, interval_activity
from forgesight.state import PAGES, get_run, get_settings, thumb

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
SEV_COLORS = {"Critical": "#FF8A80", "High": "#FCA5A0", "Medium": "#FCD34D", "Low": "#9AA4B2"}


def _style(df: pd.DataFrame):
    return (df.style
            .map(lambda v: f"color: {SEV_COLORS.get(v, '')}; font-weight: 600", subset=["Severity"])
            .map(lambda v: "color: #6EE7B7" if v == inc.REVIEWED else "color: #9AA4B2", subset=["Status"]))


def _filters(incidents: pd.DataFrame, duration: float) -> pd.DataFrame:
    with ui.card("filters"):
        c1, c2, c3, c4, c5 = st.columns([2.2, 2, 1.6, 1.1, 2.2], vertical_alignment="bottom")
        types = c1.multiselect("Type", sorted(incidents["type"].unique()), key="f_type", placeholder="All types")
        sevs = c2.multiselect("Severity", inc.SEVERITIES, key="f_sev", placeholder="All severities")
        status = c3.segmented_control("Status", ["All", inc.OPEN, inc.REVIEWED], default="All", key="f_status")
        track = c4.text_input("Track ID", key="f_track", placeholder="e.g. 7, 12")
        hi = max(1.0, float(duration))
        tr = c5.slider("Time window", 0.0, hi, (0.0, hi), step=0.5, format="%.1f s", key="f_time")
    statuses = None if status in (None, "All") else [status]
    window = None if tr == (0.0, hi) else tr
    return inc.filter_incidents(incidents, types, sevs, statuses, window, track)


def _review_panel(run, row) -> None:
    ui.section("Review", "Saved to review.json in the run folder")
    st.markdown(f"{ui.sev_badge(row.status)} {ui.esc(f'reviewed {row.reviewed_at}' if row.reviewed_at else '')}",
                unsafe_allow_html=True)
    note = st.text_area("Reviewer note", value=row.note, key=f"note_{row.review_key}", height=80,
                        placeholder="e.g. Maintenance under lock-out/tag-out permit")
    c1, c2 = st.columns(2)
    if row.status == inc.OPEN:
        if c1.button("Mark reviewed", type="primary", icon=":material/task_alt:", width="stretch",
                     key=f"rev_{row.review_key}"):
            inc.save_review(run.run_dir, [row.review_key], inc.REVIEWED, note)
            st.rerun()
    elif c1.button("Reopen", icon=":material/undo:", width="stretch", key=f"reopen_{row.review_key}"):
        inc.save_review(run.run_dir, [row.review_key], inc.OPEN, note)
        st.rerun()
    if c2.button("Save note", icon=":material/save:", width="stretch", key=f"save_{row.review_key}",
                 disabled=note == row.note):
        inc.save_review(run.run_dir, [row.review_key], row.status, note)
        st.rerun()


def _ppe_results(run) -> None:
    with ui.card("ppe-results"):
        ui.section("PPE results per person · BETA", "Every tracked person that could be judged")
        df = run.ppe.sort_values(["status", "track_id"])
        st.dataframe(pd.DataFrame({
            "Person": df["track_id"].map(lambda v: f"ID {v}"),
            "Item": df["item_label"],
            "Status": df["status"].map(ppe.STATUS_LABELS),
            "Found in checks": [f"{h}/{c}" for h, c in zip(df["hits"], df["checks"])],
            "Share": df["hit_ratio"] * 100,
            "Checked": [f"{fmt_ts(a)}–{fmt_ts(b)}" for a, b in zip(df["first_check_s"], df["last_check_s"])],
            "Incident": df["event_id"].fillna("").replace("", "—"),
        }), hide_index=True, height=min(36 * len(df) + 40, 320),
            column_config={"Share": st.column_config.ProgressColumn("Share", min_value=0, max_value=100,
                                                                    format="%.0f%%")})
        st.caption(f"Detected: found in ≥ {ppe.DETECT_RATIO:.0%} of ≥ {ppe.MIN_EVALS} clear-view checks · Not "
                   f"detected: ≤ {ppe.MISSING_RATIO:.0%}, a lead to verify on video · Uncertain: too few or mixed "
                   "checks. The model's own 'no hard hat' class is not used (precision 0.20 in validation).")


def _exports(run, cfg, incidents, shown) -> None:
    with ui.card("export", height=318):
        ui.section("Report & exports", "Offline files · no cloud")
        if st.button("Prepare report", icon=":material/description:", width="stretch", type="primary",
                     help="Builds a self-contained HTML report (interactive charts, evidence, settings, limitations)."):
            st.session_state["report_html"] = (run.run_id, report.build_report_html(run, cfg, incidents))
        cached = st.session_state.get("report_html")
        if cached and cached[0] == run.run_id:
            html = cached[1]
            c1, c2 = st.columns(2)
            c1.download_button("Report (HTML)", data=html.encode("utf-8"), icon=":material/download:",
                               file_name=f"forgesight_report_{run.run_id}.html", mime="text/html", width="stretch")
            c2.download_button("Data package (ZIP)", data=report.build_data_package(run, cfg, incidents, html),
                               file_name=f"forgesight_{run.run_id}.zip", mime="application/zip",
                               icon=":material/folder_zip:", width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button(f"Incidents CSV ({len(shown)})",
                           data=report.incidents_export(run, shown).to_csv(index=False).encode("utf-8"),
                           file_name=f"forgesight_incidents_{run.run_id}.csv", mime="text/csv",
                           icon=":material/table:", width="stretch", help="Exports the incidents currently shown")
        c2.download_button("Zone event log CSV", data=events_export(run).to_csv(index=False).encode("utf-8"),
                           file_name=f"forgesight_events_{run.run_id}.csv", mime="text/csv",
                           icon=":material/download:", width="stretch")
        st.caption(f"Run folder: `{run.run_dir}`")


def render() -> None:
    run = get_run()
    if run is None:
        ui.page_header("Safety Events", "Incident review",
                       "Every safety incident with severity, review status, evidence frame and video reference.",
                       [ui.chip("No analysis loaded", "warn")])
        ui.empty_state("No safety events to review",
                       "Incidents are produced by analyzing a video. Run an analysis in Video Intelligence to "
                       "populate this page.")
        st.markdown("")
        _, c, _ = st.columns([2, 1, 2])
        if "video" in PAGES:
            c.page_link(PAGES["video"], label="Open Video Intelligence", icon=":material/videocam:", width="stretch")
        return

    s, m, cfg = run.summary, run.metrics, get_settings()
    incidents = inc.build_incidents(run, cfg)
    n_open = int((incidents["status"] == inc.OPEN).sum())
    zone_name = s["settings"]["zone_name"]
    chips = [ui.chip(f"{len(incidents)} incidents", "crit" if n_open else "ok"), ui.chip(f"{n_open} open"),
             ui.chip("Tracked IDs" if run.tracking_active else "No tracking", "live" if run.tracking_active else "warn")]
    if run.ppe_active:
        chips.append(ui.chip("PPE beta", "warn"))
    ui.page_header("Safety Events", "Incident review",
                   f"{s['original_name']} · {s['camera_label']} · zone “{zone_name}”", chips)

    if not m["zone_enabled"]:
        ui.note("Zone monitoring was disabled for this analysis, so no zone events were generated.")
    if not run.tracking_active:
        ui.note("Tracking was disabled for this analysis. Zone events are raised when the zone changes from empty to "
                "occupied. Individual people are not distinguished, and entries by a second person while the zone is "
                "already occupied are not counted separately.")

    dwell = evt.intrusions(run.events)["dwell_s"].dropna() if len(run.events) else pd.Series(dtype=float)
    by_src = incidents["source"].value_counts()
    n_crit = int((incidents["severity"] == "Critical").sum())
    cards = [
        ui.kpi("Open incidents", n_open, f"of {len(incidents)} · {len(incidents) - n_open} reviewed",
               tone="amber" if n_open else "green"),
        ui.kpi("Critical", n_crit, f"fire, or zone dwell ≥ {cfg.sev_critical_dwell_s:g} s", tone="red" if n_crit else "green"),
        ui.kpi("Zone intrusions", m["zone_entries"], f"+{m.get('zone_initial_occupants', 0)} present at start",
               tone="red" if m["zone_entries"] else "green"),
        ui.kpi("Median intrusion dwell", evt.format_dwell(dwell.median() if len(dwell) else None),
               "observed time inside", tone="amber"),
        ui.kpi("Occupancy alerts", int(by_src.get("occupancy", 0)),
               "disabled in Settings" if not cfg.occ_alert_enabled else
               f"≥ {cfg.occ_alert_threshold} for ≥ {cfg.occ_alert_min_s:g} s", tone="amber"),
    ]
    if run.ppe_active:
        cards.append(ui.kpi("PPE not detected", int(by_src.get("ppe", 0)), "beta · verify on video", tone="amber"))
    ui.kpi_row(cards)
    st.markdown("")

    if incidents.empty:
        with ui.card("none"):
            ui.section("No incidents")
            st.markdown("No zone events, occupancy alerts or PPE flags for this analysis. If you expected zone "
                        "events, check the zone position on the Settings page and re-run the analysis.")
        if run.ppe_active and len(run.ppe):
            _ppe_results(run)
        return

    shown = _filters(incidents, s["processing"]["analyzed_duration_s"]).reset_index(drop=True)

    with ui.card("log"):
        c1, c2, c3 = st.columns([3, 1.2, 1.2], vertical_alignment="center")
        c1.markdown(f'<div class="fs-section"><h3>Incident log</h3><span>{len(shown)} of {len(incidents)} shown · '
                    f'select a row to inspect</span></div>', unsafe_allow_html=True)
        open_keys = list(shown.loc[shown["status"] == inc.OPEN, "review_key"])
        rev_keys = list(shown.loc[shown["status"] == inc.REVIEWED, "review_key"])
        if c2.button(f"Mark {len(open_keys)} shown reviewed", icon=":material/done_all:", width="stretch",
                     disabled=not open_keys):
            inc.save_review(run.run_dir, open_keys, inc.REVIEWED)
            st.rerun()
        if c3.button(f"Reopen {len(rev_keys)} shown", icon=":material/undo:", width="stretch", disabled=not rev_keys):
            inc.save_review(run.run_dir, rev_keys, inc.OPEN)
            st.rerun()
        if shown.empty:
            st.info("No incidents match the filters.")
            return
        table = pd.DataFrame({
            "Evidence": [thumb(run.path(p), 160) if isinstance(p, str) and p else None for p in shown["evidence_crop"]],
            "ID": shown["incident_id"],
            "Severity": shown["severity"],
            "Status": shown["status"],
            "Type": shown["type"],
            "Time": shown["time_s"].map(lambda v: fmt_ts(v) if pd.notna(v) else "—"),
            "Track": shown["track_id"].map(lambda v: f"ID {v}" if pd.notna(v) else "—"),
            "Dwell / duration": shown["dwell"],
            "Context": shown["context"],
            "Why this severity": shown["severity_reason"],
            "Email": shown["notification"].replace("", "—"),
        })
        selection = st.dataframe(
            _style(table), hide_index=True, on_select="rerun", selection_mode="single-row", key="events_table",
            row_height=64, height=min(64 * len(table) + 40, 460),
            column_config={"Evidence": st.column_config.ImageColumn("Evidence", width="small"),
                           "Why this severity": st.column_config.TextColumn(width="large")},
        )
        rows = selection.selection.rows if selection and selection.selection else []
        idx = rows[0] if rows and rows[0] < len(shown) else 0

    r = shown.iloc[idx]
    st.markdown("")
    ui.section(f"Evidence · {r.incident_id}",
               "Selected incident" if rows else "Showing the first incident · select a row above")
    left, right = st.columns([7, 4], gap="medium")
    img_path = run.path(r.evidence_image) if isinstance(r.evidence_image, str) and r.evidence_image else None
    with left:
        with ui.card("evidence"):
            t1, t2 = st.tabs(["Evidence frame", "Annotated video at incident"])
            with t1:
                if img_path and img_path.exists():
                    st.image(str(img_path), width="stretch", caption=f"{r.incident_id} · {fmt_ts(r.time_s)}")
                else:
                    st.warning("No evidence image is available for this incident.")
            with t2:
                if s["processing"]["browser_playable_video"] and pd.notna(r.video_time_s):
                    st.video(str(run.video_path), start_time=max(0, int(r.video_time_s) - 1))
                    st.caption(f"Starts about 1 s before {fmt_ts(r.video_time_s)} · {run.video_path.name}")
                else:
                    st.info("The annotated video uses a fallback codec. Download it from Video Intelligence.")
    with right:
        with ui.card("evidence-meta"):
            ui.section("Incident details")
            st.markdown(f"{ui.sev_badge(r.severity)} <span style='color:#9AA4B2;font-size:12.5px'>"
                        f"{ui.esc(r.severity_reason)}</span>", unsafe_allow_html=True)
            details = [("Incident", r.incident_id), ("Type", r.type), ("Context", r.context),
                       ("Track ID", f"ID {r.track_id}" if pd.notna(r.track_id) else "—"),
                       ("Time", fmt_ts(r.time_s) if pd.notna(r.time_s) else "—"),
                       ("Until", fmt_ts(r.end_s) if pd.notna(r.end_s) else "—"),
                       ("Dwell / duration", r.dwell),
                       ("Video reference",
                        f"annotated.mp4 @ {fmt_ts(r.video_time_s)}" if pd.notna(r.video_time_s) else "—")]
            if r.source == "hazard" and run.hazards is not None:
                h = run.hazards.set_index("event_id").loc[r.incident_id]
                details += [("Max score", f"{h.max_score:.2f}"), ("Confirmed at", fmt_ts(h.confirmed_time_s)),
                            ("Email", h.email_status), ("Email detail", h.email_detail)]
            if r.source == "zone":
                e = run.events.set_index("event_id").loc[r.incident_id]
                details += [("Event type", e.event_type),
                            ("Exit reason", e.exit_reason if pd.notna(e.exit_reason) else "—"),
                            ("Confidence", f"{e.confidence:.2f}"), ("People in zone", f"{e.people_in_zone}")]
            st.markdown(ui.key_values(details), unsafe_allow_html=True)
            if img_path and img_path.exists():
                st.markdown("")
                st.download_button("Download evidence frame", data=img_path.read_bytes(), file_name=img_path.name,
                                   mime="image/jpeg", icon=":material/image:", width="stretch",
                                   key=f"dl_{r.incident_id}")
        with ui.card("review"):
            _review_panel(run, r)

    st.markdown("")
    left, right = st.columns([7, 5], gap="medium")
    with left:
        with ui.card("evt-chart"):
            bin_s = choose_bin_seconds(s["processing"]["analyzed_duration_s"])
            ui.section("Restricted-zone intrusions over time", f"{bin_s}-second intervals · present-at-start excluded")
            st.plotly_chart(charts.events_chart(interval_activity(run.frames, run.events, bin_s), bin_s, height=250),
                            config=PLOT_CONFIG, key="se_chart")
    with right:
        _exports(run, cfg, incidents, shown)

    if run.ppe_active and len(run.ppe):
        st.markdown("")
        _ppe_results(run)
    if run.hazard_active and run.alerts is not None:
        st.markdown("")
        with ui.card("alert-log"):
            ui.section("Email alert delivery log", "alerts.csv in the run folder · every attempt is recorded")
            if run.alerts.empty:
                st.caption("No fire/smoke event was confirmed, so no email was attempted.")
            else:
                st.dataframe(run.alerts, hide_index=True)
