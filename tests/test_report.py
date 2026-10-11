"""Report and data package, generated from a real (scripted-detector) pipeline run."""
import io
import json
import re
import zipfile

import pandas as pd

from forgesight import incidents as inc
from forgesight import report
from forgesight.config import Settings
from test_ppe import ppe_run  # noqa: F401  (module-scoped fixture: pipeline run with PPE beta)

CFG = Settings(occ_alert_scope="frame", occ_alert_threshold=3, occ_alert_min_s=2.0)


def test_report_contains_every_incident_and_key_facts(ppe_run):  # noqa: F811
    run = ppe_run
    incidents = inc.build_incidents(run, CFG)
    html = report.build_report_html(run, CFG, incidents)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    for iid in incidents["incident_id"]:
        assert iid in html
    assert run.summary["original_name"] in html
    assert "PPE checks" in html and "BETA" in html
    assert "not a certified industrial safety system" in html
    # self-contained: plotly bundled inline, images embedded, nothing fetched from the network
    assert "plotly" in html.lower()
    assert not re.search(r"<script[^>]+src=[\"']https?://", html)
    assert html.count("data:image/jpeg;base64,") >= 1 + int(incidents["evidence_crop"].astype(bool).sum())


def test_findings_are_derived_from_data(ppe_run):  # noqa: F811
    incidents = inc.build_incidents(ppe_run, CFG)
    alerts = inc.occupancy_alerts(ppe_run, CFG)
    text = " ".join(report.key_findings(ppe_run, incidents, alerts))
    assert f"{ppe_run.metrics['zone_entries']} restricted-zone intrusion" in text
    assert f"{len(incidents)} incident(s) in total" in text


def test_data_package(ppe_run):  # noqa: F811
    run = ppe_run
    incidents = inc.build_incidents(run, CFG)
    inc.save_review(run.run_dir, [incidents.iloc[0]["review_key"]], inc.REVIEWED, "checked")
    incidents = inc.build_incidents(run, CFG)
    data = report.build_data_package(run, CFG, incidents, "<html></html>")
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"report.html", "incidents.csv", "zone_events.csv", "summary.json", "frames.csv", "ppe.csv",
            "review.json", "review_settings.json"} <= names
    evidence = {n for n in names if n.startswith("evidence/")}
    assert len(evidence) == len(list((run.run_dir / "evidence").glob("*.jpg")))
    exported = pd.read_csv(z.open("incidents.csv"))
    assert list(exported["incident_id"]) == list(incidents["incident_id"])
    assert exported.iloc[0]["status"] == inc.REVIEWED and exported.iloc[0]["note"] == "checked"
    assert json.loads(z.read("review_settings.json"))["occ_alert_threshold"] == 3
