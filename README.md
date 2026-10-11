# ForgeSight AI — Industrial Vision & Safety Intelligence

ForgeSight AI is a Streamlit app that analyzes recorded factory videos for safety and workforce insights. It
detects and tracks people, watches a restricted area you draw on the video (like a robot cell), and turns what it
sees into charts, safety events with evidence images, and reports you can download.

I built it as a portfolio project to explore how computer vision could support shop-floor safety in automotive
manufacturing. Reviewing hours of CCTV by hand is slow. This prototype tries to answer simple questions from a
video: how many people were there, where did they spend time, and did anyone step into an area they shouldn't?

Everything runs locally on a normal laptop CPU. No GPU or cloud service is needed.

## Demo video

▶️ **[Watch the ForgeSight AI walkthrough](https://drive.google.com/file/d/1J9FDnoG2JbHIJxR4PsCqQIdUeMXp7nmH/view?usp=sharing)** (Google Drive)

The video is hosted on Google Drive. Depending on its sharing settings, you may need to request access to view it.

## Features

- **Video upload.** Upload a recorded MP4, MOV, AVI or MKV file. The app checks the file and shows a preview first.
- **Person detection.** Uses YOLO11n (the official Ultralytics model, exported to ONNX) and runs it with ONNX
  Runtime on the CPU. To keep it fast, it analyzes a few frames per second instead of every frame.
- **Person tracking.** ByteTrack gives each person an ID that stays the same from frame to frame, so the app can
  count people and follow their movement.
- **Restricted-zone monitoring.** Draw a zone using presets, a rectangle or a list of points. A person counts as
  inside when the bottom-center of their box (roughly their feet) is in the zone. Entries are confirmed over a few
  frames so a single noisy detection doesn't trigger an alert.
- **Safety events with evidence.** Each zone entry saves a timestamp, the track ID, how long the person stayed,
  and an evidence image. People who were already in the zone when the video started are listed separately, so
  they aren't counted as intrusions.
- **Charts and heatmaps.** Occupancy over time, activity per interval, sustained-occupancy alerts (for example,
  3 or more people in the zone for 3 seconds), and heatmaps of where people walked and where they stood still.
- **Incident review.** All events in one list, with filters, a severity level based on simple rules you can
  change, and an Open/Reviewed status with notes.
- **Exports and reports.** Download the event and incident lists as CSV, an HTML report that works offline, or a
  ZIP with all the data and evidence images.
- **PPE detection (optional, experimental).** Checks for hard hats and hi-vis vests on tracked people with a
  separate PPE model. It reports "detected", "not detected (please verify)" or "uncertain". Off by default.
- **Fire/smoke detection and email alerts (optional, experimental).** A separate image classifier checks frames
  for fire. When fire is confirmed over several checks, the app saves an evidence image and can send an email
  with the image attached. Smoke detection is unreliable with this model and is off by default.

## How it works

1. **Upload** a video on the *Video Intelligence* page and set up the restricted zone.
2. **Run the analysis.** The app reads a few frames per second and runs the person model on each one.
3. **Track and check zones.** ByteTrack links detections into tracks. Each track's foot position is checked
   against the zone to find entries and exits.
4. **Optional checks.** If enabled, the PPE and fire models run on some of the frames too.
5. **Save the run.** The app writes an annotated video (boxes, IDs, zone and alerts drawn on it), CSV files,
   heatmaps and evidence images to a folder under `outputs/runs/`.
6. **Review.** The *Command Center* page shows the overview, the *Safety Events* page is where you review
   incidents, and you can export a report. Earlier runs can be reopened from the sidebar.

## Tech stack

- **Python 3.11**
- **Streamlit**: web interface
- **YOLO11n (ONNX)** and **ONNX Runtime**: person detection on the CPU
- **ByteTrack** (via the `supervision` package): tracking
- **OpenCV**: video reading and drawing
- **NumPy** and **Pandas**: data handling
- **Plotly**: interactive charts
- **imageio-ffmpeg**: writes the annotated video as H.264 so it plays in the browser
- **pytest**: tests

PyTorch is not used. The models run as ONNX files, which keeps the install small and avoids PyTorch DLL problems
on Windows.

## Run locally (Windows)

You need Python 3.11. Open PowerShell in the project folder and run:

```powershell
# 1. Create a virtual environment
py -3.11 -m venv .venv

# 2. Install the dependencies
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

# 3. Start the app
.venv\Scripts\python -m streamlit run app.py
```

The app opens at <http://localhost:8501>.

Calling `.venv\Scripts\python` directly means you don't need to activate the environment. If you prefer to activate
it, run `.venv\Scripts\Activate.ps1`. If PowerShell blocks that script, the direct commands above still work.

The model files aren't stored in the repository. The person model (`yolo11n.onnx`) downloads into `models/` the
first time you run an analysis. The PPE and fire models download the first time you enable those features. You
need an internet connection for those first downloads.

To try it out, use the clips in `sample_videos/`. Their sources and licences are listed in
`sample_videos/SOURCES.md`.

**Optional checks:**

```powershell
# Check that libraries, models and sample videos are ready
.venv\Scripts\python scripts\check_environment.py

# Run the tests (installs pytest first)
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q tests
```

Notes for hosting the app online (Streamlit Community Cloud) are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

On Windows 11 with Smart App Control turned on, Windows can occasionally block an unsigned Python library file. It
checks unknown files online, so it's most likely to happen while your network is offline or reconnecting. If the
app reports a blocked file, reconnect, run `scripts\check_environment.py`, and start the app again.

## Email alert setup (optional)

Email alerts are sent only for confirmed fire/smoke events, and only if you turn them on in **Settings**.

For Gmail, you need a **Google App Password**, not your normal Gmail password. App Passwords require 2-Step
Verification on your Google account. You can create one at <https://myaccount.google.com/apppasswords>
([Google's help page](https://support.google.com/accounts/answer/185833)).

Set the details as environment variables **in the same PowerShell window** you use to start the app:

```powershell
$env:FORGESIGHT_SMTP_HOST="smtp.gmail.com"
$env:FORGESIGHT_SMTP_PORT="587"
$env:FORGESIGHT_SMTP_USER="your.address@gmail.com"
$env:FORGESIGHT_SMTP_PASSWORD="your-16-character-app-password"
$env:FORGESIGHT_ALERT_FROM="your.address@gmail.com"
$env:FORGESIGHT_ALERT_TO="recipient@example.com"
.venv\Scripts\python -m streamlit run app.py
```

Then open **Settings → Email alerts**, turn alerts on, and click **Send test email**.

You can also copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. That file is
already listed in `.gitignore`.

> **Never commit email credentials to GitHub.** Keep passwords in environment variables or in the git-ignored
> `secrets.toml`, never in the code or in a committed file.

## Limitations

- **This is a prototype.** It is not a certified industrial safety system and it is not a fire-alarm system. It
  must not replace real safety equipment such as light curtains, interlocks, or smoke and heat detectors.
- **Detections can be wrong.** The models can miss people, PPE or fire. They can also raise false alerts, for
  example when people are partly hidden behind machines, or when welding, furnaces or warm lighting look like
  fire. Every event should be checked by a person.
- **Recorded video only.** It analyzes uploaded files. There is no live camera (CCTV) connection.
- **Camera setup matters.** Zone monitoring works best with a fixed camera where people's feet are visible. If
  feet are hidden, the estimated position can be off.
- **Not validated on real factory footage.** The PPE and fire models were trained on other kinds of images,
  mostly construction sites and general fire photos. The checks I ran are described in [`docs/`](docs/), but they
  don't prove the models work in a real plant.
- **Model licences.** The YOLO11 and PPE models are released under AGPL-3.0, which matters if you want to use
  this commercially.

## Project structure

```
app.py               Streamlit entry point (page setup, navigation, sidebar)
forgesight/          Core logic: video reading, detection, tracking, zones, analytics,
                     PPE and fire checks, email alerts, incidents and reports
views/               The app pages: Command Center, Video Intelligence, Safety Events, Settings
scripts/             Environment check, command-line test run, model validation scripts
tests/               Automated tests (pytest)
docs/                Notes on how the PPE and fire models were chosen and checked
sample_videos/       Short test clips; sources and licences are listed in SOURCES.md
.streamlit/          Streamlit theme and an example secrets file
requirements.txt     Python dependencies (requirements-dev.txt adds pytest)
models/, outputs/    Created when the app runs (downloaded models, saved analyses); not in git
```

## Author

**Manoj J**
