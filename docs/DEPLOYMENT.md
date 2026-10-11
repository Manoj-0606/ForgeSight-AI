# Deploying ForgeSight AI (Streamlit Community Cloud)

These notes describe how the app was prepared and tested for hosting. Nothing here deploys automatically.

## What was checked locally

- `requirements.txt` resolves to prebuilt Linux wheels for Python 3.11 and 3.12 (no source builds, no PyTorch).
- OpenCV is the **headless** build (`opencv-python-headless`), so no `libGL` system package is needed.
  No `packages.txt` is required.
- The app was run in a `python:3.11-slim-bookworm` (Debian 12) container limited to 2 CPUs and 2.7 GB RAM:
  all tests passed, every page loaded, and a sample clip was uploaded and analyzed through the browser.
- Model files are not in the repository. On first use the app downloads them into `models/` and checks each file's
  SHA-256: `yolo11n.onnx` (11 MB, GitHub/Ultralytics), the PPE model (12 MB) and the fire model (87 MB, both from
  Hugging Face). The server needs outbound internet access for that, which Community Cloud provides.

## Deploy settings

| Setting | Value |
|---|---|
| Main file | `app.py` |
| Python version (Advanced settings) | **3.11** (the version the app was developed and tested with) |
| Dependencies | `requirements.txt` (picked up automatically) |
| Secrets (optional) | only needed for email alerts, see below |

## Email alerts (optional)

The app starts and works without any email settings. Settings then shows "Email alerts are not configured".
To enable alerts, paste this into the app's **Secrets** box on Community Cloud. Don't put it in any file in the
repository:

```toml
[email]
host = "smtp.gmail.com"
port = 587
security = "starttls"
username = "your.address@gmail.com"
password = "your-16-character-app-password"
sender = "your.address@gmail.com"
recipients = ["you@example.com"]
```

## Things to know before sharing the link

- **Everyone shares one app instance.** Settings, saved analyses, uploaded videos and review notes are stored in
  `outputs/` on the server and are visible to every visitor. Don't upload private or confidential footage to a
  public deployment.
- **Email alerts on a public app** can be triggered by anyone, through *Send test email* or by analyzing a fire
  clip. Either leave email unconfigured on the public app, or make the app private (Community Cloud lets you
  restrict viewers to specific email addresses) and invite the interviewer.
- **Storage is temporary.** `outputs/` and downloaded models are wiped when the app restarts or goes to sleep. The
  first analysis after a restart downloads the person model again.
- **Speed.** Hosted CPUs are slower than a laptop. In the 2-CPU test, an 11-second clip took about a minute with PPE
  and fire checks on, and about 30 seconds with them off. Use short clips, and leave PPE and fire off unless you
  are showing them.
- **Uploads** are limited to 200 MB (`.streamlit/config.toml`), because uploaded files are held in memory.
- **Sleep.** Community Cloud apps sleep after a period without visitors. Open the link yourself shortly before the
  interview so it is awake.
