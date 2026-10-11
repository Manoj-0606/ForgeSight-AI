# Fire / smoke beta: model choice, validation and limitations

All numbers below were measured on this project's CPU laptop on 2026-10-10 with
`scripts/validate_fire.py` and the scratch evaluation described here. Nothing is copied from model cards.

## Shipped model

| | |
|---|---|
| Model | [`prithivMLmods/Fire-Detection-Engine-ONNX`](https://huggingface.co/prithivMLmods/Fire-Detection-Engine-ONNX), file `onnx/model_int8.onnx` |
| Pinned revision | `02bd7f981aac3e27a75f83e0a3b97dfadaffc228` |
| SHA-256 (verified on download) | `4917442f15fd7014ccf9cf9509b7dd6692db2dcc3aace56f03ae18808ced0785` |
| Licence | Apache-2.0 (model card) |
| Architecture / input | ViT-B/16 image **classifier**, 224×224 RGB, normalised to [-1, 1] |
| Classes (from `config.json` id2label) | 0 `Fire Needed Action`, 1 `Normal Conditions`, 2 `Smoky Environment` |
| Dependencies | `onnxruntime` and `opencv-python` (already required). No PyTorch. |
| CPU cost | about 0.2–0.35 s per check on this laptop. The app runs 2 checks/s by default. |
| Output | Frame-level probabilities. **No bounding boxes**, so evidence is the full frame with the scores. |

The model card reports 98% accuracy on the author's own validation split. That is not independent, so
it was not relied on.

## Candidates evaluated and rejected

| Model | Format / licence | Why rejected |
|---|---|---|
| `fiacecson20/cctv-ai-fire-smoke` (YOLOv8n detector, 320 px) | ONNX; Hub tag MIT, but the weights' own metadata says AGPL-3.0; training data undocumented | Independent D-Fire test AP50: fire **0.145**, smoke **0.092** (the card claims 0.91 / 0.57). It **missed the real house-fire video** entirely (max fire score 0.44) and dense indoor smoke (0.00). |
| `fireviewer/fire-smoke-yolo11m-strict-v1` | ONNX, AGPL-3.0 | The authors' own independent evaluation: flame recall **0.016**, false alarms on 34% of negative images. YOLO11m at 960 px is also slow on CPU. |
| `aryantyagiase/yolov8-fire-smoke-detection` | ONNX, **no licence** | All rights reserved by default, so it cannot be used legitimately. |
| Other community fire models | `.pt` only | PyTorch is blocked on this machine (Windows Smart App Control), so they could not be converted or run. |

## 1. Labelled images: D-Fire test split

D-Fire ([gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset), CC BY 4.0, mirror
`hanvithSai/gavin-dfire-raw`). The test split has 4,306 images and contains deliberately hard negatives
(sunsets, lamps, fog, clouds). A seeded random sample of 1,500 images was used. Image level means: does the
image contain the class?

| Fire, p ≥ | Precision | Recall |
|---|---|---|
| 0.40 | 0.59 | 0.75 |
| **0.45 (default)** | **0.64** | **0.72** |
| 0.50 | 0.69 | 0.67 |
| 0.55 | 0.75 | 0.64 |

| Smoke, p ≥ 0.45 | Precision 0.36 | Recall **0.08** |
|---|---|---|

False alarms (fire or smoke ≥ 0.45) on the 698 sampled images with neither: **29.8%**. These negatives
are hard by design: warm sunsets and lights look like fire to a frame classifier.

For comparison, the rejected YOLO detector at 0.45: fire precision 0.79, recall 0.39; smoke 0.57 / 0.24;
22% false alarms on all 2,005 negatives.

## 2. Real video, default app settings (2 checks/s, confirm after 3 consecutive, close after 4 clear)

| Clip (all Pexels licence, see `sample_videos/SOURCES.md`) | Checks ≥ 0.45 | Confirmed fire events |
|---|---|---|
| House on fire (65 s) | 61% | **3 episodes** (0.0–3.5 s, 9.0–44.5 s, 52.6–55.6 s) |
| Industrial furnace (11 s) | 100% | 1 episode: **a nuisance alarm** (a legitimate process flame) |
| Room with thick indoor smoke (6 s) | 0% | **none: missed** |
| 5 automotive assembly-line clips (~60 s total, 7 files incl. lower-res copies) | **0%** | **none** (max fire score 0.39) |

## 3. End-to-end with email (real models, real SMTP on localhost)

`scripts/smoke_test.py ... --hazard --email` was pointed at a local SMTP sink, and the Streamlit UI was
exercised in a browser:

- House fire, first 20 s: FIRE-001 confirmed at 1.0 s and **emailed** (delivered, evidence JPEG attached,
  1.3 s). FIRE-002 at 9.0 s was **suppressed (cooldown)**. Both are recorded in `alerts.csv` and shown in Safety Events.
- Settings → *Send test email*: delivered.
- Automotive clip: no event, no email.

## What is not verified

- **Real-world email delivery** through a corporate SMTP server (TLS, authentication, spam filtering). Only the
  local SMTP round trip and mocked failures were tested. Use *Send test email* after configuring credentials.
- Accuracy on **real factory fires**: no licensed footage of a fire inside an automotive plant was found.
- Behaviour with welding, grinding sparks, heat-treatment furnaces, orange-painted machinery or warm lighting.
  The furnace result suggests these will trigger alerts.
- **Smoke** detection: unreliable, and off by default.

## Limitations to state in a demo

- This is a frame classifier for recorded video. It is **not** a certified fire detection or alarm system and must
  not replace smoke/heat detectors.
- About 30% false alarms on hard negative images. Consecutive-check confirmation removes flicker, but not a
  static scene that looks like fire.
- Process flames are detected as fire. A deployment would need exclusion areas or a site-specific model.
- Cooldown is measured in video time, per hazard type. At most `email_max_per_run` emails are sent per analysis.
