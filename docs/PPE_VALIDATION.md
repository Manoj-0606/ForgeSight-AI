# PPE beta: model choice and validation

## Why this model

No official Ultralytics PPE model exists. Community models were screened for: ONNX weights (PyTorch is
blocked by Windows Smart App Control on the development machine, and ONNX files cannot execute code when
loaded), explicit hard-hat and vest classes, CPU-sized architecture, and a stated licence.

Selected: [`mancalazure/construction-ppe-safety`](https://huggingface.co/mancalazure/construction-ppe-safety),
pinned to revision `70cc801710f246765f70ada7248d9560ec749bb2`, SHA-256 verified on download.

- YOLOv8n, fine-tuned on Roboflow *Construction Site Safety* v28 (CC BY 4.0 data). The weights carry
  Ultralytics' AGPL-3.0 licence.
- Classes used: `Hardhat`, `Safety Vest`. Indices are read from the model's metadata, never assumed.
- About 110 ms per check on a laptop CPU. ForgeSight runs it at about 2 checks per second, not on every analyzed frame.
- The model author publishes no validation metrics, so they were measured here.

## 1. Detection level: single images

Ultralytics *Construction-PPE* dataset, test split (141 images; independent of the training data).
IoU ≥ 0.5, confidence ≥ 0.40, letterboxed input.

| Class | Precision | Recall | AP50 |
|---|---|---|---|
| Hard hat | 0.83 | 0.72 | 0.72 |
| Hi-vis vest | 0.86 | 0.60 | 0.66 |
| **NO-Hardhat** (model's "missing" class) | **0.20** | **0.23** | **0.11** |
| Person | 0.70 | 0.70 | 0.64 |

**Decision:** the `NO-Hardhat` class is not used. "Missing PPE" is only ever stated as *not detected while
the person was clearly visible in several checks*.

## 2. Negative clip: nobody wears PPE

`people-walking.mp4` sample, 28 frames, about 560 person instances:

| Confidence | False hard hats | False vests |
|---|---|---|
| ≥ 0.25 | 75 | 4 |
| ≥ 0.40 | 27 | 1 |
| ≥ 0.55 | 4 | 0 |

That is why the default PPE confidence is 0.50 and a status needs agreement across several checks.

## 3. Person level: what the app actually reports

`scripts/validate_ppe.py` turns the 133 test images containing people into a slideshow video (3 s per image,
3 s blank gap so tracks end). It runs the **complete pipeline** (YOLO11 people, ByteTrack, PPE checks at
2 per second, voting) and compares each tracked person's final status with the labels.

| Hard hat | Detected | Not detected | No result* | Total |
|---|---|---|---|---|
| Wearing | **115** | 18 | 26 | 159 |
| Not wearing | 2 | **32** | 6 | 40 |

| Hi-vis vest | Detected | Not detected | Uncertain | No result* | Total |
|---|---|---|---|---|---|
| Wearing | **108** | 23 | 1 | 35 | 167 |
| Not wearing† | 6 | **46** | 0 | 17 | 69 |

\* Person not detected or tracked by YOLO11, or not judgeable (head outside the frame, too small).
† The dataset has no "no vest" class: "not wearing" means no vest was labelled on that person.

**Reading the numbers**

- "Detected" is reliable: 98% (hard hat) and 95% (vest) of those statuses are correct.
- "Not detected" is a lead, not a finding: 64% (hard hat) and 67% (vest) are correct. Roughly one in three
  would be a false alarm, so every "not detected" incident is Medium severity and labelled *verify*.
- 16–22% of labelled people get no PPE result at all, mostly because YOLO11 does not detect partially visible
  or occluded people in these construction photos.

## Limitations

- **Domain:** construction-site photos. Automotive plants differ: bump caps instead of hard hats in many areas,
  vests mainly in logistics and forklift zones, other lighting, uniforms and camera angles. **Validate on site
  footage before relying on it.**
- Static images inflate temporal stability. On real video, motion blur and pose changes add noise, though
  multi-check voting helps.
- Gloves, glasses, safety shoes and bump caps are not covered.

## Reproduce

```powershell
# dataset: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip (178 MB, AGPL-3.0)
.venv\Scripts\python scripts\validate_ppe.py path\to\construction-ppe --split test
```

Takes about 12 minutes on a laptop CPU.
