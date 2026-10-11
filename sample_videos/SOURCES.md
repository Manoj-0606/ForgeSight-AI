# Sample videos: sources, licence and suitability

Both clips are real footage from an **automobile assembly line** (Bangladesh), downloaded on 2026-10-10 as the
1280×720 H.264 renditions offered on Pexels. They are unmodified.

| File | Pexels page | Title on Pexels | Creator | Duration |
|---|---|---|---|---|
| `pexels_32305292_automobile_assembly_line_720p.mp4` | https://www.pexels.com/video/assembly-line-at-automobile-factory-in-bangladesh-32305292/ | Assembly Line at Automobile Factory in Bangladesh | emrulkausar (Pexels) | 11.0 s |
| `pexels_32305277_automobile_assembly_line_720p.mp4` | https://www.pexels.com/video/automobile-assembly-line-in-bangladesh-factory-32305277/ | Automobile Assembly Line in Bangladesh Factory | emrulkausar (Pexels) | 10.2 s |

SHA-256 (first 16 hex digits): 32305292 → `28a14769629d6add`, 32305277 → `92fec9ed84c2758e`.

## Licence

[Pexels License](https://www.pexels.com/license/): free to use and modify, attribution not required (credited
above anyway). The licence also says:

- Don't sell unaltered copies, and don't redistribute the clips on other stock platforms.
- **Don't portray identifiable people in a bad light or in a way that is offensive.**

The workers are identifiable. ForgeSight overlays such as "RESTRICTED ZONE ENTRY" or "PPE NOT DETECTED" on
real people could be read as presenting them negatively. Use these clips for internal or interview demos only,
explain that the overlays are synthetic test output, and don't publish annotated versions.

## What these clips can and cannot demonstrate (verified 2026-10-10)

| Capability | Result on these clips |
|---|---|
| Opens and processes in ForgeSight | ✅ Both, about 11–13 analysed frames/s on CPU |
| Person detection and tracking on automotive footage | ✅ 32305292: 9 tracks, peak 6 people · 32305277: 14 tracks, peak 9 people |
| Camera stability | ⚠️ Steady handheld, not a fixed CCTV mount (median global shift ~1–2 px per 0.5 s). Fine for short clips. |
| **Restricted-zone intrusion** | ❌ **Not demonstrable.** Both clips are close shots of people working at their stations. Nobody walks across a visible floor boundary, and most feet are hidden by the line. |
| Fire / smoke | Negative footage only (no fire): used to measure false alarms |

**Zone test performed:** zone on the bottom-left hazard-striped floor area (0–34% x, 74–100% y).

- 32305292: 0 intrusions, 1 "present at start". That is the foreground worker, whose box is cut off by the frame
  bottom, not a real occupant of the floor zone.
- 32305277: 1 "intrusion" (EVT-002, ID 7 at 3.2 s). **On inspection this is false:** the worker stands behind
  the conveyor, but his lower body is hidden, so his box extends to the frame bottom and the estimated foot point
  falls in the zone. This is the documented foot-point limitation of a person detector on occluded people.

**Conclusion:** no intrusion was demonstrated with this footage. To demonstrate zone monitoring credibly, use a
fixed, elevated camera with workers' feet visible as they walk into a marked area, for example your own recording
from a tripod, or a licensed stock clip of a walkway or robot-cell perimeter.

## Fire / smoke test clips (`fire_tests/`)

Downloaded 2026-10-10 from Pexels (Pexels License, as above), 1280×720, unmodified.

| File | Pexels page | Creator | Duration | Purpose / result |
|---|---|---|---|---|
| `pexels_5622766_house_fire_720p.mp4` | https://www.pexels.com/video/video-of-a-house-on-fire-5622766/ | katy-landers-3688183 | 65.3 s | Positive: fire detected and confirmed (3 episodes) |
| `pexels_35069292_industrial_furnace_720p.mp4` | https://www.pexels.com/video/furnace-fire-in-industrial-setting-35069292/ | diego-castro-calderon-531372879 | 11.3 s | Hard case: a legitimate process flame is flagged as fire (nuisance alarm) |
| `pexels_8965440_room_thick_smoke_720p.mp4` | https://www.pexels.com/video/a-man-carrying-a-fire-extinguisher-entering-a-room-with-thick-smoke-8965440/ | mikael-blomkvist | 6.3 s | Known miss: dense indoor smoke is not detected |

None of these shows an automotive plant. No licensed footage of a fire inside a car factory was found.
Validation details: `docs/FIRE_SMOKE_VALIDATION.md`.
