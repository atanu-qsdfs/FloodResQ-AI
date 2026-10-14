# FloodResQ AI v0.2.0 - Mapping flood damage from space (hackathon prototype)

IIT Mandi Multimodal AI Hackathon 2026, Track B. **Educational prototype. Not an operational emergency-response system.**
Results are unvalidated estimates; "potentially affected" never means "confirmed damaged".

## What it does
1. Finds a same-track, same-direction **Sentinel-1** before/after pair (plus an earlier "control" scene for noise).
2. Downloads terrain-flattened backscatter (VV, VH, and shadow/layover bands when available) cropped to your box.
3. Loads elevation from the **public Copernicus DEM (GLO-30, fallback GLO-90)**, so no Sentinel Hub DEM permission is needed.
4. Downloads **pre-event OpenStreetMap** (as of a snapshot date), verified by the server's data timestamp.
5. Detects **suspected open water** and separate **surface-change candidates** (increase / decrease), reports noise from the control pair, and decides **water_detected / no_water_detected / inconclusive**.
6. Intersects the flagged area with roads, bridges, buildings, settlements and hospitals; tests which settlements may lose their road link (two scenarios: likely / possible).
7. Writes `results.json` (the only source of numbers), an English + Nepali situation report, a copilot that answers only from those results, and an interactive dashboard.

## Files
| File | Purpose |
|---|---|
| `floodresq.py` | The library (all logic). Upload it to Colab; the web app imports it too. |
| `FloodResQ_AI_full.ipynb` | The Colab notebook. Run cells top to bottom. |
| `app.py` | The Flask web server (Render): serves `dashboard.html` and the checked Copilot at `/api/chat`. |
| `dashboard.html`, `results.json` | The interactive map and the numbers the Copilot may use (produced by the notebook). |
| `requirements.txt` | Dependencies of the web app only (the notebook installs its own extras in Colab). |
| `tests/` | Unit tests, synthetic benchmark, notebook glue test. |
| `VALIDATION_REPORT.md` | What was tested, what was not, and a table to fill with real scores. |

## Run it in Google Colab
1. Open `FloodResQ_AI_full.ipynb` in Colab (File, Upload notebook).
2. Run section 1. When asked, upload `floodresq.py` and wait for "100% done". The notebook deletes any old copy and checks the version is 0.2.0.
3. Edit `CONFIG` (dates, box). Keep the OSM snapshot before the event.
4. Section 3: type your Copernicus OAuth client id and secret (hidden). Create a new OAuth client if you see `401 invalid_client`.
5. Run sections 4 to 7 in order. The OpenStreetMap step can take 10-30 minutes; if it stops, run it again (it resumes from `osm_cache`).
6. Run sections 8 to 9. Read the diagnosis table and the verdict. **Inconclusive or "no water detected" are valid results.**
7. Run sections 10 to 14, download `dashboard.html` and `situation_report.html` (open from a normal folder, not inside a zip, in Chrome).
8. Section 15 (EMSR927 scoring) last and only once. Section 16 builds `floodresq_outputs.zip`.

Run the tests anywhere with Python: `python -m unittest discover -s tests -p "test_*.py"` (and `python tests/benchmark_synthetic.py`).

## Web app on Render
* Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
* Environment variable (Render, Environment tab): `GEMINI_API_KEY` (secret, never in GitHub). Optional: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`).
* Without a key, or if Gemini fails, the Copilot answers from `results.json` only and says so. `/api/chat` always replies with JSON.
* The Copilot reads both results layouts: the current one (with `evidence`) and the older software-0.1.x layout, for which it says the file comes from an earlier analysis. To get the current layout, re-run the notebook and replace `results.json` and `dashboard.html`.
* Health check: `/healthz`.

## Architecture
```
Copernicus Data Space (catalogue + Sentinel Hub Process API)   Copernicus DEM (AWS Open Data)   OSM (Overpass, snapshot)
        |                                                              |                              |
 pairs, radar (VV, VH, shadow, incidence)                    resample_to_grid (EPSG:4326)       rivers, roads, buildings, places
        |                                                              |                              |
        +------------ detect_flood: Lee filter, bias removal, noise from control, HAND, slope, built-up mask -------------+
                                           |
                       assess_evidence (verdict, reasons, confidence)
                                           |
        hazards -> edge_hazard / access_scenarios / bridge_damage / building_damage
                                           |
                       build_results -> results.json -> reports (EN/NE), copilot, dashboard
```

## Method (short)
* **Preprocessing.** Sentinel Hub orthorectifies with the Copernicus 30 m DEM and applies radiometric terrain flattening (gamma0). We do not filter speckle there; we apply an edge-preserving Lee filter (ENL 4.4 for GRDH) with a normalised mean that ignores invalid pixels. A scene-wide brightness offset (rain, soil moisture, calibration) is measured far from rivers and removed.
* **Reliability.** Pixels in layover/shadow or with local incidence angle outside 15-65 degrees are ignored when those bands are available (and this is reported when they are not).
* **Open water.** Post VV < -18 dB and VH < -24 dB, not already dark before (pre VV > -16 dB), VV dropped at least 3 dB, slope <= 12 degrees, approximate height above the nearest mapped river <= 15 m, within 1000 m of a mapped river, hysteresis growth from seeds, and at least 4 connected pixels. These are literature-style starting values, **not tuned or validated for this event**.
* **Surface change.** VV and VH change in the same direction by more than max(3 dB, 3 x noise); noise is the robust standard deviation of the control pair. Built-up pixels (dense OSM buildings) and pixels that changed strongly even in the control pair are excluded and reported as "not assessed" (not as "safe"). Water is **not** excluded in built-up areas.
* **Control.** The same detector on two pre-event scenes estimates the false-alarm level. If it flags about as much "water" as the event, the verdict is inconclusive.
* **Evidence.** Inconclusive if corridor coverage < 70%, noise > 2.5 dB, scene coverage < 90%, or the control is not clearly lower. Confidence is never above "medium". Sentinel-2 (cloud-free pixels only) can support or contradict the radar water.
* **DEM.** GLO-30 (30 m) first, else GLO-90 (90 m). The native resolution is stored in the results. Resampling only interpolates; it never adds detail.
* **OSM.** Cached, patient retries, server rotation. A server's reply is accepted only if its data timestamp is not newer than the snapshot date.
* **Accessibility.** Road graph from OSM (drivable classes incl. tracks). A road segment is blocked if >= 30 m of it lies in flagged pixels. Settlements are snapped to the nearest road node (<= 500 m). Targets are hospitals and towns/cities. Status: reachable / cut_off (newly disconnected) / unconnected_before_event / not_on_road_network. Two scenarios are reported: *likely* (suspected water only) and *possible* (any flagged pixel).

## What changed from v0.1 (root causes found by reading your code)
* **DEM 403:** the Sentinel Hub `dem` collection is not allowed for free accounts. Replaced by the public AWS Copernicus DEM.
* **Empty water mask:** water required 8 tests at once and used a 3x3 box average on 30 m pixels (a 90 m window), which smears narrow Himalayan channels into their surroundings; plus a 400 m corridor and a slope-only terrain test. Now: Lee filter, VV+VH, hysteresis, a wider water corridor with a height-above-river test, and a diagnosis table (`funnel`) that shows which test removes the pixels in YOUR data. I could not run your real data, so I cannot tell you which test removes them; the notebook shows it.
* **Rectangular blocks near built-up areas:** the old "debris" class flagged any change of at least 3 dB in VV and 2 dB in VH of either sign, with no noise estimate and no built-up handling; towns are bright and variable. The blocks themselves are simply 30 m pixels drawn as squares. Now: sign consistency, noise-based threshold, built-up and unstable masks, and separate increase/decrease layers.
* **Bridges (11 of 95):** only the end points of each bridge were checked and any radar change counted. Now the whole line is sampled and the label is "intersects flagged pixels".
* **Roads (5.73 km):** the whole length of a segment was counted if it touched one flagged pixel. Now the actual flagged length is reported.
* **Cut-off settlements (3 of 140):** one hazard definition only. Now two scenarios.
* **Notebook:** CHECK 4 stopped the run if nothing was flagged (forcing a result); now zero detections are allowed. The upload cell reused stale copies of the library; now the version is checked. Pair search ignored orbit direction; fixed.
* **Not changed on purpose:** the radar scene time window (60 s before, 90 s after the scene start) because it worked for you; it is now a setting (`scene_pad_s`). Check the scene footprint coverage that the notebook prints.

## Known limits (please read)
* Live services (Copernicus catalogue/Process API, the AWS DEM download, Overpass) were **not** tested by the author. The requested terrain-geometry bands (`localIncidenceAngle`, `shadowMask`) are optional: if Sentinel Hub rejects them the notebook continues without them and says so.
* No validation against reference flood maps has been run. Do not quote precision, recall, F1 or IoU until you have scored a frozen run.
* The synthetic benchmark shows the code behaves sensibly in controlled cases only.
* SAR cannot see flooding under dense vegetation, in very narrow channels, or in most built-up areas, and monsoon rain and shadow cause false flags.
* Nepali text must be reviewed by a native speaker.

## Data and attribution
Contains modified Copernicus Sentinel data 2026. Copernicus DEM: Produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. (c) OpenStreetMap contributors. Basemaps: Esri / OpenTopoMap (CC-BY-SA). Copernicus EMS EMSR927 is used only for after-the-fact checking.
