# FloodResQ AI v0.3.0 - Validation report

**Status: NOT validated on real data.** No precision, recall, F1 or IoU exists for any real flood. Do not quote accuracy.

## 1. Automated tests that were actually run (Python 3.12, numpy, scipy, Pillow, matplotlib, Flask; no internet)
Command: `python -m unittest discover -s tests -p "test_*.py"` -> **57 tests, all passed**. (First run: 4 failed; two were real bugs
(a `%` formatting crash in the map header panel, and the claim checker rejecting "not confirmed damaged"), two were wrong tests; all fixed.)

| File | What it checks |
|---|---|
| `test_detection.py` | synthetic radar: flood found, no-flood gives ~nothing, permanent water is not new flood, no-data never flagged, steep ground excluded, masked town not flagged, blob counts reported |
| `test_evidence.py` | verdict rules incl. the real run's numbers -> inconclusive; cloud-covered water cannot be contradicted; never "high"/validated; `s2_support` ignores no-data; claim/number checker |
| `test_results_data.py` | the shipped `results.json` schema, not simulated, not validated; old layout detected; invalid values rejected; dashboard and results are the same run (time, water area, dates, layer names, identical overlay bounds); Sentinel-2 no-data is transparent |
| `test_imagery.py` | NaN/no-data transparency, block-average shrink, output size, raster masks |
| `test_api.py` | JSON (never HTML) for bad JSON, empty/long/array bodies, 404/405, rate limit; missing/corrupt/wrong-schema `results.json`; missing `dashboard.html`; no model key; model failure; invented numbers and over-claims rejected; API key never appears in served pages; `/api/status` |

## 2. What was NOT tested
* Colab pipeline end to end, Copernicus login/downloads, DEM download, Overpass (no internet/credentials here).
* `make_dashboard` (needs `folium`, `rasterio`; not installed in the test environment). Its v0.3.0 edits were only syntax-checked.
* A real Gemini call (no key here); Render deployment; the browser rendering of the page (no browser run).
* Nepali wording quality (needs a native reviewer).
* The earlier 60-test suite / benchmark mentioned in v0.2.0 documents was not in the files supplied, so it could not be re-run and is not claimed.

## 3. Findings about the stored run (from `dashboard.html` and `results.json`, not from re-processing imagery)
* Radar water 0.05 km2 (about 55 px) in 6 separate blobs in the southern part of the box; noise check flagged 0.01 km2 (about 11 px).
* Cloud-free Sentinel-2 covered 100% of that radar water and classified none as water; optical "new water" 0.01 km2.
* Therefore the candidate is unconfirmed. It is not proven false either: muddy water, mixed acquisition dates in the optical mosaic and 30 m pixels all limit the optical check.
* Verdict (v0.3.0 rules): **inconclusive**. The numbers for buildings, roads, bridges and settlements are conditional on those pixels being real (2 buildings, 0.62 km of road, 1 bridge, 0 cut-off settlements).

## 4. How to validate for real
1. Re-run the notebook with v0.3.0 (new numbers, same method); read the diagnosis table and the pair scan.
2. Look at the Esri satellite basemap along the river by eye and write down what you see.
3. Section 15 once, after freezing: download the EMSR927 flood extent and AOI yourself; it scores radar water only.
4. Fill in the table with the exact printed numbers (including bad ones):

| Product | AOI area (km2) | TP px | FP px | FN px | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|---|---|---|
| EMSR927 flood extent | | | | | | | | |

## 5. Prototype vs operational system
Student prototype: unvalidated thresholds, 30 m pixels, incomplete OSM, no ground truth, no field confirmation. It must not be used to decide where to send rescuers.
