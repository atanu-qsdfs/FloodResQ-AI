# FloodResQ AI v0.2.0 - Validation report

**Status: NOT validated on real data.** No precision, recall, F1 or IoU score exists for any real flood. Nothing below should be quoted as real-world accuracy.

## 1. What was actually run (author's sandbox, no internet)
Python 3.12.3, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, networkx 3.6.1, matplotlib 3.10.8, requests 2.33.1.

| Check | Result |
|---|---|
| Unit tests (`python -m unittest discover -s tests`) | **60 tests, all passed** |
| Notebook glue test (`tests/e2e_notebook_glue.py`): notebook sections 8-16 on synthetic inputs | Passed (all cells ran; results, EN/NE report, dashboard (stub), ZIP) |
| Synthetic old-vs-new benchmark (`tests/benchmark_synthetic.py`, 5 random seeds) | See section 2 |

What the 60 unit tests cover: grid maths; DEM tile names (GLO-30/GLO-90) and tile selection; **DEM resampling alignment** (a linear elevation surface is reproduced exactly at every cell centre; nodata never becomes elevation; outside the source is NaN); slope; height above river; speckle filter (variance down, mean unbiased, invalid pixels ignored); detector (flood found, no-flood gives almost nothing, permanent water is not flood, steep or high ground excluded, brightness offset removed, built-up noise not flagged as change, sign-consistent change); evidence verdicts (detected / none / inconclusive, never "high" confidence); road network (newly cut-off vs already unconnected vs not on network, blocked length is the real flagged length, bridges sampled along their whole line, routes); results schema; EN/NE report content; every number in the reports exists in `results.json`; copilot rejects invented numbers and over-claims; freeze/verify; metrics (NaN when undefined); scene-pair logic by track AND direction; Overpass failover, caching, snapshot-date verification, bounded waiting.

## 2. Synthetic benchmark (invented data: gamma speckle, an invented river, flood and town)
Mean of 5 seeds. "old" = the v0.1 detector copied from your original code; "new" = v0.2.

| Scenario | Metric | old | new |
|---|---|---|---|
| Wide flood (about 300 m wide) | water recall / precision | 0.79 / 1.00 | 1.00 / 0.97 |
| Narrow flooded channel (about 2 pixels) | water recall / precision | 0.32 / 1.00 | 1.00 / 0.93 |
| No flood | false water (km2) | 0.002 | 0.018 |
| Bright, variable town (all scenarios) | false "change" inside the town (km2) | 0.23-0.27 | 0.00 |

Reading it honestly: it supports the explanation that the old 3x3 box average over 30 m pixels loses narrow water and that unmasked built-up areas produce blocky false change. It also shows the new detector has a **slightly higher false-water area on a flood-free scene** (0.018 vs 0.002 km2; this is below the 0.02 km2 evidence minimum, so the verdict stays "no_water_detected"). The synthetic water is far darker than the land (-22 vs -10 dB), which real scenes do not guarantee, so real recall will be lower.

## 3. What was NOT tested (needs you, in Colab)
* Copernicus catalogue search, login, Sentinel Hub downloads (including the optional `localIncidenceAngle` / `shadowMask` bands: if rejected, the code falls back and says so).
* The AWS DEM download and reading with rasterio (the resampling maths is tested; the file reading is not). Tile URLs follow the documented public bucket layout (`copernicus-dem-30m`, `copernicus-dem-90m`) but were not fetched.
* Overpass against the real servers (failover and timestamp logic are tested against a simulated server).
* Real folium rendering (a recording stub was used; the dashboard has never been opened in a browser by the author), geopandas/rasterio polygon export, and the EMSR927 evaluation code.
* Everything on your real Nepal data, including why the original water mask was empty (the notebook's diagnosis table (section 8b) tells you).
* Nepali wording quality (needs a native reviewer).

## 4. How to validate for real
1. Run the full notebook on your event. Check the verdict, the diagnosis table and the control comparison.
2. Look at the satellite basemap along the river by eye. Write down what you see.
3. Section 15 **once**, after freezing: download the EMSR927 flood-extent vector and AOI yourself; it scores radar **water only** inside the AOI and refuses to run if the frozen results changed.
4. Fill in the table (copy the numbers exactly as printed, including bad ones):

| Product | AOI area (km2) | TP px | FP px | FN px | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|---|---|---|
| EMSR927 flood extent | | | | | | | | |

5. Optionally repeat on a second, independently labelled event (for example a public Sentinel-1 flood dataset) with unchanged thresholds, and report both. One event is not enough to claim the method generalises.

## 5. Prototype vs operational system
This is a student prototype: unvalidated thresholds, 30 m pixels, incomplete OSM, no ground truth, no uncertainty calibration, no field confirmation, no security or reliability engineering. It must not be used to decide where to send rescuers.
