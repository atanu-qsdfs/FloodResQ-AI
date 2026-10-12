"""
floodresq.py - helper library for FloodResQ AI (educational hackathon prototype).

IIT Mandi Multimodal AI Hackathon 2026, Track B: Mapping Flood Damage from Space.
NOT an operational emergency-response tool.

Data rules (from the problem statement):
  allowed inputs : Sentinel-1, Sentinel-2, Copernicus DEM, PRE-event OpenStreetMap
  checking only  : Copernicus EMS (EMSR927), UNOSAT, any published damage map

Design rules of this library
  * Nothing here contains or invents satellite results. Every number comes from data you download.
  * "Potentially affected" is never the same as "confirmed damaged".
  * If the evidence is weak the library says "inconclusive" instead of forcing a flood result.
  * Everything that can be tested without internet is tested in tests/ (synthetic data only).
"""
import hashlib
import html
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

__version__ = "0.2.3"

logger = logging.getLogger("floodresq")


def setup_logging(path="floodresq.log", level=logging.INFO):
    """Log to the screen and to a file (the file goes into the submission ZIP)."""
    logger.setLevel(level)
    logger.propagate = False
    have_stream = any(getattr(h, "_frq", "") == "stream" for h in logger.handlers)
    have_file = any(getattr(h, "_frq", "") == "file" for h in logger.handlers)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if not have_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._frq = "stream"
        logger.addHandler(sh)
    if path and not have_file:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        fh._frq = "file"
        logger.addHandler(fh)
    return logger


ATTRIBUTION = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010\u20132014 and \u00a9 Airbus Defence and Space GmbH "
    "2014\u20132018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "\u00a9 OpenStreetMap contributors.",
]
CITATIONS = [
    "European Union, Copernicus Emergency Management Service data (EMSR927) - used for checking only.",
]

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
# Overpass servers, in order of preference. A reply is ACCEPTED ONLY IF the server states that its data
# is not newer than your OSM snapshot (see _overpass_query), so a server without history is rejected.
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OSM_CACHE_DIR = "osm_cache"
USER_AGENT = "FloodResQ-AI-hackathon/%s (student project; educational use)" % __version__

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0

_sleep = time.sleep  # replaced in tests


# --------------------------------------------------------------------------
# Grid: one common pixel grid shared by every raster layer (EPSG:4326, north-up)
# --------------------------------------------------------------------------
class Grid:
    """A north-up pixel grid covering a lon/lat bounding box (EPSG:4326)."""

    crs = "EPSG:4326"

    def __init__(self, bbox, width, height):
        w, s, e, n = [float(v) for v in bbox]
        if not (w < e and s < n):
            raise ValueError("bbox must be (west, south, east, north) with west<east and south<north")
        self.bbox = (w, s, e, n)
        self.west, self.south, self.east, self.north = self.bbox
        self.width = int(width)
        self.height = int(height)
        lat0 = math.radians((s + n) / 2.0)
        self.dx_m = (e - w) / self.width * M_PER_DEG_LON_EQ * math.cos(lat0)
        self.dy_m = (n - s) / self.height * M_PER_DEG_LAT

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def pixel_area_km2(self):
        return self.dx_m * self.dy_m / 1e6

    def rowcol(self, lon, lat):
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        col = np.floor((lon - self.west) / (self.east - self.west) * self.width).astype(int)
        row = np.floor((self.north - lat) / (self.north - self.south) * self.height).astype(int)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return row, col, inside

    def centers(self):
        """1-D arrays: longitude of every column centre, latitude of every row centre."""
        lon = self.west + (np.arange(self.width) + 0.5) * (self.east - self.west) / self.width
        lat = self.north - (np.arange(self.height) + 0.5) * (self.north - self.south) / self.height
        return lon, lat

    def transform(self):
        from rasterio.transform import from_bounds
        return from_bounds(self.west, self.south, self.east, self.north, self.width, self.height)


def make_grid(bbox, res_m=30.0, max_px=2500):
    w, s, e, n = bbox
    lat0 = math.radians((s + n) / 2.0)
    width = int(math.ceil((e - w) * M_PER_DEG_LON_EQ * math.cos(lat0) / res_m))
    height = int(math.ceil((n - s) * M_PER_DEG_LAT / res_m))
    if max(width, height) > max_px:
        raise ValueError(
            "Grid would be %d x %d pixels; the Sentinel Hub limit is %d. Use a larger RES_M or a smaller BBOX."
            % (width, height, max_px))
    return Grid(bbox, width, height)


def sample_mask(grid, mask, lon, lat):
    """True/False for each lon/lat point: is it inside a True pixel of `mask`?"""
    r, c, inside = grid.rowcol(lon, lat)
    out = np.zeros(len(r), dtype=bool)
    out[inside] = mask[r[inside], c[inside]]
    return out


def area_km2(mask, grid):
    return float(np.count_nonzero(mask)) * grid.pixel_area_km2


def haversine_m(a, b):
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def _densify(x0, y0, x1, y1, step_m):
    length = haversine_m((x0, y0), (x1, y1))
    n = max(2, int(length / step_m) + 1)
    t = np.linspace(0.0, 1.0, n)
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


# --------------------------------------------------------------------------
# Copernicus Data Space: catalogue search, login, Sentinel Hub Process API
# --------------------------------------------------------------------------
def search_s1(bbox, start, end, top=100):
    """Search Sentinel-1 IW GRD scenes (catalogue only, no download)."""
    w, s, e, n = bbox
    poly = "POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s))" % (w, s, e, s, e, n, w, n, w, s)
    flt = (
        "Collection/Name eq 'SENTINEL-1' and "
        "OData.CSC.Intersects(area=geography'SRID=4326;%s') and "
        "ContentDate/Start gt %sT00:00:00.000Z and ContentDate/Start lt %sT23:59:59.000Z and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S')" % (poly, start, end)
    )
    params = {"$filter": flt, "$expand": "Attributes", "$top": top, "$orderby": "ContentDate/Start asc"}
    r = requests.get(CATALOGUE_URL, params=params, timeout=90)
    r.raise_for_status()
    return r.json().get("value", [])


def _attr(product, name):
    for a in product.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return None


def footprint_coverage(geojson_geometry, bbox):
    """Fraction of the study box covered by a scene footprint (None if it cannot be computed)."""
    try:
        from shapely.geometry import box, shape
        g = shape(geojson_geometry)
        b = box(*bbox)
        return float(g.intersection(b).area / b.area)
    except Exception:
        return None


def scenes_table(products, bbox=None):
    import pandas as pd
    rows = []
    for p in products:
        geo = p.get("GeoFootprint")
        rows.append({
            "name": p["Name"],
            "start": p["ContentDate"]["Start"][:19],
            "satellite": _attr(p, "platformSerialIdentifier"),
            "rel_orbit": _attr(p, "relativeOrbitNumber"),
            "direction": _attr(p, "orbitDirection"),
            "polarisation": _attr(p, "polarisationChannels"),
            "id": p["Id"],
            "cover": footprint_coverage(geo, bbox) if (geo and bbox) else None,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["start"]).dt.date
    return df


def find_pairs(scenes, event_date):
    """Latest scene before the event + earliest scene on/after it, per orbit track AND direction.
    Pairs from the same relative orbit and direction have the same viewing geometry, which is what
    makes a before/after backscatter comparison meaningful."""
    import pandas as pd
    pairs = []
    if len(scenes) == 0:
        return pd.DataFrame(pairs)
    for (orbit, direction), g in scenes.groupby(["rel_orbit", "direction"]):
        pre = g[g["date"] < event_date]
        post = g[g["date"] >= event_date]
        if pre.empty or post.empty:
            continue
        a = pre.sort_values("start").iloc[-1]
        b = post.sort_values("start").iloc[0]
        ca, cb = a.get("cover"), b.get("cover")
        cover = None if (ca is None or cb is None or ca != ca or cb != cb) else min(float(ca), float(cb))
        pairs.append({
            "rel_orbit": orbit, "direction": direction,
            "pre_scene": a["name"], "pre_date": a["date"], "pre_satellite": a["satellite"],
            "post_scene": b["name"], "post_date": b["date"], "post_satellite": b["satellite"],
            "gap_days": (b["date"] - a["date"]).days, "min_cover": cover,
        })
    return pd.DataFrame(pairs)


def pick_pair(pairs, index=None):
    """Choose a pair. index=None: best footprint coverage first, then earliest post-event scene."""
    if len(pairs) == 0:
        raise ValueError("No same-track before/after pairs. Widen SEARCH_DAYS or check BBOX and EVENT_DATE.")
    if index is not None:
        return pairs.iloc[int(index)]
    p = pairs.copy()
    p["_cov"] = p["min_cover"].fillna(0.0).round(2)
    p = p.sort_values(["_cov", "post_date"], ascending=[False, True])
    return p.drop(columns="_cov").iloc[0]


def check_pair(pair, event_date):
    """Hard errors for an invalid pair; returns a list of warning strings for a weak one."""
    if not (pair["pre_date"] < event_date <= pair["post_date"]):
        raise ValueError("The pair does not straddle the event date.")
    warns = []
    if pair["gap_days"] % 6 != 0 or pair["gap_days"] > 36:
        warns.append("Unusual gap of %d days between the two scenes (normal repeat cycles are multiples of 6)."
                     % pair["gap_days"])
    elif pair["gap_days"] > 24:
        warns.append("Long gap of %d days: changes unrelated to the flood become more likely." % pair["gap_days"])
    cov = pair.get("min_cover")
    if cov is not None and cov == cov and cov < 0.9:
        warns.append("A scene covers only %.0f%% of your BBOX." % (100 * cov))
    return warns


def find_control(scenes, pair, target_gap_days=12, tolerance_days=2):
    """A scene on the SAME track and direction about 12 days BEFORE the 'pre' scene (noise check)."""
    g = scenes[(scenes["rel_orbit"] == pair["rel_orbit"]) & (scenes["direction"] == pair["direction"])]
    best = None
    for _, row in g.iterrows():
        gap = (pair["pre_date"] - row["date"]).days
        if gap > 0 and abs(gap - target_gap_days) <= tolerance_days:
            if best is None or abs(gap - target_gap_days) < best[0]:
                best = (abs(gap - target_gap_days), row)
    return None if best is None else best[1]


def scene_window(name, pad_s=60, pad_after_s=90):
    """Time window around a scene, built from its file name."""
    m = re.search(r"_(\d{8}T\d{6})_", name)
    if not m:
        raise ValueError("Cannot read acquisition time from scene name: %s" % name)
    t = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (t - timedelta(seconds=pad_s)).strftime(fmt), (t + timedelta(seconds=pad_after_s)).strftime(fmt)


class CDSEClient:
    """Keeps a Copernicus Data Space login token fresh (tokens last only ~10 minutes)."""

    def __init__(self, client_id, client_secret):
        self._id = (client_id or "").strip()
        self._secret = (client_secret or "").strip()
        self._token = None
        self._t = 0.0

    def token(self):
        if self._token is None or time.time() - self._t > 420:
            r = requests.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials", "client_id": self._id, "client_secret": self._secret},
                timeout=60)
            if r.status_code != 200:
                raise RuntimeError("Login failed (HTTP %s): %s. Check the OAuth client id/secret "
                                   "(create a new client in the Copernicus dashboard if unsure)."
                                   % (r.status_code, r.text[:200]))
            self._token = r.json()["access_token"]
            self._t = time.time()
        return self._token


def _read_tiff(content):
    from rasterio.io import MemoryFile
    with MemoryFile(content) as mf:
        with mf.open() as ds:
            arr = ds.read()
    return arr.astype("float32")


def process_tiff(client, data_block, evalscript, grid, retries=3):
    payload = {
        "input": {
            "bounds": {"bbox": list(grid.bbox),
                       "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": data_block,
        },
        "output": {"width": grid.width, "height": grid.height,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript,
    }
    last = ""
    for attempt in range(retries):
        r = requests.post(
            PROCESS_URL,
            headers={"Authorization": "Bearer " + client.token(), "Content-Type": "application/json",
                     "Accept": "image/tiff"},
            json=payload, timeout=300)
        if r.status_code == 200:
            arr = _read_tiff(r.content)
            if arr.shape[1:] != grid.shape:
                raise RuntimeError("Unexpected image size %s, expected %s" % (arr.shape[1:], grid.shape))
            return arr
        last = "HTTP %s: %s" % (r.status_code, r.text[:400])
        if r.status_code in (429, 500, 502, 503, 504):
            _sleep(5 * (attempt + 1))
            continue
        break
    raise RuntimeError("Sentinel Hub request failed. " + last)


S1_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "dataMask"]}],
          output: {bands: 3, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask]; }
"""

# Same as above plus terrain-geometry bands (layover/shadow mask and local incidence angle).
S1_EVALSCRIPT_GEOM = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "dataMask", "localIncidenceAngle", "shadowMask"]}],
          output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask, s.localIncidenceAngle, s.shadowMask]; }
"""

S2_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B04", "B03", "B02", "SCL", "dataMask"]}],
          output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.B04, s.B03, s.B02, s.SCL, s.dataMask]; }
"""

# DEM_EVALSCRIPT / fetch_dem (Sentinel Hub "dem" collection) are intentionally NOT used any more:
# that collection returned HTTP 403 COMMON_INSUFFICIENT_PERMISSIONS for free accounts.


def fetch_s1_scene(client, grid, scene_name, orbit_direction=None, pad_s=60, pad_after_s=90, with_geometry=True):
    """Download one Sentinel-1 IW scene cropped to the grid.

    Processing is done by Sentinel Hub: orthorectification with the Copernicus 30 m DEM and radiometric
    terrain flattening (GAMMA0_TERRAIN), no speckle filter (we filter ourselves). Values are linear power.
    If the terrain-geometry bands are not available the request is repeated without them and
    result['geometry_available'] is False (the shadow/layover check is then skipped and reported)."""
    t0, t1 = scene_window(scene_name, pad_s, pad_after_s)
    flt = {"timeRange": {"from": t0, "to": t1}, "acquisitionMode": "IW", "polarization": "DV",
           "resolution": "HIGH"}
    if orbit_direction:
        flt["orbitDirection"] = str(orbit_direction)
    data = [{"type": "sentinel-1-grd", "dataFilter": flt,
             "processing": {"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True,
                            "demInstance": "COPERNICUS_30", "speckleFilter": {"type": "NONE"}}}]
    out = {"scene": scene_name, "window": (t0, t1), "geometry_available": False, "lia": None, "shadow": None}
    arr = None
    if with_geometry:
        try:
            arr = process_tiff(client, data, S1_EVALSCRIPT_GEOM, grid)
            lia = arr[3].copy()
            if np.isfinite(lia).any() and np.nanmax(lia) <= 6.4:  # radians -> degrees
                lia = np.degrees(lia)
            out.update(lia=lia, shadow=arr[4] > 0.5, geometry_available=True)
        except RuntimeError as ex:
            logger.warning("Terrain-geometry bands not available (%s); continuing without shadow/layover mask.",
                           str(ex)[:160])
            arr = None
    if arr is None:
        arr = process_tiff(client, data, S1_EVALSCRIPT, grid)
    out.update(vv=arr[0], vh=arr[1], mask=arr[2] > 0.5)
    return out


def fetch_s2_rgb(client, grid, t_from, t_to, max_cloud=70):
    """Sentinel-2 L2A true colour plus the scene-classification layer (SCL: 6 = water, 3/8/9/10 = cloud/shadow)."""
    data = [{"type": "sentinel-2-l2a",
             "dataFilter": {"timeRange": {"from": t_from, "to": t_to}, "maxCloudCoverage": max_cloud,
                            "mosaickingOrder": "leastCC"}}]
    arr = process_tiff(client, data, S2_EVALSCRIPT, grid)
    rgb = np.clip(np.stack([arr[0], arr[1], arr[2]], axis=-1) * 3.0, 0, 1)
    scl = arr[3]
    valid = arr[4] > 0.5
    cloudy = np.isin(scl, [3, 8, 9, 10]) & valid
    frac = float(cloudy.sum()) / max(1, int(valid.sum()))
    return {"rgb": rgb, "scl": scl, "cloud_fraction": frac, "valid": valid, "window": (t_from, t_to)}


def reliable_mask(radar, params=None):
    """Pixels whose radar value can be trusted: data present, finite, positive, not in layover/shadow,
    local incidence angle in a usable range (when the geometry bands exist)."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    ok = radar["mask"] & np.isfinite(radar["vv"]) & np.isfinite(radar["vh"]) & (radar["vv"] > 0) & (radar["vh"] > 0)
    if radar.get("shadow") is not None:
        ok &= ~radar["shadow"]
    lia = radar.get("lia")
    if lia is not None:
        with np.errstate(invalid="ignore"):
            ok &= (lia >= p["min_incidence"]) & (lia <= p["max_incidence"])
    return ok


def check_radar(radar, min_valid=0.5, params=None):
    """Quality facts for one radar scene; raises if the image is unusable."""
    vv = radar["vv"]
    ok = radar["mask"] & np.isfinite(vv) & (vv > 0)
    frac = float(ok.mean())
    if frac < min_valid:
        raise ValueError("Only %.0f%% of %s has valid data. The scene may not cover your BBOX." %
                         (100 * frac, radar["scene"]))
    rel = reliable_mask(radar, params)
    db = 10 * np.log10(vv[ok])
    vh_ok = ok & np.isfinite(radar["vh"]) & (radar["vh"] > 0)
    out = {"scene": radar["scene"], "valid_fraction": round(frac, 3),
           "reliable_fraction": round(float(rel.mean()), 3),
           "median_vv_db": round(float(np.median(db)), 2),
           "median_vh_db": round(float(np.median(10 * np.log10(radar["vh"][vh_ok]))), 2) if vh_ok.any() else None,
           "p1_vv_db": round(float(np.percentile(db, 1)), 2),
           "p99_vv_db": round(float(np.percentile(db, 99)), 2),
           "geometry_bands": bool(radar.get("geometry_available"))}
    if not (-25 < out["median_vv_db"] < -3):
        out["warning"] = "Median VV of %.1f dB is outside the usual -25..-3 dB range: check units/calibration." % \
                         out["median_vv_db"]
    return out


def valid_mask(*radars):
    ok = np.ones(radars[0]["vv"].shape, dtype=bool)
    for r in radars:
        ok &= reliable_mask(r)
    return ok


# --------------------------------------------------------------------------
# Terrain: public Copernicus DEM (no Sentinel Hub permission needed)
# --------------------------------------------------------------------------
DEM_SOURCES = {
    # name: (public AWS Open Data bucket, tile-name prefix, native pixel size in metres, label)
    "GLO30": ("copernicus-dem-30m", "Copernicus_DSM_COG_10", 30, "Copernicus GLO-30 DEM"),
    "GLO90": ("copernicus-dem-90m", "Copernicus_DSM_COG_30", 90, "Copernicus GLO-90 DEM"),
}


def dem_tile_name(lat, lon, source="GLO30"):
    """Tile name for the 1x1 degree tile whose SOUTH-WEST corner is (lat, lon)."""
    prefix = DEM_SOURCES[source][1]
    ns = ("N" if lat >= 0 else "S") + "%02d" % abs(int(lat))
    ew = ("E" if lon >= 0 else "W") + "%03d" % abs(int(lon))
    return "%s_%s_00_%s_00_DEM" % (prefix, ns, ew)


def dem_tile_url(lat, lon, source="GLO30"):
    bucket = DEM_SOURCES[source][0]
    name = dem_tile_name(lat, lon, source)
    return "https://%s.s3.amazonaws.com/%s/%s.tif" % (bucket, name, name)


def dem_tiles_for_bbox(bbox):
    """South-west corners of every 1x1 degree tile that overlaps the box (exact edges do not add a tile)."""
    w, s, e, n = bbox
    lats = range(int(math.floor(s)), int(math.ceil(n)))
    lons = range(int(math.floor(w)), int(math.ceil(e)))
    return [(la, lo) for la in lats for lo in lons]


def resample_to_grid(src, src_bounds, grid, nodata=None):
    """Bilinear resampling of a north-up lon/lat raster onto the analysis grid.

    src: 2-D array; src_bounds: (west, south, east, north) of the OUTER EDGES of the source raster.
    Both rasters are in EPSG:4326, so no re-projection is needed. Invalid source cells (nodata / NaN)
    are ignored (weights), never turned into fake elevations. Cells with no valid neighbours become NaN.
    Resampling only interpolates; it does NOT add terrain detail that the source does not contain."""
    from scipy import ndimage
    a = np.asarray(src, dtype="float64")
    valid = np.isfinite(a)
    if nodata is not None:
        valid &= a != nodata
    ws, ss, es, ns = src_bounds
    nrows, ncols = a.shape
    dxs = (es - ws) / ncols
    dys = (ns - ss) / nrows
    lon, lat = grid.centers()
    cols = (lon - ws) / dxs - 0.5
    rows = (ns - lat) / dys - 0.5
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    v0 = np.where(valid, a, 0.0)
    w0 = valid.astype("float64")
    num = ndimage.map_coordinates(v0, [rr, cc], order=1, mode="constant", cval=0.0)
    den = ndimage.map_coordinates(w0, [rr, cc], order=1, mode="constant", cval=0.0)
    out = np.full(grid.shape, np.nan, dtype="float32")
    good = den >= 0.5
    out[good] = (num[good] / den[good]).astype("float32")
    return out


def _download(url, dest, retries=4, timeout=180):
    """Download with retries; writes to a temporary file first so a broken download never poses as a tile."""
    last = ""
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as r:
                if r.status_code == 404:
                    raise FileNotFoundError("HTTP 404 (tile does not exist): " + url)
                if r.status_code != 200:
                    last = "HTTP %s" % r.status_code
                    raise RuntimeError(last)
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if os.path.getsize(tmp) < 1024:
                raise RuntimeError("downloaded file is too small")
            os.replace(tmp, dest)
            return dest
        except FileNotFoundError:
            raise
        except Exception as ex:
            last = "%s: %s" % (type(ex).__name__, ex)
            logger.warning("DEM download attempt %d/%d failed (%s)", attempt + 1, retries, last)
            _sleep(3 * (attempt + 1))
    raise RuntimeError("Could not download %s (%s)" % (url, last))


def _read_dem_tile(path):
    import rasterio
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError("%s is not in EPSG:4326 (found %s); refusing to guess." % (path, ds.crs))
        arr = ds.read(1).astype("float64")
        if ds.transform.e > 0:  # south-up raster: flip so row 0 is north
            arr = arr[::-1]
        b = ds.bounds
        nodata = ds.nodata
    return arr, (b.left, b.bottom, b.right, b.top), nodata


def check_dem(dem, grid, min_valid=0.98):
    """Sanity checks. Raises on an unusable DEM; returns the facts otherwise."""
    if dem.shape != grid.shape:
        raise ValueError("DEM shape %s does not match the grid %s" % (dem.shape, grid.shape))
    ok = np.isfinite(dem)
    frac = float(ok.mean())
    if frac < min_valid:
        raise ValueError("Only %.1f%% of the DEM covers the study area." % (100 * frac))
    lo, hi = float(np.nanmin(dem)), float(np.nanmax(dem))
    if lo < -500 or hi > 9000:
        raise ValueError("DEM values look wrong (%.0f to %.0f m)." % (lo, hi))
    return {"valid_fraction": round(frac, 4), "min_m": round(lo, 1), "max_m": round(hi, 1)}


def fetch_dem_public(grid, cache_dir="dem_cache", prefer=("GLO30", "GLO90")):
    """Elevation from the public Copernicus DEM (AWS Open Data), resampled onto the analysis grid.

    GLO-30 (30 m) is tried first, then GLO-90 (90 m). The native resolution is recorded: a 90 m DEM
    resampled to a 30 m grid is smoother than the real terrain and has NO extra detail."""
    os.makedirs(cache_dir, exist_ok=True)
    errors = []
    for source in prefer:
        label, native = DEM_SOURCES[source][3], DEM_SOURCES[source][2]
        try:
            dem = np.full(grid.shape, np.nan, dtype="float32")
            names = []
            for lat, lon in dem_tiles_for_bbox(grid.bbox):
                name = dem_tile_name(lat, lon, source)
                path = os.path.join(cache_dir, name + ".tif")
                if not (os.path.exists(path) and os.path.getsize(path) > 1024):
                    logger.info("Downloading DEM tile %s", name)
                    _download(dem_tile_url(lat, lon, source), path)
                arr, bounds, nodata = _read_dem_tile(path)
                part = resample_to_grid(arr, bounds, grid, nodata)
                fill = ~np.isfinite(dem) & np.isfinite(part)
                dem[fill] = part[fill]
                names.append(name)
            info = check_dem(dem, grid)
            info.update({"source": label, "source_key": source, "native_resolution_m": native,
                         "tiles": names, "grid_resolution_m": round(math.sqrt(grid.dx_m * grid.dy_m), 1),
                         "note": ("Resampled onto the analysis grid by bilinear interpolation. "
                                  "No terrain detail finer than %d m is present." % native)})
            logger.info("DEM ready: %s, %d m native, range %.0f-%.0f m", label, native, info["min_m"], info["max_m"])
            return dem, info
        except Exception as ex:
            errors.append("%s: %s" % (source, ex))
            logger.warning("DEM source %s failed: %s", source, ex)
    raise RuntimeError("No public DEM could be loaded (%s). You can download a GeoTIFF of your area yourself "
                       "(for example from OpenTopography) and use dem_from_geotiff(path, grid)."
                       % "; ".join(errors))


def dem_from_geotiff(path, grid, native_resolution_m=None):
    """Manual fallback: use a DEM GeoTIFF (EPSG:4326) that you uploaded yourself."""
    arr, bounds, nodata = _read_dem_tile(path)
    dem = resample_to_grid(arr, bounds, grid, nodata)
    info = check_dem(dem, grid)
    info.update({"source": "user-supplied GeoTIFF " + os.path.basename(path), "native_resolution_m": native_resolution_m,
                 "tiles": [os.path.basename(path)]})
    return dem, info


def slope_degrees(dem, grid):
    """Slope in degrees. Cells next to missing data stay NaN (they are excluded later, never filled)."""
    dem = dem.astype("float64")
    dzdy, dzdx = np.gradient(dem, grid.dy_m, grid.dx_m)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def approx_hand(dem, river_mask):
    """Approximate 'height above nearest mapped river' (metres).

    For every pixel: its elevation minus the elevation of the nearest river pixel (straight-line, not
    flow-path, so it is only a rough guide). Flood water cannot sit tens of metres above the river."""
    from scipy import ndimage
    if not river_mask.any():
        raise ValueError("No river pixels: cannot compute height above river.")
    d = np.where(np.isfinite(dem), dem, np.inf)
    low = ndimage.minimum_filter(d, size=3)            # river pixel = lowest of its 3x3 neighbourhood
    low = np.where(np.isfinite(low), low, np.nan)
    _, (ri, ci) = ndimage.distance_transform_edt(~river_mask, return_indices=True)
    return (dem - low[ri, ci]).astype("float32")


def trace_downstream(dem, grid, lon, lat, max_steps=50000):
    """BONUS: follow the steepest downhill neighbour from a start point (simple, can stop in pits)."""
    r, c, inside = grid.rowcol(lon, lat)
    if not bool(inside[0]):
        raise ValueError("Start point is outside the grid")
    r, c = int(r[0]), int(c[0])
    path = [(r, c)]
    seen = {(r, c)}
    reason = "max_steps"
    for _ in range(max_steps):
        r0, c0 = path[-1]
        best, bestz = None, dem[r0, c0]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if rr < 0 or cc < 0 or rr >= dem.shape[0] or cc >= dem.shape[1] or (rr, cc) in seen:
                    continue
                if dem[rr, cc] < bestz:
                    best, bestz = (rr, cc), dem[rr, cc]
        if best is None:
            on_edge = r0 in (0, dem.shape[0] - 1) or c0 in (0, dem.shape[1] - 1)
            reason = "reached_grid_edge" if on_edge else "stopped_in_pit_or_flat_area"
            break
        path.append(best)
        seen.add(best)
    lons = grid.west + (np.array([p[1] for p in path]) + 0.5) / grid.width * (grid.east - grid.west)
    lats = grid.north - (np.array([p[0] for p in path]) + 0.5) / grid.height * (grid.north - grid.south)
    return {"lon": lons, "lat": lats, "stopped_because": reason}


def settlements_near_path(path, settlements, radius_m=500.0):
    lat0 = float(np.mean(path["lat"]))
    kx = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    px = path["lon"] * kx
    py = path["lat"] * M_PER_DEG_LAT
    out = []
    for s in settlements:
        d = np.sqrt((px - s["lon"] * kx) ** 2 + (py - s["lat"] * M_PER_DEG_LAT) ** 2).min()
        if d <= radius_m:
            out.append(dict(s, distance_to_path_m=round(float(d), 0)))
    return sorted(out, key=lambda x: x["distance_to_path_m"])


# --------------------------------------------------------------------------
# Pre-event OpenStreetMap via Overpass (cache, patient retries, snapshot verification)
# --------------------------------------------------------------------------
ROAD_TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|"
              "living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")

_OSM_STATE = {"bad_servers": {}, "servers_used": set(), "base_timestamps": [], "legacy_cache_used": False,
              "unverified_pieces": 0, "pieces": 0}


def reset_osm_state():
    _OSM_STATE.update({"bad_servers": {}, "servers_used": set(), "base_timestamps": [],
                       "legacy_cache_used": False, "unverified_pieces": 0, "pieces": 0})


def _snapshot_ok(osm_base, snapshot):
    """True if the server says its data is not newer than the end of the snapshot day."""
    try:
        t = datetime.strptime(osm_base[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None
    limit = datetime.strptime(snapshot, "%Y-%m-%d") + timedelta(days=1, minutes=1)
    return t <= limit


def _overpass_query(query, snapshot, label, budget_s=480, servers=None):
    """POST one query. Rotates over servers, backs off when they are busy, never waits forever.
    A reply is accepted only if its data timestamp (osm3s.timestamp_osm_base) is not newer than the snapshot:
    that is how we make sure no post-event edits slip in."""
    servers = list(servers or OVERPASS_SERVERS)
    deadline = time.time() + budget_s
    errors, rnd = [], 0
    while time.time() < deadline:
        usable = [s for s in servers if _OSM_STATE["bad_servers"].get(s, 0) < 2]
        if not usable:
            break
        waited_for_busy = False
        for url in usable:
            host = url.split("/")[2]
            try:
                r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=150)
            except requests.exceptions.RequestException as ex:
                errors.append("%s: %s" % (host, type(ex).__name__))
                continue
            if r.status_code == 200:
                try:
                    js = r.json()
                except ValueError:
                    errors.append("%s: reply was not JSON" % host)
                    continue
                base = (js.get("osm3s") or {}).get("timestamp_osm_base", "")
                ok = _snapshot_ok(base, snapshot) if base else None
                if ok is False:
                    _OSM_STATE["bad_servers"][url] = _OSM_STATE["bad_servers"].get(url, 0) + 2
                    errors.append("%s: no history (data from %s is newer than snapshot) - skipped" % (host, base))
                    logger.warning("%s ignores the snapshot date; it will not be used again.", host)
                    continue
                _OSM_STATE["servers_used"].add(host)
                if base:
                    _OSM_STATE["base_timestamps"].append(base)
                else:
                    _OSM_STATE["unverified_pieces"] += 1
                els = js.get("elements", [])
                logger.info("  [%s] %s: %d elements", host, label, len(els))
                return els, base
            errors.append("%s: HTTP %s" % (host, r.status_code))
            if r.status_code in (400, 403, 500):
                _OSM_STATE["bad_servers"][url] = _OSM_STATE["bad_servers"].get(url, 0) + 1
            if r.status_code in (429, 502, 503, 504):
                waited_for_busy = True
        rnd += 1
        if time.time() >= deadline:
            break
        wait = min(120, 15 * (2 ** min(rnd - 1, 3))) * (0.8 + 0.4 * random.random())
        if not waited_for_busy:
            wait = min(wait, 20)
        logger.info("  servers busy; waiting %.0f s (round %d)...", wait, rnd)
        _sleep(wait)
    raise RuntimeError("Overpass gave no usable answer for '%s' within %d s (%s). Re-run the cell later: "
                       "finished pieces are cached. A smaller BBOX also helps."
                       % (label, budget_s, "; ".join(errors[-6:]) or "no details"))


def overpass_snapshot(bbox, snapshot, body, out="out geom tags;", label="query", budget_s=480, pause_s=2.0):
    """Elements as of `snapshot` (YYYY-MM-DD) for one bbox piece. Results are cached on disk."""
    w, s, e, n = bbox
    stmt = body.format(s=s, w=w, n=n, e=e)
    query = '[out:json][timeout:120][date:"%sT00:00:00Z"];(%s);%s' % (snapshot, stmt, out)
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(OSM_CACHE_DIR, hashlib.sha256(query.encode()).hexdigest()[:24] + ".json")
    _OSM_STATE["pieces"] += 1
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list):             # older cache format: no timestamp stored
                _OSM_STATE["legacy_cache_used"] = True
                _OSM_STATE["unverified_pieces"] += 1
                els = cached
            else:
                els = cached.get("elements", [])
                if cached.get("osm_base"):
                    _OSM_STATE["base_timestamps"].append(cached["osm_base"])
                else:
                    _OSM_STATE["unverified_pieces"] += 1
            logger.info("  [cache] %s: %d elements", label, len(els))
            return els
        except Exception:
            logger.warning("Cache file %s unreadable; downloading again.", cache_file)
    els, base = _overpass_query(query, snapshot, label, budget_s)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"elements": els, "osm_base": base, "snapshot": snapshot}, f)
    _sleep(pause_s)   # be polite to the public server
    return els


def split_bbox(bbox, nx, ny):
    w, s, e, n = bbox
    tiles = []
    for i in range(nx):
        for j in range(ny):
            tiles.append((w + (e - w) * i / nx, s + (n - s) * j / ny,
                          w + (e - w) * (i + 1) / nx, s + (n - s) * (j + 1) / ny))
    return tiles


def _tiled(bbox, snapshot, body, out, nx, ny, label="tiles"):
    seen, result = set(), []
    tiles = split_bbox(bbox, nx, ny)
    for k, tile in enumerate(tiles, 1):
        for el in overpass_snapshot(tile, snapshot, body, out, label="%s %d/%d" % (label, k, len(tiles))):
            key = (el.get("type"), el.get("id"))
            if key not in seen:
                seen.add(key)
                result.append(el)
    return result


def element_point(el):
    if "lat" in el and "lon" in el:
        return float(el["lon"]), float(el["lat"])
    c = el.get("center")
    if c:
        return float(c["lon"]), float(c["lat"])
    return None


def parse_places(elements):
    out = []
    for el in elements:
        pt = element_point(el)
        tags = el.get("tags", {})
        if pt is None or not tags.get("name"):
            continue
        out.append({"osm_id": el.get("id"), "name": tags["name"], "name_ne": tags.get("name:ne", ""),
                    "name_en": tags.get("name:en", ""), "place": tags.get("place", tags.get("amenity", "")),
                    "lon": pt[0], "lat": pt[1]})
    return out


def parse_ways(elements):
    out = []
    for el in elements:
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
        if len(coords) >= 2:
            out.append({"id": el["id"], "tags": el.get("tags", {}), "coords": coords})
    return out


def fetch_osm_overpass_v02(bbox, snapshot):
    """Fetch pre-event OSM layers (as of `snapshot`). Returns plain Python lists plus osm['meta'].
    Progress is printed; finished pieces are cached in ./osm_cache so a re-run resumes."""
    reset_osm_state()
    logger.info("Rivers...")
    rivers = parse_ways(_tiled(bbox, snapshot, 'way["waterway"="river"]({s},{w},{n},{e});',
                               "out geom tags;", 2, 2, "rivers"))
    logger.info("Roads...")
    roads = parse_ways(_tiled(bbox, snapshot, 'way["highway"~"^(%s)$"]({s},{w},{n},{e});' % ROAD_TYPES,
                              "out geom tags;", 3, 3, "roads"))
    logger.info("Buildings...")
    b_el = _tiled(bbox, snapshot, 'nwr["building"]({s},{w},{n},{e});', "out center qt;", 4, 4, "buildings")
    buildings = [element_point(el) for el in b_el if element_point(el) is not None]
    logger.info("Places and hospitals...")
    places = parse_places(overpass_snapshot(
        bbox, snapshot, 'nwr["place"~"^(city|town|village|hamlet)$"]["name"]({s},{w},{n},{e});',
        "out center tags;", label="places"))
    hospitals = parse_places(overpass_snapshot(
        bbox, snapshot,
        'nwr["amenity"="hospital"]({s},{w},{n},{e});nwr["healthcare"="hospital"]({s},{w},{n},{e});',
        "out center tags;", label="hospitals"))
    bridges = [w for w in roads if w["tags"].get("bridge", "no") not in ("no", "")]
    st = _OSM_STATE
    verified = st["unverified_pieces"] == 0 and len(st["base_timestamps"]) > 0
    meta = {"snapshot": snapshot, "servers": sorted(st["servers_used"]),
            "newest_data_timestamp": max(st["base_timestamps"]) if st["base_timestamps"] else None,
            "snapshot_verified": bool(verified),
            "note": ("All pieces carry a server timestamp not newer than the snapshot." if verified else
                     "Some pieces came from an older cache without a stored timestamp (they were downloaded "
                     "with the same [date:] setting, but this could not be re-verified).")}
    return {"rivers": rivers, "roads": roads, "buildings": buildings, "places": places,
            "hospitals": hospitals, "bridges": bridges, "meta": meta}


def make_targets(osm):
    """Hospitals plus towns/cities: where people need to be able to get to."""
    return osm["hospitals"] + [p for p in osm["places"] if p["place"] in ("town", "city")]


# --------------------------------------------------------------------------
# Flood water and surface-change detection (radar)
# --------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # speckle: Sentinel-1 GRDH has ~4.4 equivalent looks; 'lee' keeps edges better than a plain box average
    "speckle_filter": "lee", "filter_size": 3, "enl_native": 4.4,
    # open water (backscatter is low in both polarisations)
    "water_vv_db": -18.0, "water_vh_db": -24.0, "grow_db": 2.0,
    "pre_bright_db": -16.0,   # it must NOT already have been this dark before the event
    "drop_db": 3.0,           # and VV must have dropped by at least this much
    # general surface change (candidate debris / sediment / wet ground)
    "k_sigma": 3.0, "change_db": 3.0, "vh_change_db": 2.0, "unstable_db": 3.0,
    # terrain and geometry gates
    "max_slope_water": 12.0, "max_slope_change": 25.0, "max_hand_m": 15.0,
    "min_incidence": 15.0, "max_incidence": 65.0,
    # where to look (distance from pre-event OSM rivers)
    "corridor_m": 400.0, "water_corridor_m": 1000.0,
    "min_pixels_water": 4, "min_pixels_change": 8,
    # built-up areas: bright, variable scatterers; excluded from the CHANGE class only (not from water)
    "builtup_density_per_km2": 300.0, "builtup_window_px": 5,
    "normalise_bias": True,
    # evidence rules
    "min_water_km2": 0.02, "max_control_ratio": 0.5, "min_corridor_valid": 0.7, "max_noise_sigma_db": 2.5,
}


def to_db(x):
    return 10.0 * np.log10(np.clip(x, 1e-4, None))


def _nmean(x, valid, size):
    """Moving average that ignores invalid pixels (normalised convolution)."""
    from scipy.ndimage import uniform_filter
    v = valid.astype("float64")
    num = uniform_filter(np.where(valid, x, 0.0), size=size, mode="nearest")
    den = uniform_filter(v, size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out


def speckle_filter(x, valid, method="lee", size=3, enl=4.4):
    """Reduce radar speckle. x is linear power. Returns linear power (NaN where invalid)."""
    x = np.nan_to_num(x.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0)
    if method == "none":
        return np.where(valid, x, np.nan)
    mean = _nmean(x, valid, size)
    if method == "boxcar":
        return np.where(valid, mean, np.nan)
    if method != "lee":
        raise ValueError("speckle filter must be 'lee', 'boxcar' or 'none'")
    mean2 = _nmean(x * x, valid, size)
    var = np.clip(mean2 - mean ** 2, 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci2 = np.where(mean > 0, var / (mean ** 2), 0.0)
        cu2 = 1.0 / max(enl, 1.0)
        w = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-12), 0.0, 1.0)
    out = mean + w * (x - mean)
    return np.where(valid & np.isfinite(out), np.clip(out, 1e-6, None), np.nan)


def theoretical_sigma_db(params):
    """Rough standard deviation (dB) of the difference of two filtered images from speckle alone.
    Uses the box-average look count as an approximation; the empirical value from the control pair is preferred."""
    looks = params["enl_native"] * params["filter_size"] ** 2
    return float(4.343 / math.sqrt(looks) * math.sqrt(2.0))


def remove_small(mask, min_pixels):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask.copy()
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_pixels
    return keep[lab]


def rasterize_lines(grid, lines, step_m=10.0):
    mask = np.zeros(grid.shape, dtype=bool)
    for coords in lines:
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            lon, lat = _densify(x0, y0, x1, y1, step_m)
            r, c, inside = grid.rowcol(lon, lat)
            mask[r[inside], c[inside]] = True
    return mask


def river_distance_m(grid, river_lines):
    """Distance (m) from every pixel to the nearest pre-event OSM river line, plus the river mask."""
    from scipy import ndimage
    rivers = rasterize_lines(grid, river_lines)
    if not rivers.any():
        raise ValueError("No river from OpenStreetMap falls inside the grid; cannot build a corridor.")
    dist = ndimage.distance_transform_edt(~rivers, sampling=(grid.dy_m, grid.dx_m))
    return dist, rivers


def corridor_mask(grid, river_lines, buffer_m):
    dist, _ = river_distance_m(grid, river_lines)
    return dist <= buffer_m


def builtup_mask(grid, building_points, params=None):
    """Pixels inside dense clusters of pre-event OSM buildings. Radar backscatter there is bright and
    changes a lot for reasons unrelated to floods, so these pixels are NOT assessed for surface change."""
    from scipy.ndimage import binary_dilation, uniform_filter
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    cnt = np.zeros(grid.shape, dtype="float64")
    if building_points:
        lon = np.array([b[0] for b in building_points], dtype=float)
        lat = np.array([b[1] for b in building_points], dtype=float)
        r, c, inside = grid.rowcol(lon, lat)
        np.add.at(cnt, (r[inside], c[inside]), 1.0)
    size = int(p["builtup_window_px"])
    per_px = uniform_filter(cnt, size=size, mode="constant")          # mean buildings per pixel in the window
    density = per_px / max(grid.pixel_area_km2, 1e-12)                # buildings per km2
    mask = density >= p["builtup_density_per_km2"]
    return binary_dilation(mask, iterations=1) if mask.any() else mask


def terrain_context(grid, dem, river_dist_m, river_mask, building_points=None, params=None):
    """Everything the detector needs besides the radar images."""
    return {"slope": slope_degrees(dem, grid), "hand": approx_hand(dem, river_mask),
            "river_dist_m": river_dist_m, "builtup": builtup_mask(grid, building_points or [], params)}


def _robust_sigma(x):
    x = x[np.isfinite(x)]
    if x.size < 200:
        return None
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _filtered_db(radar, valid, p):
    vv = speckle_filter(radar["vv"], valid, p["speckle_filter"], p["filter_size"], p["enl_native"])
    vh = speckle_filter(radar["vh"], valid, p["speckle_filter"], p["filter_size"], p["enl_native"])
    return to_db(vv), to_db(vh)


def detect_flood(pre, post, ctx, ctrl=None, params=None, use_unstable=True):
    """Detect (1) open-water flooding and (2) other surface change from a before/after radar pair.

    pre/post/ctrl: dicts from fetch_s1_scene (same orbit track and direction!).
    ctx: dict from terrain_context(). ctrl: an earlier PRE-event scene on the same track, used only to
    measure noise and to flag pixels that are unstable even without any flood.

    Returns masks plus 'criteria' (the ordered list of tests behind the water mask, used by funnel())."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    valid = reliable_mask(pre, p) & reliable_mask(post, p)
    pre_vv, pre_vh = _filtered_db(pre, valid, p)
    post_vv, post_vh = _filtered_db(post, valid, p)
    d_vv, d_vh = post_vv - pre_vv, post_vh - pre_vh
    rd = ctx["river_dist_m"]
    near_w = rd <= p["water_corridor_m"]
    near_c = rd <= p["corridor_m"]
    stats = {"valid_fraction": float(valid.mean()),
             "corridor_valid_fraction": float(valid[near_w].mean()) if near_w.any() else 0.0}

    # radiometric normalisation: remove a scene-wide offset (rain, soil moisture, calibration) measured
    # far away from the rivers, where no flood can be.
    bias_vv = bias_vh = 0.0
    if p["normalise_bias"]:
        far = valid & ~near_w & np.isfinite(d_vv)
        ref = far if far.sum() >= 1000 else (valid & np.isfinite(d_vv))
        if ref.any():
            bias_vv, bias_vh = float(np.median(d_vv[ref])), float(np.median(d_vh[ref]))
        d_vv, d_vh = d_vv - bias_vv, d_vh - bias_vh
    stats.update(bias_vv_db=round(bias_vv, 2), bias_vh_db=round(bias_vh, 2))

    # noise: empirical (control pair) if available, theoretical otherwise
    sigma_theory = theoretical_sigma_db(p)
    sigma_emp = None
    unstable = np.zeros(grid_shape(pre), dtype=bool)
    if ctrl is not None:
        vc = reliable_mask(ctrl, p) & reliable_mask(pre, p)
        c_vv, _ = _filtered_db(ctrl, vc, p)
        p_vv, _ = _filtered_db(pre, vc, p)
        d_c = c_vv - p_vv
        d_c = d_c - (np.nanmedian(d_c[vc]) if p["normalise_bias"] and vc.any() else 0.0)
        sigma_emp = _robust_sigma(d_c[vc & near_w])
        if use_unstable:
            with np.errstate(invalid="ignore"):
                unstable = vc & (np.abs(d_c) >= p["unstable_db"])
    sigma = sigma_emp if sigma_emp is not None else sigma_theory
    thr_change = max(p["change_db"], p["k_sigma"] * sigma)
    stats.update(sigma_theory_db=round(sigma_theory, 2),
                 sigma_empirical_db=None if sigma_emp is None else round(sigma_emp, 2),
                 change_threshold_db=round(thr_change, 2))

    slope, hand = ctx["slope"], ctx.get("hand")
    with np.errstate(invalid="ignore"):
        slope_w = slope <= p["max_slope_water"]
        slope_c = slope <= p["max_slope_change"]
        hand_ok = (hand <= p["max_hand_m"]) if hand is not None else np.ones_like(slope_w)

    # ---- open water: ordered criteria (kept so funnel() can show where candidates are lost) ----
    crit = [
        ("valid, reliable radar pixel", valid),
        ("within %d m of a mapped river" % p["water_corridor_m"], near_w),
        ("after: VV < %.0f dB" % p["water_vv_db"], post_vv < p["water_vv_db"]),
        ("after: VH < %.0f dB" % p["water_vh_db"],
         (post_vh < p["water_vh_db"]) if p["water_vh_db"] is not None else np.ones_like(valid)),
        ("before: VV > %.0f dB (not already dark)" % p["pre_bright_db"], pre_vv > p["pre_bright_db"]),
        ("VV dropped by >= %.0f dB" % p["drop_db"], d_vv <= -p["drop_db"]),
        ("slope <= %.0f deg" % p["max_slope_water"], slope_w),
        ("height above river <= %.0f m" % p["max_hand_m"], hand_ok),
    ]
    seeds = np.ones(valid.shape, dtype=bool)
    for _, m in crit:
        seeds &= np.where(np.isfinite(m.astype(float)), m, False).astype(bool)
    relaxed = (valid & near_w & (post_vv < p["water_vv_db"] + p["grow_db"])
               & (d_vv <= -max(1.5, p["drop_db"] - 1.5)) & (pre_vv > p["pre_bright_db"] - p["grow_db"])
               & slope_w & hand_ok)
    from scipy import ndimage
    lab, n = ndimage.label(relaxed | seeds)
    if n:
        keep = np.zeros(n + 1, dtype=bool)
        keep[np.unique(lab[seeds & (lab > 0)])] = True
        keep[0] = False
        water = keep[lab]
    else:
        water = np.zeros_like(seeds)
    water = remove_small(water, p["min_pixels_water"])
    permanent = valid & near_w & (pre_vv < p["water_vv_db"]) & (post_vv < p["water_vv_db"])

    # ---- other surface change (not open water) ----
    builtup = ctx["builtup"]
    sig = (valid & near_c & (np.abs(d_vv) >= thr_change) & (np.abs(d_vh) >= p["vh_change_db"])
           & (np.sign(d_vv) == np.sign(d_vh)) & slope_c)
    cand = sig & ~builtup & ~unstable & ~water & ~permanent
    inc = remove_small(cand & (d_vv > 0), p["min_pixels_change"])
    dec = remove_small(cand & (d_vv < 0), p["min_pixels_change"])

    stats["builtup_excluded_px"] = int((sig & builtup).sum())
    stats["unstable_excluded_px"] = int((sig & ~builtup & unstable).sum())
    return {"water": water, "change_increase": inc, "change_decrease": dec, "permanent_water": permanent,
            "builtup": builtup, "unstable": unstable, "valid": valid, "criteria": crit,
            "seed_pixels": int(seeds.sum()), "relaxed_pixels": int(relaxed.sum()),
            "diff_vv_db": d_vv, "diff_vh_db": d_vh, "pre_vv_db": pre_vv, "post_vv_db": post_vv,
            "post_vh_db": post_vh, "stats": stats, "params": p}


def grid_shape(radar):
    return radar["vv"].shape


def funnel(det):
    """Why is the water mask what it is? Counts how many pixels survive each test, one after another."""
    rows = []
    cum = np.ones(det["valid"].shape, dtype=bool)
    for name, m in det["criteria"]:
        m = np.where(np.isfinite(m.astype(float)), m, False).astype(bool)
        alone = int(m.sum())
        cum &= m
        rows.append({"test": name, "pixels_passing_this_test_alone": alone, "pixels_passing_all_so_far": int(cum.sum())})
    rows.append({"test": "joined with neighbours (hysteresis) and >= %d connected pixels" % det["params"]["min_pixels_water"],
                 "pixels_passing_this_test_alone": det["relaxed_pixels"],
                 "pixels_passing_all_so_far": int(det["water"].sum())})
    return rows


def explain_empty_water(det, grid):
    """Plain-language list of likely reasons for an empty water mask, based on the numbers (not guesses)."""
    rows = funnel(det)
    out = []
    names = [r["test"] for r in rows]
    cum = [r["pixels_passing_all_so_far"] for r in rows]
    for i in range(1, len(rows)):
        if cum[i - 1] > 0 and cum[i] == 0:
            out.append("All remaining candidates were removed by the test: '%s'." % names[i])
            break
    corridor_px = rows[1]["pixels_passing_this_test_alone"]
    dark = rows[2]["pixels_passing_all_so_far"]
    if corridor_px and dark == 0:
        out.append("No pixel near the rivers is darker than the water threshold after the event. "
                   "Either there was no wide open water, or the water is narrower than the %.0f m pixels "
                   "(Himalayan river channels are often only 20-60 m wide)." % math.sqrt(grid.dx_m * grid.dy_m))
    if det["stats"].get("corridor_valid_fraction", 1) < 0.7:
        out.append("Less than 70% of the river corridor has reliable radar data (layover/shadow or missing data).")
    return out


def assess_evidence(flood, qc, params=None, s2=None, pair_warnings=None):
    """Decide how much the evidence supports. Never forces a positive result.

    verdict: 'water_detected' | 'no_water_detected' | 'inconclusive'. confidence: 'low' | 'medium'
    ('high' is never used: nothing here is validated against reference data)."""
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    reasons = []

    def add(code, **kw):
        reasons.append(dict(code=code, **kw))

    if qc.get("corridor_valid_fraction", 1.0) < p["min_corridor_valid"]:
        add("low_valid_coverage", value=round(100 * qc["corridor_valid_fraction"], 0))
    sig = qc.get("sigma_empirical_db")
    if sig is not None and sig > p["max_noise_sigma_db"]:
        add("noisy_scene", value=round(sig, 2))
    if pair_warnings:
        for w in pair_warnings:
            if "covers only" in w:
                add("low_pair_coverage")
    blockers = [r for r in reasons if r["code"] in ("low_valid_coverage", "noisy_scene", "low_pair_coverage")]

    water = flood["water_km2"]
    cw = flood.get("control_water_km2")
    ratio_bad = (cw is not None and water > 0 and cw >= p["max_control_ratio"] * water)
    if flood.get("control_water_km2") is None:
        add("no_control")
    if ratio_bad:
        add("water_not_above_noise", control=round(cw, 2), event=round(water, 2))

    s2_note = None
    if s2 and s2.get("usable"):
        s2_note = s2

    if blockers:
        verdict, conf = "inconclusive", None
    elif water >= p["min_water_km2"] and not ratio_bad:
        verdict = "water_detected"
        strong = (cw is not None and water >= 0.5 and cw <= 0.1 * water)
        s2_ok = bool(s2_note and s2_note.get("agreement") is not None and s2_note["agreement"] >= 0.5)
        conf = "medium" if (strong and s2_ok) else "low"
        if s2_note and s2_note.get("agreement") is not None and s2_note["agreement"] < 0.2:
            add("optical_disagrees")
            conf = "low"
    elif ratio_bad:
        verdict, conf = "inconclusive", None
    else:
        verdict, conf = "no_water_detected", None
    return {"verdict": verdict, "confidence": conf, "reasons": reasons,
            "validated_against_reference": False}


def s2_support(s2_pre, s2_post, water, near_mask, grid, min_cloud_free=0.3):
    """Optional optical evidence: where radar says water, does cloud-free Sentinel-2 (SCL class 6) agree?"""
    def clear(s):
        return s["valid"] & ~np.isin(s["scl"], [3, 8, 9, 10])
    cp, cq = clear(s2_pre), clear(s2_post)
    cover_water = float((cq[water]).mean()) if water.any() else 0.0
    cover_area = float(cq[near_mask].mean()) if near_mask.any() else 0.0
    usable = cover_area >= min_cloud_free
    agreement = None
    if water.any() and cq[water].any():
        agreement = float(((s2_post["scl"] == 6) & cq)[water & cq].mean())
    new_water = (s2_post["scl"] == 6) & cq & ~((s2_pre["scl"] == 6) & cp) & cp & near_mask
    return {"usable": bool(usable), "cloud_free_fraction_of_corridor": round(cover_area, 3),
            "cloud_free_fraction_of_radar_water": round(cover_water, 3),
            "agreement": None if agreement is None else round(agreement, 3),
            "optical_new_water_km2": round(area_km2(new_water, grid), 2) if usable else None,
            "cloud_fraction_pre": round(float(s2_pre["cloud_fraction"]), 3),
            "cloud_fraction_post": round(float(s2_post["cloud_fraction"]), 3)}


# --------------------------------------------------------------------------
# Road network, impacts and "who might be cut off"
# --------------------------------------------------------------------------
def build_road_graph(ways):
    import networkx as nx
    G = nx.Graph()
    for w in ways:
        coords = [(round(x, 7), round(y, 7)) for x, y in w["coords"]]
        for a, b in zip(coords[:-1], coords[1:]):
            if a == b:
                continue
            L = haversine_m(a, b)
            if G.has_edge(a, b):
                continue
            G.add_edge(a, b, length=L, way=w["id"])
    return G


def edge_hazard(G, grid, hazard, step_m=10.0, min_block_m=30.0):
    """For every road segment: how many metres of it lie inside flagged pixels?
    A segment counts as 'blocked' if at least `min_block_m` metres are flagged (about one pixel).
    Returns (blocked_edges, hazard_length_km)."""
    blocked, total = [], 0.0
    for a, b, data in G.edges(data=True):
        lon, lat = _densify(a[0], a[1], b[0], b[1], step_m)
        hit = sample_mask(grid, hazard, lon, lat)
        if not hit.any():
            continue
        length = data["length"]
        hazard_m = float(hit.mean()) * length
        total += hazard_m
        if hazard_m >= min_block_m or hit.all():
            blocked.append((a, b))
    return blocked, total / 1000.0


class Snapper:
    """Find the nearest road-graph node to a point."""

    def __init__(self, G):
        from scipy.spatial import cKDTree
        self.nodes = list(G.nodes)
        if not self.nodes:
            raise ValueError("Road graph is empty")
        arr = np.array(self.nodes, dtype=float)
        self.lon0 = float(arr[:, 0].mean())
        self.lat0 = float(arr[:, 1].mean())
        self.kx = M_PER_DEG_LON_EQ * math.cos(math.radians(self.lat0))
        self.tree = cKDTree(self._xy(arr[:, 0], arr[:, 1]))

    def _xy(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * M_PER_DEG_LAT])

    def snap(self, lon, lat):
        d, i = self.tree.query(self._xy([lon], [lat])[0])
        return self.nodes[int(i)], float(d)


def cutoff_analysis(G_pre, blocked, settlements, targets, snap_max_m=500.0, snapper=None):
    """Which settlements lose their road link to every hospital/town if `blocked` segments are impassable?

    status: 'reachable' | 'cut_off' (NEWLY disconnected) | 'unconnected_before_event' (already disconnected
    in the pre-event road map) | 'not_on_road_network' (no mapped road within snap_max_m)."""
    import networkx as nx
    G_post = G_pre.copy()
    G_post.remove_edges_from(blocked)
    snap = snapper or Snapper(G_pre)
    tnodes = set()
    for t in targets:
        node, d = snap.snap(t["lon"], t["lat"])
        if d <= snap_max_m:
            tnodes.add(node)
    if not tnodes:
        raise ValueError("No hospital or town could be matched to the road network. "
                         "Enlarge the BBOX or add targets by hand.")
    dist_pre = nx.multi_source_dijkstra_path_length(G_pre, tnodes, weight="length")
    dist_post = nx.multi_source_dijkstra_path_length(G_post, tnodes, weight="length")
    rows = []
    for s in settlements:
        node, d = snap.snap(s["lon"], s["lat"])
        row = dict(s)
        row["snap_distance_m"] = round(d, 0)
        row["dist_pre_km"] = None
        row["dist_post_km"] = None
        if d > snap_max_m:
            row["status"] = "not_on_road_network"
        else:
            pre, post = dist_pre.get(node), dist_post.get(node)
            if pre is None:
                row["status"] = "unconnected_before_event"
            elif post is None:
                row["status"] = "cut_off"
                row["dist_pre_km"] = round(pre / 1000.0, 2)
            else:
                row["status"] = "reachable"
                row["dist_pre_km"] = round(pre / 1000.0, 2)
                row["dist_post_km"] = round(post / 1000.0, 2)
        rows.append(row)
    return rows, len(tnodes)


def access_scenarios(G, grid, hazards, settlements, targets, min_block_m=30.0):
    """Run the cut-off analysis for several hazard definitions.
    hazards: {'likely': water mask, 'possible': water + change masks}. Returns {name: {...}}."""
    snap = Snapper(G)
    out = {}
    for name, mask in hazards.items():
        blocked, hz_km = edge_hazard(G, grid, mask, min_block_m=min_block_m)
        rows, n_t = cutoff_analysis(G, blocked, settlements, targets, snapper=snap)
        total_km = sum(d["length"] for _, _, d in G.edges(data=True)) / 1000.0
        out[name] = {"rows": rows, "blocked": blocked, "n_targets": n_t,
                     "roads": {"total_km": round(total_km, 2), "hazard_km": round(hz_km, 2),
                               "blocked_segments": len(blocked)}}
    return out


def baseline_components(G, targets, snap_max_m=500.0):
    """How fragmented is the pre-event road map? (number of separate road networks, share reaching a target)"""
    import networkx as nx
    snap = Snapper(G)
    comps = list(nx.connected_components(G))
    tn = {snap.snap(t["lon"], t["lat"])[0] for t in targets if snap.snap(t["lon"], t["lat"])[1] <= snap_max_m}
    with_t = sum(1 for c in comps if c & tn)
    return {"road_components": len(comps), "components_with_a_target": with_t}


def routes_to_nearest_target(G, targets, settlements_rows, status="cut_off", limit=40, snap_max_m=500.0):
    """Pre-event shortest route from each flagged settlement to its nearest hospital/town (for the map)."""
    import networkx as nx
    snap = Snapper(G)
    tn = {snap.snap(t["lon"], t["lat"])[0] for t in targets if snap.snap(t["lon"], t["lat"])[1] <= snap_max_m}
    if not tn:
        return []
    _, paths = nx.multi_source_dijkstra(G, tn, weight="length")
    routes = []
    for r in settlements_rows:
        if r["status"] != status:
            continue
        node, d = snap.snap(r["lon"], r["lat"])
        path = paths.get(node)
        if path and len(path) > 1:
            routes.append({"name": r.get("name_en") or r["name"], "coords": list(reversed(path))})
        if len(routes) >= limit:
            break
    return routes


def building_damage(grid, buildings, water, change):
    """Buildings whose centre point lies in a flagged pixel. 'Potentially affected', not 'damaged'."""
    if not buildings:
        return {"total": 0, "in_water": 0, "in_change": 0, "in_hazard": 0, "hit_points": []}
    lon = np.array([b[0] for b in buildings])
    lat = np.array([b[1] for b in buildings])
    w = sample_mask(grid, water, lon, lat)
    d = sample_mask(grid, change, lon, lat) & ~w
    hit = w | d
    pts = [(float(lon[i]), float(lat[i])) for i in np.nonzero(hit)[0]]
    return {"total": len(buildings), "in_water": int(w.sum()), "in_change": int(d.sum()),
            "in_hazard": int(hit.sum()), "hit_points": pts}


def bridge_damage(grid, bridges, water, change, step_m=10.0):
    """Bridges whose line (sampled every 10 m, not just the end points) crosses flagged pixels.
    This means 'possibly affected' only: radar change next to a bridge is not evidence that it is damaged."""
    rows = []
    for b in bridges:
        hits_w = hits_c = False
        for (x0, y0), (x1, y1) in zip(b["coords"][:-1], b["coords"][1:]):
            lon, lat = _densify(x0, y0, x1, y1, step_m)
            hits_w |= bool(sample_mask(grid, water, lon, lat).any())
            hits_c |= bool(sample_mask(grid, change, lon, lat).any())
        rows.append({"way_id": b["id"], "name": b["tags"].get("name", ""),
                     "possibly_hit": bool(hits_w or hits_c), "hit_by_water": bool(hits_w)})
    return {"total": len(rows), "possibly_hit": sum(1 for r in rows if r["possibly_hit"]), "rows": rows}


# --------------------------------------------------------------------------
# Results file (the ONLY source of numbers for reports and the copilot)
# --------------------------------------------------------------------------
LIMITATIONS = [
    "Sentinel-1 revisits an area only every few days, so this cannot warn of a sudden event such as a glacial lake outburst.",
    "Radar water detection can miss narrow channels (narrower than a pixel), flooded vegetation and flooded built-up areas, and can give false flags from radar shadow, layover, wet snow, rain and soil moisture.",
    "'Surface change' is a statistical radar change near mapped rivers. It is not confirmed debris, and built-up and unstable pixels were not assessed.",
    "Roads, buildings and bridges come from pre-event OpenStreetMap, which is incomplete in remote areas; unmapped roads and paths are invisible.",
    "Cut-off status uses mapped roads only: it ignores footpaths, helicopters, repairs and the real road condition, and it depends on the flagged pixels being truly impassable.",
    "The thresholds were not validated against reference flood maps. Treat all results as unvalidated estimates.",
    "Educational prototype, not an operational tool; every result needs field confirmation.",
]


def sources_list(meta_in):
    s = [
        {"id": "S1", "name": "Copernicus Sentinel-1 GRD (IW, VV+VH), processed by Sentinel Hub to terrain-flattened "
                              "gamma0 (Copernicus Data Space Ecosystem)", "scenes": meta_in.get("scenes", [])},
        {"id": "DEM", "name": meta_in.get("dem_source", "Copernicus DEM"),
         "native_resolution_m": meta_in.get("dem_native_m")},
        {"id": "OSM", "name": "OpenStreetMap contributors (Overpass API), data as of %s" % meta_in.get("osm_snapshot")},
    ]
    if meta_in.get("s2_used"):
        s.append({"id": "S2", "name": "Copernicus Sentinel-2 L2A (optical support only)"})
    return s


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def environment_info():
    info = {"python": platform.python_version(), "floodresq": __version__, "numpy": np.__version__}
    for m in ("scipy", "pandas", "networkx", "rasterio", "geopandas", "shapely", "folium", "matplotlib", "requests"):
        try:
            mod = __import__(m)
            info[m] = getattr(mod, "__version__", "?")
        except Exception:
            info[m] = None
    return info


def build_results(grid, event_date, pair, control_scene, det, det_control, event_raw, flood_extra, qc, dem_info,
                  osm_meta, evidence, bld, bridges, access, baseline, config=None, s2=None, trace=None,
                  max_names=30):
    """Collect everything into one dictionary (saved as results.json). `flood_extra` holds areas in km2."""
    possible, likely = access["possible"], access["likely"]

    def names(rows):
        cut = [r for r in rows if r["status"] == "cut_off"]
        lst = [{"name": r.get("name") or "", "name_ne": r.get("name_ne") or "", "name_en": r.get("name_en") or ""}
               for r in cut[:max_names]]
        return lst, len(cut) > max_names

    def count(rows, s):
        return sum(1 for r in rows if r["status"] == s)

    pos_names, pos_trunc = names(possible["rows"])
    lik_names, _ = names(likely["rows"])
    fl = flood_extra
    return {
        "schema_version": 2,
        "meta": {
            "event_date": str(event_date), "bbox": [round(v, 4) for v in grid.bbox],
            "osm_snapshot": str(osm_meta["snapshot"]), "orbit_track": int(pair["rel_orbit"]),
            "orbit_direction": str(pair["direction"]),
            "pre_scene": str(pair["pre_scene"]), "post_scene": str(pair["post_scene"]),
            "pre_date": str(pair["pre_date"]), "post_date": str(pair["post_date"]), "gap_days": int(pair["gap_days"]),
            "control_scene": None if control_scene is None else str(control_scene["name"]),
            "pixel_size_m": round(math.sqrt(grid.dx_m * grid.dy_m), 1),
            "dem_source": dem_info.get("source"), "dem_native_resolution_m": dem_info.get("native_resolution_m"),
            "osm_snapshot_verified": bool(osm_meta.get("snapshot_verified")),
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "data_is_simulated": False, "software_version": __version__,
            "config_hash": config_hash(config or {}),
        },
        "parameters": det["params"],
        "quality": {
            "corridor_valid_fraction": round(det["stats"]["corridor_valid_fraction"], 3),
            "radiometric_bias_vv_db": det["stats"]["bias_vv_db"],
            "noise_sigma_theory_db": det["stats"]["sigma_theory_db"],
            "noise_sigma_empirical_db": det["stats"]["sigma_empirical_db"],
            "change_threshold_db": det["stats"]["change_threshold_db"],
            "terrain_geometry_bands_used": bool(qc.get("geometry_bands")),
            "s2": s2,
        },
        "evidence": evidence,
        "flood": {
            "water_km2": round(fl["water_km2"], 2),
            "change_increase_km2": round(fl["change_increase_km2"], 2),
            "change_decrease_km2": round(fl["change_decrease_km2"], 2),
            "permanent_water_km2": round(fl["permanent_water_km2"], 2),
            "builtup_not_assessed_km2": round(fl["builtup_km2"], 2),
            "unstable_not_assessed_km2": round(fl["unstable_km2"], 2),
            "control_water_km2": None if fl.get("control_water_km2") is None else round(fl["control_water_km2"], 2),
            "control_change_km2": None if fl.get("control_change_km2") is None else round(fl["control_change_km2"], 2),
        },
        "damage": {
            "buildings_total": bld["total"], "buildings_in_water": bld["in_water"],
            "buildings_in_change": bld["in_change"], "buildings_in_flagged_area": bld["in_hazard"],
            "roads_total_km": possible["roads"]["total_km"],
            "roads_in_flagged_area_km": possible["roads"]["hazard_km"],
            "roads_in_water_km": likely["roads"]["hazard_km"],
            "bridges_total": bridges["total"], "bridges_possibly_hit": bridges["possibly_hit"],
        },
        "access": {
            "settlements_total": len(possible["rows"]), "targets_matched_to_roads": possible["n_targets"],
            "possibly_cut_off": count(possible["rows"], "cut_off"),
            "likely_cut_off": count(likely["rows"], "cut_off"),
            "unconnected_before_event": count(possible["rows"], "unconnected_before_event"),
            "not_on_road_network": count(possible["rows"], "not_on_road_network"),
            "road_components_before_event": baseline["road_components"],
            "possibly_cut_off_names": pos_names, "possibly_cut_off_names_truncated": pos_trunc,
            "likely_cut_off_names": lik_names,
        },
        "trace": trace,
        "sources": sources_list({
            "scenes": [str(pair["pre_scene"]), str(pair["post_scene"])] + ([str(control_scene["name"])] if control_scene is not None else []),
            "dem_source": dem_info.get("source"), "dem_native_m": dem_info.get("native_resolution_m"),
            "osm_snapshot": osm_meta["snapshot"], "s2_used": bool(s2 and s2.get("usable"))}),
        "limitations": LIMITATIONS,
        "validation": {"validated_against_reference": False,
                       "note": "No comparison with an independent reference map has been run for these numbers."},
    }


def quality_warnings(results):
    w = []
    ev, f, q = results["evidence"], results["flood"], results["quality"]
    if ev["verdict"] == "inconclusive":
        w.append("Evidence is INCONCLUSIVE (see evidence.reasons). Do not present the numbers as a flood map.")
    if f["control_water_km2"] is None:
        w.append("No control pair was available, so the noise level was only estimated from theory.")
    elif f["water_km2"] > 0 and f["control_water_km2"] >= 0.5 * f["water_km2"]:
        w.append("The control comparison flagged a lot of 'water' relative to the event: possibly noise.")
    if not q["terrain_geometry_bands_used"]:
        w.append("Layover/shadow mask was not available; steep valleys may give false or missed detections.")
    if not results["meta"]["osm_snapshot_verified"]:
        w.append("OSM snapshot could not be fully verified by server timestamps (see osm meta in the log).")
    if results["meta"]["dem_native_resolution_m"] and results["meta"]["dem_native_resolution_m"] > results["meta"]["pixel_size_m"]:
        w.append("DEM is coarser than the analysis pixel: slopes are smoother than reality.")
    a = results["access"]
    if a["not_on_road_network"] > 0.5 * max(1, a["settlements_total"]):
        w.append("Most settlements are far from any mapped road; cut-off results are weak for this area.")
    return w


def freeze_run(results, masks, path="frozen_run.json"):
    """Fingerprint the finished analysis BEFORE comparing with EMSR927 (so tuning afterwards cannot go unnoticed)."""
    rec = {"sha256": _fingerprint(results, masks), "frozen_utc": datetime.now(timezone.utc).isoformat(),
           "mask_names": sorted(masks)}
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def _fingerprint(results, masks):
    h = hashlib.sha256()
    h.update(json.dumps(results, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    for k in sorted(masks):
        h.update(k.encode("utf-8"))
        h.update(np.packbits(masks[k]).tobytes())
    return h.hexdigest()


def verify_frozen(results, masks, path="frozen_run.json"):
    """True only if the results and masks are exactly the ones that were frozen."""
    if not os.path.exists(path):
        raise RuntimeError("Run freeze_run(...) first. Reference maps are for checking only, never for tuning.")
    with open(path) as f:
        rec = json.load(f)
    if rec["sha256"] != _fingerprint(results, masks):
        raise RuntimeError("The results/masks differ from the frozen run. Re-running the analysis after seeing a "
                           "reference map invalidates the evaluation. Restore the frozen run or report that it was changed.")
    return True


# --------------------------------------------------------------------------
# Evaluation against an independent reference (EMSR927 etc.) - EVALUATION ONLY
# --------------------------------------------------------------------------
def metrics_from_masks(pred, ref, aoi=None):
    """Pixel metrics inside the AOI. Returns NaN where a metric is undefined (never a made-up number)."""
    aoi = np.ones(pred.shape, dtype=bool) if aoi is None else aoi
    tp = int((pred & ref & aoi).sum())
    fp = int((pred & ~ref & aoi).sum())
    fn = int((~pred & ref & aoi).sum())
    tn = int((~pred & ~ref & aoi).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else float("nan")
    return {"true_pos_px": tp, "false_pos_px": fp, "false_neg_px": fn, "true_neg_px": tn,
            "precision": prec, "recall": rec, "iou": iou, "f1": f1}


def evaluate_vs_reference(pred, grid, reference_path, aoi_path=None, results=None, masks=None,
                          frozen_path="frozen_run.json"):
    """EVALUATION ONLY. Compare a prediction with an official reference layer you downloaded by hand.
    Requires the run to be frozen and unchanged (pass `results` and `masks` to verify)."""
    if results is not None and masks is not None:
        verify_frozen(results, masks, frozen_path)
    elif not os.path.exists(frozen_path):
        raise RuntimeError("Run freeze_run(...) first. Reference maps are for checking only, never for tuning.")
    import geopandas as gpd
    from rasterio import features

    def _raster(path):
        gdf = gpd.read_file(path).to_crs(4326)
        shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
        if not shapes:
            raise ValueError("No geometries in %s" % path)
        return features.rasterize(shapes, out_shape=grid.shape, transform=grid.transform(),
                                  fill=0, dtype="uint8").astype(bool)

    ref = _raster(reference_path)
    aoi = _raster(aoi_path) if aoi_path else np.ones(grid.shape, dtype=bool)
    out = metrics_from_masks(pred, ref, aoi)
    out["compared_area_km2"] = float(aoi.sum()) * grid.pixel_area_km2
    out["note"] = ("Computed only inside the reference AOI (if given). Radar-detected water is compared with the "
                   "reference flood extent; surface-change candidates are not flood extent and should not be scored as such.")
    return out


# --------------------------------------------------------------------------
# Situation report + copilot: English and Nepali text, every number from results.json
# --------------------------------------------------------------------------
def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "%.2f" % x


def _reason_text(r, lang):
    c = r["code"]
    if lang == "ne":
        return {
            "low_valid_coverage": "नदी करिडोरको मात्र %s%% भागमा भरपर्दो रडार तथ्याङ्क छ" % _f(r.get("value")),
            "noisy_scene": "तस्बिरको शोर (%s dB) धेरै उच्च छ" % _f(r.get("value")),
            "low_pair_coverage": "कुनै तस्बिरले अध्ययन क्षेत्र पूर्ण रूपमा ढाकेको छैन",
            "water_not_above_noise": "घटनाअघिका दुई तस्बिरमा पनि (%s वर्ग किमी) घटनाको (%s वर्ग किमी) जत्तिकै 'पानी' देखियो, त्यसैले शोरबाट छुट्याउन सकिएन" % (_f(r.get("control")), _f(r.get("event"))),
            "no_control": "शोर जाँचका लागि नियन्त्रण तस्बिर उपलब्ध थिएन",
            "optical_disagrees": "Sentinel-2 को स्पष्ट आकाशको तस्बिरले रडारको पानीसँग मिल्दो संकेत देखाएन",
        }.get(c, c)
    return {
        "low_valid_coverage": "only %s%% of the river corridor has reliable radar data" % _f(r.get("value")),
        "noisy_scene": "the image noise (%s dB) is too high" % _f(r.get("value")),
        "low_pair_coverage": "one scene does not cover the whole study area",
        "water_not_above_noise": "two pre-event scenes also showed %s sq km of 'water' (event: %s sq km), so it cannot be told apart from noise" % (_f(r.get("control")), _f(r.get("event"))),
        "no_control": "no control scene was available for a noise check",
        "optical_disagrees": "cloud-free Sentinel-2 imagery does not support the radar water",
    }.get(c, c)


def provenance_line(results, lang="en"):
    m = results["meta"]
    dem = "%s (%s m)" % (m.get("dem_source"), m.get("dem_native_resolution_m"))
    if lang == "ne":
        return ("स्रोत र समय: Sentinel-1 ट्र्याक %s (%s), अघि %s, पछि %s; DEM: %s; OSM स्न्यापसट %s; तयार: %s UTC; संस्करण %s।"
                % (m["orbit_track"], m["orbit_direction"], m["pre_date"], m["post_date"], dem, m["osm_snapshot"],
                   m["generated_utc"], m["software_version"]))
    return ("Sources and time: Sentinel-1 track %s (%s), before %s, after %s; DEM: %s; OSM snapshot %s; "
            "generated %s UTC; software %s." % (m["orbit_track"], m["orbit_direction"], m["pre_date"], m["post_date"],
                                                dem, m["osm_snapshot"], m["generated_utc"], m["software_version"]))


_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def display_name(item, lang):
    """Nepali text shows a Devanagari name if one exists, otherwise the English/Latin name (never Chinese script)."""
    if lang == "ne":
        for k in ("name_ne", "name"):
            if item.get(k) and _DEVANAGARI.search(item[k]):
                return item[k]
        return item.get("name_en") or item.get("name_ne") or item.get("name") or ""
    return item.get("name_en") or item.get("name") or ""


def _names(results, lang, key="possibly_cut_off_names"):
    out = [display_name(it, lang) for it in results["access"][key]]
    return ", ".join(x for x in out if x)


def drop_probably_non_nepal(places):
    """Heuristic: a place whose OSM name is written in Chinese characters is on the Tibetan side of the border.
    Returns (kept, dropped_names). Imperfect: a Nepali place with only a Chinese name would be dropped too."""
    kept, dropped = [], []
    for p in places:
        if _CJK.search(p.get("name", "")):
            dropped.append(p.get("name_en") or p.get("name"))
        else:
            kept.append(p)
    return kept, dropped


def report_sections(results, lang="en"):
    m, fl, d, a, ev = results["meta"], results["flood"], results["damage"], results["access"], results["evidence"]
    names = _names(results, lang)
    trunc = a["possibly_cut_off_names_truncated"]
    reasons = "; ".join(_reason_text(r, lang) for r in ev["reasons"] if r["code"] != "no_control")
    conf = ev.get("confidence")
    if lang == "ne":
        if ev["verdict"] == "water_detected":
            head = ("Sentinel-1 रडार तस्बिर (अघि: %s, पछि: %s) को परिवर्तन विश्लेषणले नदी करिडोरभित्र करिब %s वर्ग किमी भूभाग "
                    "खुला पानी जस्तो भएको संकेत गरेको छ। विश्वसनीयता: %s। यो सम्भावित बाढी क्षेत्र हो, पुष्टि भएको होइन।"
                    % (m["pre_date"], m["post_date"], _f(fl["water_km2"]), "मध्यम" if conf == "medium" else "कम"))
        elif ev["verdict"] == "no_water_detected":
            head = ("Sentinel-1 रडारले %s देखि %s बीच नयाँ खुला पानी जस्तो क्षेत्र देखाएन (%s वर्ग किमी)। यसको अर्थ बाढी नआएको भन्ने होइन: "
                    "साँघुरो खोला, घना वनस्पति, ठाडो भू-भाग, रडार छाया र %s मिटरको पिक्सेल आकारले बाढी लुकाउन सक्छ।"
                    % (m["pre_date"], m["post_date"], _f(fl["water_km2"]), _f(m["pixel_size_m"])))
        else:
            head = "प्रमाण अनिर्णायक छ, त्यसैले बाढी क्षेत्रको नतिजा देखाइएको छैन" + ((": " + reasons) if reasons else "") + "।"
        change = ("बस्ती क्षेत्र बाहेक, रडार प्रतिबिम्बन %s वर्ग किमीमा बढेको र %s वर्ग किमीमा घटेको देखियो; यी सतह परिवर्तनका सम्भावित क्षेत्र मात्र हुन् "
                  "(भग्नावशेष, माटो/बालुवा, भिजेको जमिन वा वनस्पति परिवर्तन हुन सक्छ), पुष्टि भएको भग्नावशेष होइन।"
                  % (_f(fl["change_increase_km2"]), _f(fl["change_decrease_km2"])))
        if fl["control_water_km2"] is not None:
            change += (" शोर जाँच (घटनाअघिका दुई तस्बिर): खुला पानी जस्तो %s वर्ग किमी र सतह परिवर्तन %s वर्ग किमी देखियो।"
                       % (_f(fl["control_water_km2"]), _f(fl["control_change_km2"])))
        change += (" बस्ती क्षेत्र %s वर्ग किमी र अस्थिर पिक्सेल %s वर्ग किमी सतह परिवर्तनका लागि जाँचिएन।"
                   % (_f(fl["builtup_not_assessed_km2"]), _f(fl["unstable_not_assessed_km2"])))
        damage = ("घटनाअघिको OpenStreetMap का %s भवन मध्ये %s भवन चिन्ह लगाइएको क्षेत्रभित्र पर्छन् (सम्भावित प्रभावित; क्षति पुष्टि भएको छैन)। "
                  "%s किमी नक्सांकित सडक मध्ये %s किमी र %s पुल मध्ये %s पुल चिन्हित पिक्सेलसँग मिल्छन् (सम्भावित प्रभावित; क्षति देखिएको होइन)।"
                  % (_f(d["buildings_total"]), _f(d["buildings_in_flagged_area"]), _f(d["roads_total_km"]),
                     _f(d["roads_in_flagged_area_km"]), _f(d["bridges_total"]), _f(d["bridges_possibly_hit"])))
        access = ("%s बस्ती जाँचिए। चिन्हित सबै पिक्सेल अवरुद्ध मान्दा %s बस्तीले नजिकको अस्पताल/शहरसम्मको नक्सांकित सडक सम्पर्क गुमाएको हुन सक्छ (माथिल्लो अनुमान); "
                  "सम्भावित खुला पानीले मात्र सडक अवरुद्ध गरेमा %s बस्ती (तल्लो अनुमान)। यो अनुमान हो, पुष्टि भएको अलगाव होइन।"
                  % (_f(a["settlements_total"]), _f(a["possibly_cut_off"]), _f(a["likely_cut_off"])))
        if names:
            access += " माथिल्लो अनुमानमा अलग हुन सक्ने बस्ती: %s।" % names
            if trunc:
                access += " (सूची छोटो पारिएको छ; पूर्ण सूची नक्सामा हेर्नुहोस्।)"
        access += (" %s बस्ती घटनाअघि नै सडक सञ्जालमा जोडिएका थिएनन् र %s बस्ती नक्सांकित सडकबाट टाढा भएकाले जाँच्न सकिएन।"
                   % (_f(a["unconnected_before_event"]), _f(a["not_on_road_network"])))
        limits = ("सीमा: Sentinel उपग्रहले केही दिनको अन्तरमा मात्र फेरि तस्बिर लिन्छ, त्यसैले अकस्मात घटनाको पूर्वसूचना दिन सक्दैन। "
                  "सडकको अवस्था प्रत्यक्ष हेरिएको छैन र OpenStreetMap अपूर्ण हुन सक्छ। नतिजा कुनै स्वतन्त्र सन्दर्भ नक्सासँग प्रमाणित गरिएको छैन। "
                  "यो शैक्षिक प्रोटोटाइप हो, सञ्चालनगत आपतकालीन उपकरण होइन; सबै नतिजा क्षेत्रमा पुष्टि गर्नुपर्छ।")
    else:
        if ev["verdict"] == "water_detected":
            head = ("Sentinel-1 radar change detection (before %s, after %s) found about %s sq km of ground that became open-water-like "
                    "inside the river corridor. Confidence: %s. This is a suspected flood extent, not a confirmed one."
                    % (m["pre_date"], m["post_date"], _f(fl["water_km2"]), conf or "low"))
        elif ev["verdict"] == "no_water_detected":
            head = ("Sentinel-1 radar did not detect newly open-water-like ground between %s and %s (%s sq km). This does not prove there was no "
                    "flooding: narrow channels, dense vegetation, steep terrain, radar shadow and the %s m pixel size can hide it."
                    % (m["pre_date"], m["post_date"], _f(fl["water_km2"]), _f(m["pixel_size_m"])))
        else:
            head = "The evidence is inconclusive, so no flood extent is claimed" + ((": " + reasons) if reasons else "") + "."
        change = ("Outside built-up areas, radar backscatter increased over %s sq km and decreased over %s sq km. These are surface-change candidates only "
                  "(possibly debris, sediment, wet ground or vegetation change), not confirmed debris."
                  % (_f(fl["change_increase_km2"]), _f(fl["change_decrease_km2"])))
        if fl["control_water_km2"] is not None:
            change += (" Noise check (two pre-event scenes): %s sq km of 'water' and %s sq km of surface change were flagged."
                       % (_f(fl["control_water_km2"]), _f(fl["control_change_km2"])))
        change += (" %s sq km of built-up area and %s sq km of unstable pixels were not assessed for surface change."
                   % (_f(fl["builtup_not_assessed_km2"]), _f(fl["unstable_not_assessed_km2"])))
        damage = ("Using pre-event OpenStreetMap, %s of %s mapped buildings have their centre inside the flagged area (potentially affected, not confirmed damaged). "
                  "%s km of %s km of mapped road and %s of %s mapped bridges intersect flagged pixels (potentially affected; no damage was observed)."
                  % (_f(d["buildings_in_flagged_area"]), _f(d["buildings_total"]), _f(d["roads_in_flagged_area_km"]),
                     _f(d["roads_total_km"]), _f(d["bridges_possibly_hit"]), _f(d["bridges_total"])))
        access = ("Road-network analysis checked %s settlements. If every flagged pixel were impassable, %s may lose their mapped road link to the nearest "
                  "hospital or town (upper bound); if only suspected open water blocks roads, %s (lower bound). This is an estimate, not confirmed isolation."
                  % (_f(a["settlements_total"]), _f(a["possibly_cut_off"]), _f(a["likely_cut_off"])))
        if names:
            access += " Possibly cut off (upper bound): %s." % names
            if trunc:
                access += " (List shortened; see the dashboard for all.)"
        access += (" %s settlements were already unconnected in the pre-event road map, and %s were too far from any mapped road to test."
                   % (_f(a["unconnected_before_event"]), _f(a["not_on_road_network"])))
        limits = ("Limits: Sentinel satellites revisit an area only every few days, so they cannot warn of a sudden event. Road condition on the ground was not "
                  "observed and OpenStreetMap is incomplete in remote areas. The results have not been validated against an independent reference map. "
                  "This is an educational prototype, not an operational tool; results need field confirmation.")
    if ev["verdict"] == "inconclusive":
        if lang == "ne":
            damage = "प्रमाण अनिर्णायक भएकाले तलका आँकडा चिन्हित पिक्सेल साँच्चै प्रभावित भए मात्र लागू हुन्छन्। " + damage
            access = "प्रमाण अनिर्णायक भएकाले तलका आँकडा सशर्त मात्र हुन्। " + access
        else:
            damage = "Because the evidence is inconclusive, the figures below hold only if the flagged pixels are real. " + damage
            access = "Because the evidence is inconclusive, these figures are conditional only. " + access
    return {"flood": head, "change": change, "damage": damage, "access": access, "limits": limits,
            "provenance": provenance_line(results, lang)}


def report_text(results, lang="en"):
    s = report_sections(results, lang)
    return "\n\n".join([s["flood"], s["change"], s["damage"], s["access"], s["limits"], s["provenance"]])


_DEV = str.maketrans("\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f", "0123456789")
_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?", re.ASCII)


def extract_numbers(text):
    t = str(text).translate(_DEV)
    return [float(x.replace(",", "")) for x in _NUM.findall(t)]


def allowed_numbers(results):
    base = set(extract_numbers(json.dumps(results, ensure_ascii=False, default=str)))

    def walk(v):
        if isinstance(v, float):
            for k in (0, 1, 2, 3):
                base.add(round(v, k))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
    walk(results)
    return base


def check_numbers(text, results):
    """Return the numbers in `text` that do not appear in the results (empty list = clean)."""
    allowed = allowed_numbers(results)
    return [x for x in extract_numbers(text) if not any(abs(x - a) < 1e-6 for a in allowed)]


_BANNED_EN = re.compile(r"\b(destroyed|collapsed|washed away|swept away|confirmed damage[d]?|(is|are|were|was) damaged|"
                        r"(is|are|were|was) (cut off|isolated)|definitely|certainly)\b", re.I)
_BANNED_NE = re.compile(r"(ध्वस्त|नष्ट भए|भत्किए|बगाए|निश्चित रूपमा)")
_HEDGE = re.compile(r"(estimate|potential|possible|possibly|may |might|suspected|unconfirmed|not confirmed|inconclusive|"
                    r"field confirmation|सम्भाव|अनुमान|पुष्टि|अनिर्णायक)", re.I)


def check_claims(text, lang="en"):
    """Wording problems in a model answer: over-claims, or no sign of uncertainty."""
    issues = []
    if _BANNED_EN.search(text) or _BANNED_NE.search(text):
        issues.append("over-claim wording (damage/isolation stated as fact)")
    if not _HEDGE.search(text):
        issues.append("no uncertainty wording")
    return issues


def answer_with_templates(question, results, lang="en"):
    """No-AI fallback: pick the right computed paragraph(s) by keywords. Always ends with sources and limits."""
    q = question.lower()
    s = report_sections(results, lang)
    picks = []
    if any(k in q for k in ["cut", "isolat", "settlement", "village", "reach", "hospital", "access", "अलग", "बस्ती", "अस्पताल", "गाउँ"]):
        picks.append(s["access"])
    if any(k in q for k in ["building", "house", "road", "bridge", "damage", "भवन", "घर", "सडक", "पुल", "क्षति"]):
        picks.append(s["damage"])
    if any(k in q for k in ["flood", "water", "area", "debris", "extent", "change", "बाढी", "पानी", "क्षेत्र", "भग्नावशेष", "परिवर्तन"]):
        picks.append(s["flood"])
        picks.append(s["change"])
    if any(k in q for k in ["reliab", "trust", "accur", "confiden", "evidence", "valid", "sure", "भरपर्दो", "विश्वास", "प्रमाण", "सही"]):
        picks.append(s["flood"])
    if any(k in q for k in ["limit", "cannot", "when", "date", "source", "सीमा", "मिति", "स्रोत"]):
        pass
    if not picks:
        return report_text(results, lang)
    uniq = []
    for p in picks + [s["limits"], s["provenance"]]:
        if p not in uniq:
            uniq.append(p)
    return "\n\n".join(uniq)


def _llm_call(api_key, model, system, question):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model, "max_tokens": 800, "system": system, "messages": [{"role": "user", "content": question}]},
        timeout=90)
    if r.status_code != 200:
        raise RuntimeError("LLM request failed (HTTP %s): %s" % (r.status_code, r.text[:300]))
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


def ask_copilot(question, results, lang="en", api_key=None, model="claude-sonnet-5-5", llm=None):
    """Answer a responder's question. With an API key a language model writes the answer, but it is REJECTED
    (and replaced by the template answer) if it contains a number that is not in results.json, states damage
    or isolation as fact, or shows no uncertainty. Without a key the template answer is used."""
    if not api_key and llm is None:
        return {"answer": answer_with_templates(question, results, lang), "source": "template", "rejected": []}
    call = llm if llm is not None else (lambda sys_prompt, q: _llm_call(api_key, model, sys_prompt, q))
    language = "Nepali" if lang == "ne" else "English"
    system = (
        "You are the FloodResQ situation-report assistant, an educational prototype for flood mapping. "
        "Answer ONLY from the JSON facts below. Copy numbers exactly as written; never calculate, round, estimate or add any "
        "number that is not in the JSON. Never say that a building, road or bridge IS damaged or that a village IS cut off: "
        "use 'potentially affected' and 'may have lost its road link'. If evidence.verdict is 'inconclusive' or "
        "'no_water_detected', say so plainly and do not suggest a flood extent. State that results are unvalidated "
        "satellite-based estimates that need field confirmation, and mention the data dates (meta). "
        "If the facts do not answer the question, say you do not have that information. "
        "Do not number your sentences or use numbered lists. Reply in %s, in 7 sentences or fewer.\n\nFACTS:\n%s"
        % (language, json.dumps(results, ensure_ascii=False, default=str)))
    rejected = []
    for _ in range(2):
        text = call(system, question)
        bad = check_numbers(text, results)
        issues = check_claims(text, lang)
        if not bad and not issues:
            return {"answer": text + "\n\n" + provenance_line(results, lang), "source": "llm", "rejected": rejected}
        rejected.append({"numbers": bad, "issues": issues})
        system += "\n\nYour previous answer was rejected (numbers not in facts: %s; wording: %s). Rewrite it." % (bad, issues)
    return {"answer": answer_with_templates(question, results, lang), "source": "template_after_rejection",
            "rejected": rejected}


# --------------------------------------------------------------------------
# Dashboard and one-page report
# --------------------------------------------------------------------------
def _mask_geojson(mask, grid, max_features=4000):
    feats = []
    try:
        from rasterio import features
        for geom, _ in features.shapes(mask.astype("uint8"), mask=mask, transform=grid.transform()):
            feats.append({"type": "Feature", "properties": {}, "geometry": geom})
            if len(feats) >= max_features:
                break
    except ImportError:   # fallback without rasterio: one rectangle per horizontal run of flagged pixels
        lon, lat = grid.centers()
        dx = (grid.east - grid.west) / grid.width / 2
        dy = (grid.north - grid.south) / grid.height / 2
        for r in range(mask.shape[0]):
            row = mask[r]
            if not row.any():
                continue
            d = np.diff(np.concatenate([[0], row.astype(int), [0]]))
            for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                x0, x1, y0, y1 = lon[a] - dx, lon[b - 1] + dx, lat[r] - dy, lat[r] + dy
                feats.append({"type": "Feature", "properties": {}, "geometry": {
                    "type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}})
                if len(feats) >= max_features:
                    return {"type": "FeatureCollection", "features": feats}, True
    return {"type": "FeatureCollection", "features": feats}, len(feats) >= max_features


def _png_uri(arr, kind="gray", vmin=-25.0, vmax=0.0, max_px=900):
    """Small PNG data-URI for a map overlay (NaN = transparent)."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    step = max(1, int(math.ceil(max(arr.shape[0], arr.shape[1]) / float(max_px))))
    a = arr[::step, ::step]
    if kind == "rgb":
        rgba = np.concatenate([np.clip(a[..., :3], 0, 1), np.ones(a.shape[:2] + (1,))], axis=-1)
    else:
        g = np.clip((a - vmin) / (vmax - vmin), 0, 1)
        rgba = np.stack([g, g, g, np.where(np.isfinite(a), 1.0, 0.0)], axis=-1)
    buf = io.BytesIO()
    plt.imsave(buf, np.nan_to_num(rgba), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


STATUS_COLOR = {"likely_cut_off": "#8b0000", "possibly_cut_off": "red", "reachable": "green",
                "unconnected_before_event": "gray", "not_on_road_network": "lightgray"}


def settlement_display_status(possible_rows, likely_rows):
    likely = {r["osm_id"] for r in likely_rows if r["status"] == "cut_off"}
    out = []
    for r in possible_rows:
        if r["status"] == "cut_off":
            out.append("likely_cut_off" if r["osm_id"] in likely else "possibly_cut_off")
        else:
            out.append(r["status"])
    return out


def make_dashboard(path, grid, masks, osm, impacts, access, targets, results, imagery=None, routes=None):
    """Interactive map. masks: dict of boolean arrays (water, change_increase, change_decrease, permanent_water,
    not_assessed). impacts: {'blocked_possible', 'blocked_likely', 'bridge_rows', 'bld'}.
    access: {'possible': {...rows...}, 'likely': {...}}. imagery: {label: {'array','kind','vmin','vmax'}}."""
    import folium
    from folium import raster_layers
    w, s, e, n = grid.bbox
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=11, tiles=None)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri, Maxar, Earthstar Geographics", name="Basemap: satellite (Esri)").add_to(m)
    folium.TileLayer(tiles="https://tile.opentopomap.org/{z}/{x}/{y}.png",
                     attr="Map data &copy; OpenStreetMap contributors, SRTM | Style &copy; OpenTopoMap (CC-BY-SA)",
                     name="Basemap: terrain (OpenTopoMap)", max_zoom=17).add_to(m)
    notes = []
    for label, im in (imagery or {}).items():
        uri = _png_uri(im["array"], im.get("kind", "gray"), im.get("vmin", -25.0), im.get("vmax", 0.0))
        raster_layers.ImageOverlay(image=uri, bounds=[[s, w], [n, e]], opacity=0.85, name=label, show=False).add_to(m)
    layer_spec = [
        ("water", "Suspected open-water flood (radar)", "#1e90ff", True),
        ("change_increase", "Surface change: backscatter increase (candidate)", "#ff8c00", True),
        ("change_decrease", "Surface change: backscatter decrease (candidate)", "#8e44ad", True),
        ("permanent_water", "Permanent / pre-existing water", "#00bcd4", False),
        ("not_assessed", "Not assessed (built-up or unstable pixels)", "#888888", False),
    ]
    for key, label, color, show in layer_spec:
        mask = masks.get(key)
        if mask is not None and mask.any():
            gj, trunc = _mask_geojson(mask, grid)
            folium.GeoJson(gj, name=label, show=show,
                           style_function=lambda x, c=color: {"color": c, "fillColor": c, "weight": 1, "fillOpacity": 0.5}).add_to(m)
            if trunc:
                notes.append(label + ": shape list truncated for display")
    major = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"}
    road_feats = [{"type": "Feature", "properties": {},
                   "geometry": {"type": "LineString", "coordinates": [list(c) for c in r["coords"]]}}
                  for r in osm["roads"] if r["tags"].get("highway") in major]
    if road_feats:
        folium.GeoJson({"type": "FeatureCollection", "features": road_feats}, name="Roads (pre-event OSM)",
                       style_function=lambda x: {"color": "#555555", "weight": 1.5}).add_to(m)
    for key, label, color, weight in (("blocked_possible", "Road segments in any flagged pixel (upper bound)", "red", 4),
                                      ("blocked_likely", "Road segments in suspected open water (lower bound)", "#8b0000", 6)):
        feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [list(a), list(b)]}}
                 for a, b in impacts.get(key, [])]
        if feats:
            folium.GeoJson({"type": "FeatureCollection", "features": feats}, name=label,
                           style_function=lambda x, c=color, wt=weight: {"color": c, "weight": wt}).add_to(m)
    if routes:
        feats = [{"type": "Feature", "properties": {"name": r["name"]},
                  "geometry": {"type": "LineString", "coordinates": [list(c) for c in r["coords"]]}} for r in routes]
        folium.GeoJson({"type": "FeatureCollection", "features": feats},
                       name="Pre-event route to nearest hospital/town for possibly cut-off settlements",
                       style_function=lambda x: {"color": "#1f4fd8", "weight": 2, "dashArray": "6 6"}).add_to(m)
    bridge_by_id = {r["way_id"]: r for r in impacts.get("bridge_rows", [])}
    fg = folium.FeatureGroup(name="Bridges (red = intersects flagged pixels; not proof of damage)")
    for b in osm["bridges"]:
        info = bridge_by_id.get(b["id"])
        coords = [[c[1], c[0]] for c in b["coords"]]
        folium.PolyLine(coords, color="red" if info and info["possibly_hit"] else "purple", weight=5,
                        tooltip=(b["tags"].get("name") or "bridge")).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Buildings inside flagged area (potentially affected)", show=False)
    bld = impacts.get("bld", {"hit_points": []})
    for lon, lat in bld["hit_points"][:1500]:
        folium.CircleMarker([lat, lon], radius=2, color="#c71585", fill=True).add_to(fg)
    fg.add_to(m)
    if len(bld["hit_points"]) > 1500:
        notes.append("Buildings layer shows the first 1500 buildings inside the flagged area")
    fg = folium.FeatureGroup(name="Settlements (colour = accessibility status)")
    pos, lik = access["possible"]["rows"], access["likely"]["rows"]
    for r, st in zip(pos, settlement_display_status(pos, lik)):
        tip = "%s (%s)" % (r.get("name_en") or r["name"], st)
        pop = html.escape(tip) + ((("<br>" + html.escape(r["name_ne"])) if r.get("name_ne") else ""))
        folium.CircleMarker([r["lat"], r["lon"]], radius=6, color=STATUS_COLOR.get(st, "blue"), fill=True,
                            fill_opacity=0.9, tooltip=tip, popup=pop).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name="Hospitals / towns used as targets")
    for t in targets:
        folium.Marker([t["lat"], t["lon"]], tooltip=t["name"], icon=folium.Icon(color="blue", icon="plus")).add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    f, a, ev, mt = results["flood"], results["access"], results["evidence"], results["meta"]
    verdict = {"water_detected": "Suspected flood water detected (unvalidated)",
               "no_water_detected": "No newly open-water-like ground detected (does not prove no flooding)",
               "inconclusive": "INCONCLUSIVE - no flood extent claimed"}[ev["verdict"]]
    reasons = "; ".join(_reason_text(r, "en") for r in ev["reasons"] if r["code"] != "no_control")
    info = (
        "<div style='position:fixed;bottom:8px;left:8px;z-index:9999;background:white;padding:8px;"
        "font:12px sans-serif;max-width:470px;border:1px solid #888;max-height:46vh;overflow:auto'>"
        "<b>FloodResQ AI</b> - educational prototype, not operational. Results are unvalidated estimates.<br>"
        "<b>Verdict:</b> %s%s<br>"
        "Water: %s sq km | Change up/down: %s / %s sq km | Noise check (water / change): %s / %s sq km<br>"
        "Possibly cut off: %s (lower bound %s) of %s settlements<br>"
        "<b>Dates:</b> radar before %s, after %s (track %s, %s); OSM as of %s; generated %s UTC<br>"
        "<b>Sources:</b> %s %s<br>DEM: %s (%s m). Basemaps: Esri / OpenTopoMap.<br>"
        "<b>Limits:</b> %s%s</div>"
        % (html.escape(verdict), (" (" + html.escape(reasons) + ")") if reasons else "",
           _f(f["water_km2"]), _f(f["change_increase_km2"]), _f(f["change_decrease_km2"]),
           _f(f["control_water_km2"]), _f(f["control_change_km2"]),
           _f(a["possibly_cut_off"]), _f(a["likely_cut_off"]), _f(a["settlements_total"]),
           mt["pre_date"], mt["post_date"], mt["orbit_track"], mt["orbit_direction"], mt["osm_snapshot"], mt["generated_utc"],
           html.escape(ATTRIBUTION[0]), html.escape(ATTRIBUTION[2] + " " + ATTRIBUTION[1]), html.escape(str(mt["dem_source"])),
           mt["dem_native_resolution_m"], html.escape(results["limitations"][1]),
           (("<br>" + html.escape("; ".join(notes))) if notes else "")))
    sw = lambda c, t: "<div><span style='display:inline-block;width:12px;height:12px;background:%s;margin-right:6px'></span>%s</div>" % (c, t)
    legend = (
        "<div style='position:fixed;bottom:8px;right:8px;z-index:9999;background:white;padding:8px;font:12px sans-serif;"
        "border:1px solid #888'><b>Legend</b>"
        + sw("#1e90ff", "Suspected open water") + sw("#ff8c00", "Backscatter increase (candidate)")
        + sw("#8e44ad", "Backscatter decrease (candidate)") + sw("#8b0000", "Likely cut off (water only)")
        + sw("red", "Possibly cut off (any flagged pixel)") + sw("green", "Reachable") + sw("gray", "Already unconnected before event")
        + sw("lightgray", "Not near a mapped road") + sw("#1f4fd8", "Pre-event route (dashed)") + "</div>")
    m.get_root().html.add_child(folium.Element(info))
    m.get_root().html.add_child(folium.Element(legend))
    m.save(path)
    return path


def make_report_html(path, results):
    en = report_sections(results, "en")
    ne = report_sections(results, "ne")
    m, ev = results["meta"], results["evidence"]
    esc = html.escape
    color = {"water_detected": "#e8f1ff", "no_water_detected": "#eef7ee", "inconclusive": "#fff3cd"}[ev["verdict"]]
    page = """<!doctype html><html><head><meta charset="utf-8"><title>FloodResQ AI situation report</title>
<style>
@page { size: A4; margin: 9mm; }
body { font: 8.5pt/1.28 'Noto Sans','Noto Sans Devanagari',Arial,sans-serif; color:#111; }
h1 { font-size: 12.5pt; margin: 0 0 2pt; } h2 { font-size: 9.5pt; margin: 5pt 0 2pt; border-bottom: 1px solid #999; }
p { margin: 2pt 0; } .meta { color:#333; font-size: 7.5pt; } .small { font-size: 6.8pt; color:#444; }
.warn { background:#fff3cd; padding:3pt; border:1px solid #d9b100; } .verdict { background:%s; padding:3pt; border:1px solid #999; }
</style></head><body>
<h1>FloodResQ AI - Situation Report / स्थिति प्रतिवेदन</h1>
<p class="meta">Event date %s | Radar track %s (%s) | Before %s, after %s | OSM snapshot %s | Generated %s UTC | v%s</p>
<p class="warn">Educational prototype, not an operational tool. Unvalidated estimates; potentially affected does not mean damaged. All numbers were computed by the system from satellite and map data, none typed by hand.</p>
<h2>English</h2><p class="verdict">%s</p><p>%s</p><p>%s</p><p>%s</p><p>%s</p><p class="small">%s</p>
<h2 lang="ne">नेपाली</h2><p lang="ne" class="verdict">%s</p><p lang="ne">%s</p><p lang="ne">%s</p><p lang="ne">%s</p><p lang="ne">%s</p><p lang="ne" class="small">%s</p>
<h2>Attribution</h2>
<p class="small">%s<br>%s<br>%s<br>%s</p>
</body></html>""" % (
        color, esc(m["event_date"]), esc(str(m["orbit_track"])), esc(m["orbit_direction"]), esc(m["pre_date"]),
        esc(m["post_date"]), esc(m["osm_snapshot"]), esc(m["generated_utc"]), esc(m["software_version"]),
        esc(en["flood"]), esc(en["change"]), esc(en["damage"]), esc(en["access"]), esc(en["limits"]), esc(en["provenance"]),
        esc(ne["flood"]), esc(ne["change"]), esc(ne["damage"]), esc(ne["access"]), esc(ne["limits"]), esc(ne["provenance"]),
        esc(ATTRIBUTION[0]), esc(ATTRIBUTION[1]), esc(ATTRIBUTION[2]),
        esc("Basemap tiles: Esri, Maxar, Earthstar Geographics; OpenTopoMap (CC-BY-SA)."))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def bundle_outputs(zip_path, files):
    """Zip every existing file in `files` (missing ones are listed, not silently skipped)."""
    import zipfile
    missing = [f for f in files if not os.path.exists(f)]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f)
    return missing



# ==========================================================================
# v0.2.1: pre-event OpenStreetMap, VERIFIED ELEMENT BY ELEMENT
# --------------------------------------------------------------------------
# Why this exists: the public ohsome server refuses its shape-download endpoints (HTTP 403 from Colab), and an
# Overpass server may silently ignore the [date:] setting. So this code never trusts a server's claim. It
#   1. probes each Overpass server: does the road count change with the date (2020 vs snapshot)?
#      do the returned elements carry edit timestamps not newer than the snapshot?
#   2. downloads with "out meta" so EVERY element carries its last-edit timestamp, and refuses the whole piece
#      if any element is newer than the snapshot day (that would be a post-event edit = disqualification risk).
# If no server passes, it STOPS with an explanation. It never falls back to current OSM data silently.
# Not tested against live servers by the author (no internet); tested against simulated servers only.
# ==========================================================================
OSM_VERIFIED_CACHE_DIR = "osm_cache_v2"
_PROBE_LAST = {}


def _snapshot_limit(snapshot):
    return datetime.strptime(snapshot, "%Y-%m-%d") + timedelta(days=1, minutes=1)


def _parse_ts(ts):
    try:
        return datetime.strptime(str(ts)[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def _op_post(url, query, retries=3, timeout=300):
    """POST an Overpass query; returns (json, error_text). Backs off when the server is busy."""
    err = ""
    for k in range(retries):
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except requests.exceptions.RequestException as ex:
            err = type(ex).__name__
            _sleep(10 * (k + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json(), ""
            except ValueError:
                return None, "reply was not JSON"
        err = "HTTP %s" % r.status_code
        if r.status_code in (429, 502, 503, 504):
            _sleep(20 * (k + 1))
            continue
        break
    return None, err


def _dated(snapshot):
    return '[out:json][timeout:180][date:"%sT00:00:00Z"]' % snapshot


def probe_overpass_server(url, bbox, snapshot):
    """Test whether a server really answers for the snapshot date. Returns a dict with ok True/False and why."""
    host = url.split("/")[2]
    w, s, e, n = bbox
    cx, cy = (w + e) / 2.0, (s + n) / 2.0
    hw, hh = (e - w) * 0.2, (n - s) * 0.2
    sub = (cx - hw, cy - hh, cx + hw, cy + hh)           # central 40% of the box: light but representative
    box = "%s,%s,%s,%s" % (sub[1], sub[0], sub[3], sub[2])
    out = {"server": host, "ok": False, "why": ""}

    def count(date_setting):
        q = date_setting + ';way["highway"](%s);out count;' % box
        js, err = _op_post(url, q)
        if js is None:
            return None, err
        try:
            return int(js["elements"][0]["tags"]["ways"]), ""
        except Exception:
            return None, "unexpected count reply"

    c_snap, err = count(_dated(snapshot))
    if c_snap is None:
        out["why"] = "no usable answer (%s)" % err
        return out
    c_old, err = count(_dated("2020-01-01"))
    if c_old is None:
        out["why"] = "no usable answer for the 2020 test (%s)" % err
        return out
    out["roads_at_snapshot"], out["roads_at_2020"] = c_snap, c_old
    if c_snap < 20:
        out["why"] = "too few roads in the test area (%d) to judge" % c_snap
        return out
    if not c_old < c_snap:
        out["why"] = "road count did not change between 2020 and the snapshot (%d vs %d): the date looks ignored" % (c_old, c_snap)
        return out
    hw2, hh2 = (e - w) * 0.04, (n - s) * 0.04
    tiny = "%s,%s,%s,%s" % (cy - hh2, cx - hw2, cy + hh2, cx + hw2)   # very small box: light query
    js, err = _op_post(url, _dated(snapshot) + ';way["highway"](%s);out meta 400;' % tiny, retries=2)
    if js is None:
        # Busy server: the date test above already passed, and EVERY downloaded element is checked individually
        # later, so this is allowed to be inconclusive (it is recorded in the results).
        out["timestamp_test"] = "inconclusive (%s)" % err
        out["ok"] = True
        out["why"] = "date test passed (road count differs by date); timestamp test inconclusive (%s)" % err
        return out
    stamps = [_parse_ts(el.get("timestamp")) for el in js.get("elements", []) if el.get("timestamp")]
    stamps = [t for t in stamps if t is not None]
    if not stamps:
        out["why"] = "server returned no edit timestamps, so it cannot be verified"
        return out
    newest = max(stamps)
    out["newest_edit_in_test"] = newest.strftime("%Y-%m-%dT%H:%M:%SZ")
    if newest > _snapshot_limit(snapshot):
        out["why"] = "contains edits from %s, newer than the snapshot" % out["newest_edit_in_test"]
        return out
    out["timestamp_test"] = "passed"
    out["ok"] = True
    out["why"] = "passed (road count differs by date; newest edit %s)" % out["newest_edit_in_test"]
    return out


def _verify_elements(elements, snapshot, label):
    limit = _snapshot_limit(snapshot)
    newest, missing = None, 0
    for el in elements:
        t = _parse_ts(el.get("timestamp"))
        if t is None:
            missing += 1
            continue
        if t > limit:
            raise RuntimeError("'%s' contains an element last edited %s, newer than the snapshot %s. "
                               "Refusing to use it (post-event edits are not allowed)." % (label, el.get("timestamp"), snapshot))
        newest = t if newest is None or t > newest else newest
    return newest, missing


class _PieceFailed(RuntimeError):
    """Servers were busy/unreachable for this piece (NOT a verification failure)."""


def _fetch_piece(servers, bbox, snapshot, body, out_stmt, label):
    w, s, e, n = bbox
    stmt = body.format(s=s, w=w, n=n, e=e)
    query = _dated(snapshot) + ";(%s);%s" % (stmt, out_stmt)
    os.makedirs(OSM_VERIFIED_CACHE_DIR, exist_ok=True)
    errors = []
    for url in servers:
        host = url.split("/")[2]
        cache = os.path.join(OSM_VERIFIED_CACHE_DIR, hashlib.sha256((host + query).encode()).hexdigest()[:24] + ".json")
        if os.path.exists(cache):
            try:
                with open(cache, encoding="utf-8") as f:
                    els = json.load(f)
                newest, missing = _verify_elements(els, snapshot, label)   # cached data is re-verified, never trusted
                logger.info("  [cache] %s: %d elements", label, len(els))
                return els, host, newest, missing
            except (ValueError, OSError):
                pass
        js, err = _op_post(url, query)
        if js is None:
            errors.append("%s: %s" % (host, err))
            continue
        els = js.get("elements", [])
        newest, missing = _verify_elements(els, snapshot, label)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(els, f)
        logger.info("  [%s] %s: %d elements", host, label, len(els))
        return els, host, newest, missing
    raise _PieceFailed("No server answered for '%s' (%s). Run the cell again later; finished pieces are cached."
                       % (label, "; ".join(errors) or "no details"))


def fetch_osm(bbox, snapshot, servers=None):
    """Pre-event OSM (as of `snapshot`), verified element by element. Same return structure as v0.2."""
    servers = list(servers or OVERPASS_SERVERS)
    probes = [probe_overpass_server(u, bbox, snapshot) for u in servers]
    _PROBE_LAST["probes"] = probes
    for p in probes:
        logger.info("  probe %-28s %s", p["server"], ("OK: " if p["ok"] else "REJECTED: ") + p["why"])
    good = [u for u, p in zip(servers, probes) if p["ok"]]
    if not good:
        raise RuntimeError(
            "No Overpass server could be shown to answer for the snapshot date %s, so NO OSM data was downloaded "
            "(using current data could include post-event edits, which the rules forbid). Probe results: %s"
            % (snapshot, "; ".join("%s -> %s" % (p["server"], p["why"]) for p in probes)))
    stats = {"used": set(), "newest": [], "missing": 0}

    def tiled(body, out_stmt, nx, ny, label):
        seen, result = set(), []
        tiles = split_bbox(bbox, nx, ny)

        def get(tile, lab, depth):
            try:
                return [_fetch_piece(good, tile, snapshot, body, out_stmt, lab)]
            except _PieceFailed:
                if depth >= 2:
                    raise
                logger.info("  %s: servers struggled; retrying as 4 smaller pieces", lab)
                parts = []
                for j, sub in enumerate(split_bbox(tile, 2, 2), 1):
                    parts.extend(get(sub, "%s.%d" % (lab, j), depth + 1))
                return parts

        for k, tile in enumerate(tiles, 1):
            for els, host, newest, missing in get(tile, "%s %d/%d" % (label, k, len(tiles)), 0):
                stats["used"].add(host)
                stats["missing"] += missing
                if newest:
                    stats["newest"].append(newest)
                for el in els:
                    key = (el.get("type"), el.get("id"))
                    if key not in seen:
                        seen.add(key)
                        result.append(el)
        return result

    logger.info("Rivers...")
    rivers = parse_ways(tiled('way["waterway"="river"]({s},{w},{n},{e});', "out geom meta;", 2, 2, "rivers"))
    logger.info("Roads...")
    roads = parse_ways(tiled('way["highway"~"^(%s)$"]({s},{w},{n},{e});' % ROAD_TYPES, "out geom meta;", 3, 3, "roads"))
    logger.info("Buildings...")
    b_el = tiled('nwr["building"]({s},{w},{n},{e});', "out center meta qt;", 4, 4, "buildings")
    buildings = [element_point(el) for el in b_el if element_point(el) is not None]
    logger.info("Places and hospitals...")
    places = parse_places(tiled('nwr["place"~"^(city|town|village|hamlet)$"]["name"]({s},{w},{n},{e});',
                                "out center meta;", 1, 1, "places"))
    hospitals = parse_places(tiled('nwr["amenity"="hospital"]({s},{w},{n},{e});nwr["healthcare"="hospital"]({s},{w},{n},{e});',
                                   "out center meta;", 1, 1, "hospitals"))
    places, dropped_p = drop_probably_non_nepal(places)
    hospitals, dropped_h = drop_probably_non_nepal(hospitals)
    bridges = [w for w in roads if w["tags"].get("bridge", "no") not in ("no", "")]
    verified = stats["missing"] == 0 and len(stats["newest"]) > 0
    newest = max(stats["newest"]).strftime("%Y-%m-%dT%H:%M:%SZ") if stats["newest"] else None
    meta = {"snapshot": snapshot, "servers": sorted(stats["used"]), "newest_data_timestamp": newest,
            "snapshot_verified": bool(verified),
            "probes": probes,
            "probe_timestamp_tests": [p.get("timestamp_test") for p in probes if p["ok"]],
            "excluded_probably_outside_nepal": dropped_p + dropped_h,
            "note": ("Every downloaded element carries an edit timestamp not newer than the snapshot day, and the "
                     "server passed a date test." if verified else
                     "Some elements had no edit timestamp, so they could not be checked individually.")}
    return {"rivers": rivers, "roads": roads, "buildings": buildings, "places": places,
            "hospitals": hospitals, "bridges": bridges, "meta": meta}



# ==========================================================================
# v0.2.3: scan every same-track pair for open water (diagnosis, not tuning)
# ==========================================================================
def pair_water_scan(client, grid, pairs, ctx, params=None, pad_s=60, fetch=None):
    """Run the water detector on EVERY before/after pair and show where candidates are lost.
    Uses no control scene and no reference map; it only shows what each pair's radar contains."""
    import pandas as pd
    fetch = fetch or fetch_s1_scene
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    near = ctx["river_dist_m"] <= p["water_corridor_m"]
    rows = []
    for _, pr in pairs.iterrows():
        row = {"track": pr["rel_orbit"], "direction": pr["direction"],
               "before": str(pr["pre_date"]), "after": str(pr["post_date"])}
        try:
            a = fetch(client, grid, pr["pre_scene"], pr["direction"], pad_s=pad_s)
            b = fetch(client, grid, pr["post_scene"], pr["direction"], pad_s=pad_s)
            det = detect_flood(a, b, ctx, None, p)
        except Exception as ex:
            row["error"] = str(ex)[:150]
            rows.append(row)
            continue
        f = funnel(det)
        cum = [r["pixels_passing_all_so_far"] for r in f]
        worst, worst_loss = None, 0
        for i in range(2, 8):                       # the tests after "valid" and "near a river"
            loss = cum[i - 1] - cum[i]
            if loss > worst_loss:
                worst, worst_loss = f[i]["test"], loss
        v = det["post_vv_db"][det["valid"] & near]
        d = det["diff_vv_db"][det["valid"] & near]
        v, d = v[np.isfinite(v)], d[np.isfinite(d)]
        row.update({
            "water_km2": round(area_km2(det["water"], grid), 3),
            "corridor_px": int(f[1]["pixels_passing_this_test_alone"]),
            "dark_after_px": cum[2], "also_dark_vh_px": cum[3], "and_not_dark_before_px": cum[4],
            "and_dropped_px": cum[5], "and_gentle_slope_px": cum[6], "and_low_above_river_px": cum[7],
            "water_px_final": cum[8],
            "biggest_loss_at": worst,
            "post_vv_p5_db": None if not v.size else round(float(np.percentile(v, 5)), 1),
            "post_vv_median_db": None if not v.size else round(float(np.median(v)), 1),
            "vv_change_p1_db": None if not d.size else round(float(np.percentile(d, 1)), 1),
        })
        rows.append(row)
    return pd.DataFrame(rows)
