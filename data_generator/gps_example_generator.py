#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS Distance Dataset Example Generator (JSON-driven)
---------------------------------------------------
• Generates (system, user, assistant[analysis], assistant[final]) examples to teach geodesic reasoning.
• ALL phrasing/variants are loaded from JSON files in the same folder (see PhraseBank.REQUIRED_FILES).
• 50/50 random choice of exact method:
    - Haversine (spherical great-circle with IUGG mean Earth radius)
    - Vincenty inverse (WGS‑84 ellipsoid)
• “Analysis” includes a deliberately coarse gut‑check (intuitive ballpark) and a brief English‑first overview
  of the chosen exact method. The final “content” then gives the numeric trail and conversions.

Run:
  python3 gps_example_generator.py --json-only
  python3 gps_example_generator.py --n 10 --out gps_samples.parquet

Requires:
  - Python 3.8+
  - pyarrow, pandas (only if you want Parquet output/preview)

Notes:
  - Mean Earth radius uses IUGG mean radius ≈ 6371.0088 km.
  - Length-per-degree formulas for the gut-check (short spans) come from standard approximations.
  - Vincenty may fail near antipodes; we fall back to Haversine in that rare case.

(External references for formulas/constants are included in the accompanying message text.)
"""

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Global configuration/constants
# -----------------------------
EARTH_RADIUS_KM = 6371.0088      # IUGG mean Earth radius (km)
ANALYSIS_DEFAULT_RATIO = 0.80    # ~80% of rows include analysis by default

# WGS‑84 constants (Vincenty)
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Coord:
    lat: float  # decimal degrees
    lon: float  # decimal degrees

# -----------------------------
# Utility helpers
# -----------------------------
def choose(seq):
    return random.choice(seq)

def weighted_choice(weight_map: Dict[str, float]) -> str:
    items = list(weight_map.items())
    total = sum(max(0.0, w) for _, w in items)
    if total <= 0:
        return items[0][0]
    r = random.uniform(0, total)
    upto = 0.0
    for k, w in items:
        w = max(0.0, w)
        if upto + w >= r:
            return k
        upto += w
    return items[-1][0]

def maybe(p: float) -> bool:
    return random.random() < max(0.0, min(1.0, p))

def random_coord() -> Coord:
    # keep away from exact poles/antimeridian to avoid degeneracy
    return Coord(lat=random.uniform(-89.5, 89.5), lon=random.uniform(-179.5, 179.5))

# -----------------------------
# PhraseBank: JSON-only loader
# -----------------------------
class PhraseBank:
    """
    Loads all text/variants strictly from JSON files.
    If a file is missing or keys are absent, a clear error is raised.
    """
    REQUIRED_FILES = [
        "user_patterns.json",
        "final_text.json",
        "analysis_text.json",
        "unit_synonyms.json",
        "unit_user_variants.json",
        "noise_config.json",
        "format_noise.json",
        "analysis_detail_config.json",
        "analysis_detail_blocks.json",
        "defaults_config.json",
        "intent_text.json",
        "gutcheck_explanations.json",
    ]

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or (Path(__file__).parent if "__file__" in globals() else Path("."))
        self._data = {}

        for fname in self.REQUIRED_FILES:
            self._data[fname] = self._load_required_json(fname)

        # Expose as attributes for convenience
        self.user_patterns      = self._data["user_patterns.json"]
        self.final_text         = self._data["final_text.json"]
        self.analysis_text      = self._data["analysis_text.json"]
        self.unit_synonyms      = self._data["unit_synonyms.json"]
        self.unit_user_variants = self._data["unit_user_variants.json"]
        self.noise              = self._data["noise_config.json"]
        self.format_noise       = self._data["format_noise.json"]
        self.detail_cfg         = self._data["analysis_detail_config.json"]
        self.detail_blocks      = self._data["analysis_detail_blocks.json"]
        self.defaults           = self._data["defaults_config.json"]
        self.intent_text        = self._data["intent_text.json"]
        self.gutcheck_explainers= self._data["gutcheck_explanations.json"]

        # Minimal key validation to fail early if a JSON is incomplete
        self._require_keys(self.analysis_text, [
            "openers_with_ballpark", "openers_no_ballpark", "intuitive_openers",
            "method_choice_transitions", "method_names", "haversine_overview",
            "vincenty_overview", "calc_intro_variants", "fallback_lines"
        ], "analysis_text.json")

        self._require_keys(self.final_text, [
            "intro_lines", "step1_headings", "convert_headings", "unit_heading",
            "answer_heading", "method_used_lines", "intuition_compare_lines"
        ], "final_text.json")

        self._require_keys(self.intent_text, [
            "openers", "explicit_templates", "ambiguous_templates",
            "default_templates", "coord_line_templates"
        ], "intent_text.json")

        self._require_keys(self.gutcheck_explainers, ["small", "large"], "gutcheck_explanations.json")

    def _load_required_json(self, filename: str):
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required JSON file: {filename}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
        return data

    @staticmethod
    def _require_keys(obj: Dict, keys: List[str], where: str):
        for k in keys:
            if k not in obj:
                raise KeyError(f"Key '{k}' missing in {where}")

# -----------------------------
# Coordinate formatting helpers
# -----------------------------
def _to_dms_components(value: float) -> Tuple[int, int, float, int]:
    sign = 1 if value >= 0 else -1
    v = abs(value)
    deg = int(v)
    minutes_full = (v - deg) * 60.0
    minute = int(minutes_full)
    sec = round((minutes_full - minute) * 60.0, 2)
    if sec >= 60.0:
        sec = 0.0
        minute += 1
    if minute >= 60:
        minute = 0
        deg += 1
    return deg, minute, sec, sign

def _to_dmm_components(value: float) -> Tuple[int, float, int]:
    sign = 1 if value >= 0 else -1
    v = abs(value)
    deg = int(v)
    minutes = round((v - deg) * 60.0, 3)
    if minutes >= 60.0:
        minutes = 0.0
        deg += 1
    return deg, minutes, sign

def format_dd(c: Coord) -> str:
    return f"{c.lat:.6f}, {c.lon:.6f}"

def format_dms(c: Coord) -> str:
    dlat, mlat, slat, slat_sign = _to_dms_components(c.lat)
    dlon, mlon, slon, slon_sign = _to_dms_components(c.lon)
    hemi_lat = 'N' if slat_sign >= 0 else 'S'
    hemi_lon = 'E' if slon_sign >= 0 else 'W'
    return f"{dlat}° {mlat}′ {slat:.2f}″ {hemi_lat}, {dlon}° {mlon}′ {slon:.2f}″ {hemi_lon}"

def format_dmm(c: Coord) -> str:
    dlat, mlat, slat_sign = _to_dmm_components(c.lat)
    dlon, mlon, slon_sign = _to_dmm_components(c.lon)
    hemi_lat = 'N' if slat_sign >= 0 else 'S'
    hemi_lon = 'E' if slon_sign >= 0 else 'W'
    return f"{dlat}° {mlat:.3f}′ {hemi_lat}, {dlon}° {mlon:.3f}′ {hemi_lon}"

# -----------------------------
# Distance formulas
# -----------------------------
def haversine_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> Tuple[float, Dict[str, float]]:
    lat1 = math.radians(lat1_deg); lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg); lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = (math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    d_km = EARTH_RADIUS_KM * c
    return d_km, {
        "R_km": EARTH_RADIUS_KM,
        "lat1_rad": lat1, "lon1_rad": lon1,
        "lat2_rad": lat2, "lon2_rad": lon2,
        "delta_lat": dlat, "delta_lon": dlon, "a": a, "c": c
    }

def vincenty_distance_wgs84_km(lat1_deg: float, lon1_deg: float,
                               lat2_deg: float, lon2_deg: float,
                               max_iter: int = 200, tol: float = 1e-12) -> Tuple[Optional[float], Dict, bool]:
    if (lat1_deg == lat2_deg) and (lon1_deg == lon2_deg):
        return 0.0, {"a": WGS84_A, "f": WGS84_F, "b": WGS84_B, "iterations": 0,
                     "converged": True, "sigma": 0.0, "s_m": 0.0}, True

    phi1 = math.radians(lat1_deg); phi2 = math.radians(lat2_deg)
    L = math.radians(lon2_deg - lon1_deg)
    U1 = math.atan((1.0 - WGS84_F) * math.tan(phi1))
    U2 = math.atan((1.0 - WGS84_F) * math.tan(phi2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lam = L
    prev = None
    sin_sigma = cos_sigma = sigma = sin_alpha = cos2_alpha = cos2_sigma_m = 0.0

    for iters in range(1, max_iter + 1):
        sinlam, coslam = math.sin(lam), math.cos(lam)
        sin_sigma = math.sqrt((cosU2 * sinlam) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * coslam) ** 2)
        if sin_sigma == 0.0:
            dbg = {"a": WGS84_A, "f": WGS84_F, "b": WGS84_B, "iterations": iters,
                   "converged": True, "sigma": 0.0, "s_m": 0.0, "U1": U1, "U2": U2, "L": L, "lambda_final": lam}
            return 0.0, dbg, True

        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * coslam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = (cosU1 * cosU2 * sinlam) / max(1e-16, sin_sigma)
        cos2_alpha = 1.0 - sin_alpha * sin_alpha
        cos2_sigma_m = cos_sigma - 2.0 * sinU1 * sinU2 / (cos2_alpha if cos2_alpha != 0 else 1e-16)

        C = (WGS84_F / 16.0) * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha))
        lam_next = L + (1.0 - C) * WGS84_F * sin_alpha * (
            sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1.0 + 2.0 * (cos2_sigma_m ** 2)))
        )
        if prev is not None and abs(lam_next - lam) < tol:
            lam = lam_next
            break
        prev = lam
        lam = lam_next
    else:
        return None, {"a": WGS84_A, "f": WGS84_F, "b": WGS84_B, "U1": U1, "U2": U2, "L": L,
                      "iterations": max_iter, "converged": False}, False

    u2 = cos2_alpha * ((WGS84_A*WGS84_A - WGS84_B*WGS84_B) / (WGS84_B*WGS84_B))
    A = 1.0 + (u2 / 16384.0) * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B = (u2 / 1024.0) * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
    Delta_sigma = B * sin_sigma * (
        cos2_sigma_m + (B / 4.0) * (cos_sigma * (-1.0 + 2.0 * (cos2_sigma_m ** 2)) -
        (B / 6.0) * cos2_sigma_m * (-3.0 + 4.0 * (sin_sigma ** 2)) * (-3.0 + 4.0 * (cos2_sigma_m ** 2)))
    )

    s_m = WGS84_B * A * (sigma - Delta_sigma)
    d_km = s_m / 1000.0

    debug = {
        "a": WGS84_A, "f": WGS84_F, "b": WGS84_B,
        "U1": U1, "U2": U2, "L": L, "lambda_final": lam,
        "iterations": iters, "converged": True,
        "sin_sigma": sin_sigma, "cos_sigma": cos_sigma, "sigma": sigma,
        "sin_alpha": sin_alpha, "cos2_alpha": cos2_alpha, "cos2_sigma_m": cos2_sigma_m,
        "C": C, "u2": u2, "A": A, "B": B, "Delta_sigma": Delta_sigma, "s_m": s_m
    }
    return d_km, debug, True

# -----------------------------
# Units
# -----------------------------
def convert_km(distance_km: float, unit: str) -> Tuple[float, str]:
    """Return (value, canonical_unit). Defaults to km when ambiguous."""
    u = (unit or "").lower()
    if u in ["km","kilometer","kilometers","kilometre","kilometres","klick","klicks","click","clicks","k","kms","km.","k."]:
        return distance_km, "km"
    if u in ["nautical miles","nautical mile","nmi","nm","n.m.","n mi","sea mile","sea miles"]:
        return distance_km / 1.852, "nmi"
    if u in ["miles","mi","mile","sm","m","m.","statute","statute miles","give it to me in m"]:
        return distance_km / 1.609344, "mi"
    return distance_km, "km"

# -----------------------------
# Noise / typos (system & user only)
# -----------------------------
def inject_typos(s: str, p: PhraseBank, is_system: bool = False) -> str:
    prob = p.noise.get("p_system_typos", 0.0) if is_system else p.noise.get("p_user_typos", 0.0)
    if not maybe(prob):
        return s
    kept, dropped = [], 0
    drop_budget = max(1, int(len(s) * 0.01))
    for ch in s:
        if ch.isdigit() or ch in "+-." or ch in "NSEW":
            kept.append(ch); continue
        if dropped < drop_budget and ch.isalpha() and maybe(0.25):
            dropped += 1; continue
        kept.append(ch)
    return "".join(kept)

def _apply_all_caps(s: str, p: PhraseBank, is_system: bool) -> str:
    caps_cfg = p.defaults.get("caps", {})
    prob = caps_cfg.get("p_all_caps_system", 0.0) if is_system else caps_cfg.get("p_all_caps_user", 0.0)
    if not maybe(prob):
        return s
    mode = weighted_choice(caps_cfg.get("p_all_caps_mode_weights", {"full":0.5,"partial":0.5}))
    if mode == "full":
        return s.upper()
    tokens = re.split(r"(\s+)", s)
    idxs = [i for i,t in enumerate(tokens) if t.strip() and t.isalpha()]
    random.shuffle(idxs)
    for i in idxs[:max(1, len(idxs)//6)]:
        tokens[i] = tokens[i].upper()
    return "".join(tokens)

# -----------------------------
# Coordinate string noise (user/system only)
# -----------------------------
def _maybe_replace_prime(p: PhraseBank, text: str) -> str:
    if maybe(p.noise.get("p_coord_replace_prime", 0.0)):
        text = text.replace("′", choose(["'", "", "’"]))
    if maybe(p.noise.get("p_coord_replace_dblprime", 0.0)):
        text = text.replace("″", choose(['"', "", "”"]))
    return text

def _maybe_drop_deg(p: PhraseBank, text: str) -> str:
    if maybe(p.noise.get("p_coord_drop_deg", 0.0)):
        text = text.replace("°", choose(["", " º", " deg"]))
    return text

def _random_space_variation(p: PhraseBank, text: str) -> str:
    if maybe(p.noise.get("p_coord_extra_spaces", 0.0)):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*", choose([",", ", ", " ,", "  ,  ", " "]), text)
        text = text.replace("° ", choose(["° ", " °", "  ° ", "°"]))
        text = text.replace(" ′", choose([" ′", "′", "  ′"]))
        text = text.replace(" ″", choose([" ″", "″", "  ″"]))
    return text

def _maybe_remove_comma(p: PhraseBank, text: str) -> str:
    if maybe(p.noise.get("p_coord_remove_comma", 0.0)):
        text = text.replace(",", choose(["", " ", "  "]))
    return text

def _maybe_drop_hemi(p: PhraseBank, text: str, target_hemi: List[str]) -> Tuple[str, Dict[str, bool]]:
    meta = {"hemi_dropped": False}
    for H in target_hemi:
        if H in text and maybe(p.noise.get("p_coord_drop_hemi", 0.0)):
            text = text.replace(H, "", 1)
            meta["hemi_dropped"] = True
    return text, meta

def render_noisy_coord_string(p: PhraseBank, fmt: str, c: Coord, which: str) -> Tuple[str, Dict[str, bool]]:
    if fmt == "DD":
        s = format_dd(c)
        s = _random_space_variation(p, s)
        s = _maybe_remove_comma(p, s)
        return s, {"hemi_dropped": False}
    elif fmt == "DMS":
        s = format_dms(c)
        s = _maybe_replace_prime(p, s); s = _maybe_drop_deg(p, s)
        s = _random_space_variation(p, s); s = _maybe_remove_comma(p, s)
        s, meta = _maybe_drop_hemi(p, s, ["N","S","E","W"])
        return s, meta
    else:
        s = format_dmm(c)
        s = _maybe_replace_prime(p, s); s = _maybe_drop_deg(p, s)
        s = _random_space_variation(p, s); s = _maybe_remove_comma(p, s)
        s, meta = _maybe_drop_hemi(p, s, ["N","S","E","W"])
        return s, meta

# -----------------------------
# Formatting variation (analysis/final only)
# -----------------------------
def apply_format_variation(text: str, cfg: Dict) -> str:
    lines = text.splitlines()
    if maybe(cfg.get("p_compact_bullets", 0.0)):
        bullet_pat = re.compile(r"^[\s]*([-*•—]\s+)(.*)$")
        buf, out = [], []
        for ln in lines:
            m = bullet_pat.match(ln)
            if m:
                buf.append(m.group(2).strip())
            else:
                if buf:
                    out.append(" • ".join(buf))
                    buf = []
                out.append(ln)
        if buf: out.append(" • ".join(buf))
        lines = out
    text2 = "\n".join(lines)
    roll = random.random()
    if roll < cfg.get("p_collapse_all", 0.0):
        return re.sub(r"\s*\n\s*", " ", text2).strip()
    if roll < cfg.get("p_collapse_all", 0.0) + cfg.get("p_collapse_some", 0.0):
        split = text2.split("\n")
        out = []
        frac_lo, frac_hi = cfg.get("collapse_some_fraction_range", [0.2, 0.5])
        target_frac = random.uniform(frac_lo, frac_hi)
        for i, ln in enumerate(split):
            out.append(ln)
            if i < len(split) - 1:
                out.append(" " if random.random() < target_frac else "\n")
        return re.sub(r"[ ]{2,}", " ", "".join(out)).strip()
    return text2

# -----------------------------
# Intent recap helpers
# -----------------------------
def _unit_long_label(phrases: PhraseBank, unit_short: str) -> str:
    unit_key = "km" if unit_short == "km" else ("miles" if unit_short == "mi" else "nautical miles")
    opts = phrases.unit_synonyms.get(unit_key, [unit_short])
    return choose(opts) if isinstance(opts, list) and opts else unit_short

def _make_intent_recap(phrases: PhraseBank, unit_short: str, unit_origin: str,
                       raw_unit_hint: Optional[str],
                       A_dd: Tuple[float,float], B_dd: Tuple[float,float]) -> str:
    t = phrases.intent_text
    opener = choose(t["openers"])
    unit_long = _unit_long_label(phrases, unit_short)
    if unit_origin == "explicit":
        line = choose(t["explicit_templates"]).format(op=opener, unit_long=unit_long)
    elif unit_origin == "ambiguous":
        line = choose(t["ambiguous_templates"]).format(op=opener, raw_unit=(raw_unit_hint or "").strip(), unit_long=unit_long)
    else:
        line = choose(t["default_templates"]).format(op=opener, unit_long=unit_long)
    coord_line = choose(t["coord_line_templates"]).format(alat=A_dd[0], alon=A_dd[1], blat=B_dd[0], blon=B_dd[1])
    return f"{line}\n- {coord_line}"

# -----------------------------
# Intuition helpers (hybrid gut-check)
# -----------------------------
def _wrap_delta_lon_deg(lon1_deg: float, lon2_deg: float) -> float:
    return (lon2_deg - lon1_deg + 540.0) % 360.0 - 180.0

def deg_len_lat_km(phi_deg: float) -> float:
    x = math.radians(phi_deg)
    meters = (111132.92
              - 559.82 * math.cos(2 * x)
              +   1.175 * math.cos(4 * x)
              -   0.0023 * math.cos(6 * x))
    return meters / 1000.0

def deg_len_lon_km(phi_deg: float) -> float:
    x = math.radians(phi_deg)
    meters = (111412.84 * math.cos(x)
              -    93.5 * math.cos(3 * x)
              +    0.118 * math.cos(5 * x))
    return meters / 1000.0

def central_angle_sloc(lat1_deg: float, lon1_deg: float,
                       lat2_deg: float, lon2_deg: float) -> float:
    phi1 = math.radians(lat1_deg); phi2 = math.radians(lat2_deg)
    dlam = math.radians(_wrap_delta_lon_deg(lon1_deg, lon2_deg))
    cos_c = math.sin(phi1)*math.sin(phi2) + math.cos(phi1)*math.cos(phi2)*math.cos(dlam)
    cos_c = min(1.0, max(-1.0, cos_c))
    return math.acos(cos_c)

def coarse_round_km(val_km: float) -> Tuple[int, int]:
    """
    Return (ballpark_km, step) using scale-aware rounding plus light jitter.
    This keeps the intuition deliberately approximate and avoids 0.0% gaps.
    """
    size = abs(val_km)
    if size < 100:
        step = 10
    elif size < 1000:
        step = 50
    elif size < 5000:
        step = 100
    elif size < 10000:
        step = 200
    else:
        step = 250  # coarser for very long arcs

    # Add small random jitter before rounding so we rarely land right on the exact.
    jitter = random.uniform(-0.35 * step, 0.35 * step)
    rough = val_km + jitter
    rounded = int(round(rough / step) * step)
    if rounded <= 0:
        rounded = step
    return rounded, step

def _format_why_template(tmpl: str, ctx: Dict[str, float]) -> str:
    class _SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return tmpl.format_map(_SafeDict(ctx))

def _why_lines_from_bank(p: PhraseBank, bank_key: str, ctx: Dict[str, float], k_range=(1,2)) -> List[str]:
    pool = (p.gutcheck_explainers.get(bank_key) or [])
    if not pool:
        return []
    random.shuffle(pool)
    k = random.randint(k_range[0], k_range[1])
    return [_format_why_template(line, ctx) for line in pool[:k]]

def _gutcheck_two_lines(A_lat: float, A_lon: float, B_lat: float, B_lon: float,
                        unit_short: str, phrases: PhraseBank) -> Tuple[List[str], float]:
    dlat_deg = abs(B_lat - A_lat)
    dlon_deg = abs(_wrap_delta_lon_deg(A_lon, B_lon))
    mid_lat  = 0.5 * (A_lat + B_lat)

    smallish = (dlat_deg <= 30.0 and dlon_deg <= 30.0 and abs(mid_lat) <= 60.0)

    if smallish:
        # Local-grid Pythagoras (good for short spans)
        km_per_deg_lat = deg_len_lat_km(mid_lat)
        km_per_deg_lon = deg_len_lon_km(mid_lat)
        ns_km  = km_per_deg_lat * dlat_deg
        ew_km  = km_per_deg_lon * dlon_deg
        approx_km = (ns_km**2 + ew_km**2) ** 0.5

        ball_km, _step = coarse_round_km(approx_km)
        disp_val, disp_unit = convert_km(ball_km, unit_short)

        line1 = (f"- Quick gut‑check: lon diff ~ {dlon_deg:.1f}°, lat diff ~ {dlat_deg:.1f}°; "
                 f"per‑degree @ ~{mid_lat:.0f}° → lat ≈ {km_per_deg_lat:.1f} km/°, lon ≈ {km_per_deg_lon:.1f} km/°; "
                 f"EW ~ {round(ew_km)} km; NS ~ {round(ns_km)} km.")
        line2 = f"- Ballpark ≈ {ball_km} km (≈ {round(disp_val)} {disp_unit})."

        ctx = {"mid_lat": mid_lat, "ball_km": ball_km, "dlat_deg": dlat_deg, "dlon_deg": dlon_deg}
        why_lines = _why_lines_from_bank(phrases, "small", ctx, k_range=(1,2))
        return [line1, line2] + why_lines, float(ball_km)

    # Long/high‑lat regime → spherical law‑of‑cosines central angle
    c = central_angle_sloc(A_lat, A_lon, B_lat, B_lon)
    approx_km = EARTH_RADIUS_KM * c
    ball_km, _step = coarse_round_km(approx_km)
    disp_val, disp_unit = convert_km(ball_km, unit_short)
    angle_deg = math.degrees(c)
    circumference_km = 2.0 * math.pi * EARTH_RADIUS_KM
    frac_pct = (angle_deg / 360.0) * 100.0
    per10deg_km = circumference_km / 36.0
    half_circum_km = circumference_km / 2.0
    ns_only_km = deg_len_lat_km(mid_lat) * dlat_deg
    angle_times_R_km = EARTH_RADIUS_KM * c
    circle_fraction_km = (angle_deg / 360.0) * circumference_km

    line1 = (f"- Quick gut‑check (long range/high lat): use spherical law‑of‑cosines; "
             f"longitude difference ~ {dlon_deg:.1f}°, central angle ≈ {c:.2f} rad (~{angle_deg:.0f}°).")
    line2 = f"- Ballpark ≈ {ball_km} km (≈ {round(disp_val)} {disp_unit})."

    ctx = {
        "EARTH_RADIUS_KM": EARTH_RADIUS_KM, "c": c, "angle_deg": angle_deg, "ball_km": ball_km,
        "circumference_km": circumference_km, "frac_pct": frac_pct, "per10deg_km": per10deg_km,
        "half_circum_km": half_circum_km, "dlat_deg": dlat_deg, "ns_only_km": ns_only_km,
        "circle_fraction_km": circle_fraction_km, "angle_times_R_km": angle_times_R_km
    }
    why_lines = _why_lines_from_bank(phrases, "large", ctx, k_range=(2,2))
    return [line1, line2] + why_lines, float(ball_km)

# -----------------------------
# Narrative number helpers
# -----------------------------
def narrate_haversine_numbers(dbg: dict, d_km: float, unit_short: str, for_analysis: bool) -> List[str]:
    """
    Robust narrator:
    - If full debug keys are present, print full numeric trail.
    - If not, fall back to a compact narration (prevents KeyError on minimalist dbg).
    """
    out_val, unit = convert_km(d_km, unit_short)
    # Fallback path if some keys aren't available
    required = {"lat1_rad","lon1_rad","lat2_rad","lon2_rad","delta_lat","delta_lon","c","R_km"}
    if not required.issubset(dbg.keys()):
        R = float(dbg.get("R_km", EARTH_RADIUS_KM))
        c = float(dbg.get("c", d_km / R if R else 0.0))
        if for_analysis:
            return [
                f"Using a spherical model with mean radius ≈ {R:.1f} km.",
                f"The combined angle is about {c:.4f} radians.",
                f"Multiply by R → distance ≈ {d_km:.1f} km (≈ {out_val:.1f} {unit})."
            ]
        else:
            lines = [f"Constants: mean Earth radius R = {R:.4f} km."]
            lines.append(f"Central angle ≈ {c:.6f} rad.")
            lines.append(f"Distance: d_km = R × angle = {d_km:.3f} km → {out_val:.3f} {unit}.")
            return lines

    if for_analysis:
        return [
            f"Using a spherical model with mean radius ≈ {dbg['R_km']:.1f} km.",
            f"The combined angle from the deltas is about {dbg['c']:.4f} radians.",
            f"Multiply by R → distance ≈ {d_km:.1f} km (≈ {out_val:.1f} {unit})."
        ]
    else:
        return [
            f"Constants: mean Earth radius R = {dbg['R_km']:.4f} km.",
            f"Radians (internal): lat1={dbg['lat1_rad']:.6f}, lon1={dbg['lon1_rad']:.6f}; lat2={dbg['lat2_rad']:.6f}, lon2={dbg['lon2_rad']:.6f}.",
            f"Deltas: delta_lat={dbg['delta_lat']:.6f}, delta_lon={dbg['delta_lon']:.6f}.",
            f"Central angle ≈ {dbg['c']:.6f} rad.",
            f"Distance: d_km = R × angle = {d_km:.3f} km → {out_val:.3f} {unit}."
        ]

def narrate_vincenty_numbers(dbg: dict, d_km: float, unit_short: str, for_analysis: bool) -> List[str]:
    out_val, unit = convert_km(d_km, unit_short)
    if for_analysis:
        return [
            "Ellipsoidal model (WGS‑84): reduce latitudes for flattening, iterate the longitude until stable,",
            f"apply the small flattening correction, and scale; distance ≈ {d_km:.1f} km (≈ {out_val:.1f} {unit})."
        ]
    else:
        return [
            f"Constants (WGS‑84): a={dbg['a']:.1f} m; f={WGS84_F:.12f}; b={dbg['b']:.1f} m.",
            f"Converged in {dbg.get('iterations',0)} iteration(s); σ={dbg.get('sigma',0.0):.6f} rad; Δσ={dbg.get('Delta_sigma',0.0):.6f}.",
            f"Distance: s = b·A·(σ − Δσ) = {d_km:.3f} km → {out_val:.3f} {unit}."
        ]

# -----------------------------
# Sloppy/realistic user prompt
# -----------------------------
def choose_user_unit_variant(p: PhraseBank, canonical: str) -> str:
    key = {"km":"km", "mi":"mi", "nmi":"nmi"}[canonical]
    variants = p.unit_user_variants.get(key, [])
    return choose(variants) if variants else canonical

def normalize_user_unit_guess(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    s_comp = re.sub(r"[\s\.\-_/]+", "", s)
    # nautical
    if ("seamile" in s_comp) or ("sea mile" in s):
        return "nmi"
    if ("naut" in s) or ("nmi" in s_comp) or s_comp == "nm":
        return "nmi"
    # statute / miles
    if ("statute" in s) or ("statutemile" in s_comp) or ("statutemiles" in s_comp):
        return "mi"
    if ("mile" in s) or re.search(r"\bmi\b", s) or s_comp in ("m","sm"):
        return "mi"
    # kilometers
    if ("km" in s_comp) or ("klick" in s) or ("click" in s) or s_comp == "k" or ("meter" in s) or ("metre" in s):
        return "km"
    return None

def generate_sloppy_user(p: PhraseBank, fmt: str, noisy_A: str, noisy_B: str) -> Tuple[str, Optional[str], str]:
    modes = ["normal", "short_q", "coords_only", "fragment", "arrow", "caps_shout"]
    mode = random.choice(modes)
    raw_unit_hint: Optional[str] = None

    if mode == "normal":
        canonical = weighted_choice({"km": 0.40, "mi": 0.45, "nmi": 0.15})
        raw_unit_hint = choose_user_unit_variant(p, canonical)
        template = choose(p.user_patterns)
        text = template.format(A=noisy_A, B=noisy_B, unit=raw_unit_hint)
    elif mode == "short_q":
        if maybe(0.55):
            canonical = weighted_choice({"km": 0.40, "mi": 0.50, "nmi": 0.10})
            raw_unit_hint = choose_user_unit_variant(p, canonical)
            text = f"how far from {noisy_A} to {noisy_B} in {raw_unit_hint}?"
        else:
            text = f"how far from {noisy_A} to {noisy_B}?"
    elif mode == "coords_only":
        sep = choose([" ", "  ", "\n", "  →  "])
        text = f"{noisy_A}{sep}{noisy_B}"
    elif mode == "fragment":
        canonical = weighted_choice({"km": 0.35, "mi": 0.50, "nmi": 0.15})
        raw_unit_hint = choose_user_unit_variant(p, canonical)
        text = f"distance in {raw_unit_hint}: {noisy_A} to {noisy_B}"
    elif mode == "arrow":
        canonical = weighted_choice({"km": 0.35, "mi": 0.50, "nmi": 0.15})
        raw_unit_hint = choose_user_unit_variant(p, canonical)
        text = f"{noisy_A} -> {noisy_B} in {raw_unit_hint}?"
    else:  # caps_shout
        canonical = weighted_choice({"km": 0.30, "mi": 0.60, "nmi": 0.10})
        raw_unit_hint = choose_user_unit_variant(p, canonical)
        text = f"DISTANCE??? {noisy_A} {noisy_B} ({raw_unit_hint})"

    text = inject_typos(text, p, is_system=False)
    text = _apply_all_caps(text, p, is_system=False)
    return text, raw_unit_hint, mode

# -----------------------------
# Assistant (analysis)
# -----------------------------
def render_assistant_analysis(
    phrases: PhraseBank, fmt: str, A: Coord, B: Coord, unit_canonical: str,
    noise_meta: Dict[str, Dict[str, bool]], intent_info: Dict[str, Optional[str]]
) -> Tuple[str, Dict]:
    # Normalize to decimal degrees
    if fmt == "DD":
        A_lat_dd, A_lon_dd, B_lat_dd, B_lon_dd = A.lat, A.lon, B.lat, B.lon
    elif fmt == "DMS":
        Adlat, Amlat, Aslat, sA = _to_dms_components(A.lat)
        Adlon, Amlon, Aslon, sAo = _to_dms_components(A.lon)
        Bdlat, Bmlat, Bslat, sB = _to_dms_components(B.lat)
        Bdlon, Bmlon, Bslon, sBo = _to_dms_components(B.lon)
        hemi_Alat = 'N' if sA >= 0 else 'S'
        hemi_Alon = 'E' if sAo >= 0 else 'W'
        hemi_Blat = 'N' if sB >= 0 else 'S'
        hemi_Blon = 'E' if sBo >= 0 else 'W'
        A_lat_dd = (Adlat + Amlat/60 + Aslat/3600) * (1 if hemi_Alat=='N' else -1)
        A_lon_dd = (Adlon + Amlon/60 + Aslon/3600) * (1 if hemi_Alon=='E' else -1)
        B_lat_dd = (Bdlat + Bmlat/60 + Bslat/3600) * (1 if hemi_Blat=='N' else -1)
        B_lon_dd = (Bdlon + Bmlon/60 + Bslon/3600) * (1 if hemi_Blon=='E' else -1)
    else:
        Adlat, Amlat, sA = _to_dmm_components(A.lat)
        Adlon, Amlon, sAo = _to_dmm_components(A.lon)
        Bdlat, Bmlat, sB = _to_dmm_components(B.lat)
        Bdlon, Bmlon, sBo = _to_dmm_components(B.lon)
        hemi_Alat = 'N' if sA >= 0 else 'S'
        hemi_Alon = 'E' if sAo >= 0 else 'W'
        hemi_Blat = 'N' if sB >= 0 else 'S'
        hemi_Blon = 'E' if sBo >= 0 else 'W'
        A_lat_dd = (Adlat + Amlat/60) * (1 if hemi_Alat=='N' else -1)
        A_lon_dd = (Adlon + Amlon/60) * (1 if hemi_Alon=='E' else -1)
        B_lat_dd = (Bdlat + Bmlat/60) * (1 if hemi_Blat=='N' else -1)
        B_lon_dd = (Bdlon + Bmlon/60) * (1 if hemi_Blon=='E' else -1)

    unit_short_disp = convert_km(0.0, unit_canonical)[1]
    lines = [choose(phrases.analysis_text["openers_with_ballpark"])]
    lines.append(_make_intent_recap(
        phrases, unit_short_disp, intent_info.get("unit_origin","default"),
        intent_info.get("raw_unit_hint"), (A_lat_dd, A_lon_dd), (B_lat_dd, B_lon_dd)
    ))

    if noise_meta.get("A", {}).get("hemi_dropped") or noise_meta.get("B", {}).get("hemi_dropped"):
        notes = []
        for label in ("A","B"):
            if noise_meta.get(label, {}).get("hemi_dropped"):
                notes.append(f"{label}: hemisphere letter missing → inferred from sign.")
        if notes:
            lines.append("- " + " ".join(notes))

    lines.append(choose(phrases.analysis_text["intuitive_openers"]))
    gut_lines, approx_km = _gutcheck_two_lines(A_lat_dd, A_lon_dd, B_lat_dd, B_lon_dd, unit_short_disp, phrases)
    lines.extend(gut_lines)

    # Method selection 50/50
    method = "haversine" if random.random() < 0.5 else "vincenty"
    method_label = choose(phrases.analysis_text["method_names"][method])
    lines.append(choose(phrases.analysis_text["method_choice_transitions"]) + f" {method_label}")

    if method == "haversine":
        lines.append("- " + choose(phrases.analysis_text["haversine_overview"]))
        d_km, dbg = haversine_km(A_lat_dd, A_lon_dd, B_lat_dd, B_lon_dd)
        lines.append(choose(phrases.analysis_text["calc_intro_variants"]).format(method_label=method_label))
        lines.extend(narrate_haversine_numbers(dbg, d_km, unit_short_disp, for_analysis=True))
        exact_dbg = {"method": "haversine", "dbg": dbg}
    else:
        lines.append("- " + choose(phrases.analysis_text["vincenty_overview"]))
        d_km, vdbg, ok = vincenty_distance_wgs84_km(A_lat_dd, A_lon_dd, B_lat_dd, B_lon_dd)
        lines.append(choose(phrases.analysis_text["calc_intro_variants"]).format(method_label=method_label))
        if ok and d_km is not None:
            lines.extend(narrate_vincenty_numbers(vdbg, d_km, unit_short_disp, for_analysis=True))
            exact_dbg = {"method": "vincenty", "dbg": vdbg}
        else:
            lines.append("- " + choose(phrases.analysis_text["fallback_lines"]))
            d_km, hdbg = haversine_km(A_lat_dd, A_lon_dd, B_lat_dd, B_lon_dd)
            lines.extend(narrate_haversine_numbers(hdbg, d_km, unit_short_disp, for_analysis=True))
            method = "haversine"
            method_label = choose(phrases.analysis_text["method_names"]["haversine"])
            exact_dbg = {"method": "haversine", "dbg": hdbg}

    aux = {
        "A_dd": (A_lat_dd, A_lon_dd), "B_dd": (B_lat_dd, B_lon_dd),
        "approx_km": float(approx_km), "exact_km": float(d_km),
        "method": method, "method_label": method_label,
        "exact_dbg": exact_dbg, "unit_short": unit_short_disp
    }

    analysis_text = "\n".join(lines)
    return apply_format_variation(analysis_text, phrases.format_noise.get("analysis", {})), aux

# -----------------------------
# Assistant (final)
# -----------------------------
def render_assistant_final(
    phrases: PhraseBank, fmt: str, A: Coord, B: Coord, unit_canonical: str, aux: Dict
) -> str:
    fx = phrases.final_text
    lines: List[str] = []
    if fx.get("intro_lines"): lines.append(choose(fx["intro_lines"])); lines.append("")

    step1 = choose(fx["step1_headings"])
    convh = choose(fx["convert_headings"])
    lines.append(step1)

    # Re-derive decimal degrees for explicit narration
    if fmt == "DD":
        lines.append(f"The points are in decimal degrees: A = {A.lat:.6f}°, {A.lon:.6f}°; B = {B.lat:.6f}°, {B.lon:.6f}°.")
        A_lat_dd, A_lon_dd, B_lat_dd, B_lon_dd = A.lat, A.lon, B.lat, B.lon
    elif fmt == "DMS":
        Adlat, Amlat, Aslat, sA = _to_dms_components(A.lat)
        Adlon, Amlon, Aslon, sAo = _to_dms_components(A.lon)
        Bdlat, Bmlat, Bslat, sB = _to_dms_components(B.lat)
        Bdlon, Bmlon, Bslon, sBo = _to_dms_components(B.lon)
        hemi_Alat = 'N' if sA >= 0 else 'S'
        hemi_Alon = 'E' if sAo >= 0 else 'W'
        hemi_Blat = 'N' if sB >= 0 else 'S'
        hemi_Blon = 'E' if sBo >= 0 else 'W'
        lines.append(f"A = {Adlat}° {Amlat}′ {Aslat:.2f}″ {hemi_Alat}, {Adlon}° {Amlon}′ {Aslon:.2f}″ {hemi_Alon}.")
        lines.append(f"B = {Bdlat}° {Bmlat}′ {Bslat:.2f}″ {hemi_Blat}, {Bdlon}° {Bmlon}′ {Bslon:.2f}″ {hemi_Blon}.")
        lines.append(""); lines.append(convh)
        A_lat_dd = (Adlat + Amlat/60 + Aslat/3600) * (1 if hemi_Alat=='N' else -1)
        A_lon_dd = (Adlon + Amlon/60 + Aslon/3600) * (1 if hemi_Alon=='E' else -1)
        B_lat_dd = (Bdlat + Bmlat/60 + Bslat/3600) * (1 if hemi_Blat=='N' else -1)
        B_lon_dd = (Bdlon + Bmlon/60 + Bslon/3600) * (1 if hemi_Blon=='E' else -1)
        lines.append("decimal° = deg + min/60 + sec/3600 (minus for S/W).")
        lines.append(f"A → lat = {A_lat_dd:.6f}°, lon = {A_lon_dd:.6f}°; B → lat = {B_lat_dd:.6f}°, lon = {B_lon_dd:.6f}°.")
    else:
        Adlat, Amlat, sA = _to_dmm_components(A.lat)
        Adlon, Amlon, sAo = _to_dmm_components(A.lon)
        Bdlat, Bmlat, sB = _to_dmm_components(B.lat)
        Bdlon, Bmlon, sBo = _to_dmm_components(B.lon)
        hemi_Alat = 'N' if sA >= 0 else 'S'
        hemi_Alon = 'E' if sAo >= 0 else 'W'
        hemi_Blat = 'N' if sB >= 0 else 'S'
        hemi_Blon = 'E' if sBo >= 0 else 'W'
        lines.append(f"A = {Adlat}° {Amlat:.3f}′ {hemi_Alat}, {Adlon}° {Amlon:.3f}′ {hemi_Alon}.")
        lines.append(f"B = {Bdlat}° {Bmlat:.3f}′ {hemi_Blat}, {Bdlon}° {Bmlon:.3f}′ {hemi_Blon}.")
        lines.append(""); lines.append(convh)
        A_lat_dd = (Adlat + Amlat/60) * (1 if hemi_Alat=='N' else -1)
        A_lon_dd = (Adlon + Amlon/60) * (1 if hemi_Alon=='E' else -1)
        B_lat_dd = (Bdlat + Bmlat/60) * (1 if hemi_Blat=='N' else -1)
        B_lon_dd = (Bdlon + Bmlon/60) * (1 if hemi_Blon=='E' else -1)
        lines.append("decimal° = deg + minutes/60 (minus for S/W).")
        lines.append(f"A → lat = {A_lat_dd:.6f}°, lon = {A_lon_dd:.6f}°; B → lat = {B_lat_dd:.6f}°, lon = {B_lon_dd:.6f}°.")

    method_label = aux.get("method_label", "Haversine")
    lines.append("")
    lines.append(choose(fx["method_used_lines"]).format(method_label=method_label))

    exact_km = float(aux.get("exact_km", 0.0))
    method = aux.get("method", "haversine")
    dbg = aux.get("exact_dbg", {"method": method, "dbg": {}})["dbg"]

    if method == "haversine":
        lines.extend(narrate_haversine_numbers(dbg, exact_km, aux["unit_short"], for_analysis=False))
    else:
        lines.extend(narrate_vincenty_numbers(dbg, exact_km, aux["unit_short"], for_analysis=False))

    lines.append(""); lines.append(choose(fx["unit_heading"]))
    out_val, unit_short = convert_km(exact_km, aux["unit_short"])
    unit_long = _unit_long_label(phrases, unit_short)
    conv_expl = "(already in kilometers)" if unit_short=="km" else ("mi = km ÷ 1.609344" if unit_short=="mi" else "nmi = km ÷ 1.852")
    lines.append(f"Convert to {unit_long}: {conv_expl} → {out_val:.3f} {unit_short}.")

    approx_km = aux.get("approx_km", None)
    if approx_km is not None and exact_km > 0:
        diff_km = abs(exact_km - approx_km)
        diff_pct = 100.0 * diff_km / exact_km
        comp = choose(fx["intuition_compare_lines"])
        lines.append(comp.format(approx_km=approx_km, exact_km=exact_km, diff_km=diff_km, diff_pct=diff_pct))

    lines.append(""); lines.append(choose(fx["answer_heading"]))
    lines.append(f"{out_val:.3f} {unit_short}")
    return apply_format_variation("\n".join(lines), phrases.format_noise.get("final", {}))

# -----------------------------
# Row assembly (Harmony-like)
# -----------------------------
def to_harmony_row(messages: List[Dict]) -> Dict:
    system_msg = next(m for m in messages if m.get("role") == "system")
    user_msg   = next(m for m in messages if m.get("role") == "user")
    analysis_msg = next((m for m in messages if m.get("role") == "assistant" and m.get("channel") == "analysis"), None)
    final_msg    = next(m for m in messages if m.get("role") == "assistant" and m.get("channel") == "final")

    developer = system_msg.get("content", "")
    user      = user_msg.get("content", "")
    analysis  = analysis_msg.get("content") if analysis_msg else None
    final     = final_msg.get("content", "")

    assistant_struct = {"role": "assistant", "content": final, "thinking": analysis}
    row_messages = [
        {"role": "system", "content": developer, "thinking": None},
        {"role": "user",   "content": user,      "thinking": None},
        assistant_struct,
    ]
    return {"developer": developer, "user": user, "analysis": analysis, "final": final, "messages": row_messages}

# -----------------------------
# Parquet I/O
# -----------------------------
def build_parquet(rows: List[Dict], out_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    msg_struct = pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
        pa.field("thinking", pa.string(), nullable=True),
    ])
    schema = pa.schema([
        pa.field("developer", pa.string()),
        pa.field("user", pa.string()),
        pa.field("analysis", pa.string(), nullable=True),
        pa.field("final", pa.string()),
        pa.field("messages", pa.list_(msg_struct)),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path, compression="zstd")

def preview_parquet(out_path: Path, max_rows: int = 3):
    import pyarrow.parquet as pq
    import pandas as pd
    import json as _json

    table = pq.read_table(out_path)
    df = table.to_pandas()
    n = len(df)
    rows_to_show = min(max_rows, n)
    print("\n=== Parquet written ===")
    print(f"File: {out_path.resolve()}")
    print(f"Rows: {n}")

    for i in range(rows_to_show):
        print(f"\n--- Example {i+1}/{n} ---")
        print("system:\n" + df.at[i, "developer"])
        print("\nuser:\n" + df.at[i, "user"])
        print("\nfinal:\n" + df.at[i, "final"])

    messages0 = pq.read_table(out_path, columns=["messages"]).column(0).to_pylist()[0]
    print("\n--- messages[0] (pretty) ---")
    print(_json.dumps(messages0, ensure_ascii=False, indent=2))

# -----------------------------
# Example generation
# -----------------------------
def generate_one_example(
    seed: Optional[int] = None,
    analysis_ratio: float = ANALYSIS_DEFAULT_RATIO,
    format_style: str = "harmony",
) -> List[Dict]:
    if seed is not None:
        random.seed(seed)
    phrases = PhraseBank()

    # Random coordinate format
    fmt = random.choice(["DD", "DMS", "DMM"])
    A = random_coord(); B = random_coord()
    noisy_A, meta_A = render_noisy_coord_string(phrases, fmt, A, "A")
    noisy_B, meta_B = render_noisy_coord_string(phrases, fmt, B, "B")
    noisy_meta = {"A": meta_A, "B": meta_B}

    # Build user text
    user_text_raw, raw_unit_hint, _mode = generate_sloppy_user(phrases, fmt, noisy_A, noisy_B)
    unit_from_hint = normalize_user_unit_guess(raw_unit_hint) if raw_unit_hint else None
    if unit_from_hint:
        canonical_unit = unit_from_hint
        unit_origin = "explicit" if raw_unit_hint and raw_unit_hint.strip().lower() not in ["m","m."] else "ambiguous"
    else:
        canonical_unit = weighted_choice(phrases.defaults.get("unit_default_weights", {"km": 1.0}))
        unit_origin = "default"

    # System + user
    system_text = "you are an agent that understands GPS and can calculate distances"
    system_text = inject_typos(system_text, phrases, is_system=True)
    system_text = _apply_all_caps(system_text, phrases, is_system=True)
    system_msg = {"role": "system", "content": system_text}
    user_msg   = {"role": "user",   "content": user_text_raw}

    msgs: List[Dict] = [system_msg, user_msg]

    # Analysis (probabilistic)
    if maybe(analysis_ratio):
        analysis_text, aux = render_assistant_analysis(
            phrases, fmt, A, B, canonical_unit, noisy_meta,
            intent_info={"unit_origin": unit_origin, "raw_unit_hint": raw_unit_hint}
        )
        msgs.append({"role": "assistant", "channel": "analysis", "content": analysis_text})
        final_text = render_assistant_final(phrases, fmt, A, B, canonical_unit, aux)
        msgs.append({"role": "assistant", "channel": "final", "content": final_text})
    else:
        # If no analysis, still compute exact numbers & provide FULL debug to avoid KeyError
        d_km, hdbg = haversine_km(A.lat, A.lon, B.lat, B.lon)
        aux = {
            "A_dd": (A.lat, A.lon), "B_dd": (B.lat, B.lon),
            "approx_km": None,              # no intuition compare when analysis is omitted
            "exact_km": d_km,
            "method": "haversine",
            "method_label": choose(phrases.analysis_text["method_names"]["haversine"]),
            "exact_dbg": {"method": "haversine", "dbg": hdbg},  # full dbg (prevents KeyError)
            "unit_short": convert_km(0.0, canonical_unit)[1]
        }
        final_text = render_assistant_final(phrases, fmt, A, B, canonical_unit, aux)
        msgs.append({"role": "assistant", "channel": "final", "content": final_text})

    return msgs

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate GPS reasoning examples (JSON-driven).")
    parser.add_argument("--n", type=int, default=10, help="Number of examples")
    parser.add_argument("--out", type=str, default="gps_samples.parquet", help="Output Parquet path")
    parser.add_argument("--no-preview", action="store_true", help="Skip parquet preview printout")
    parser.add_argument("--json-only", action="store_true", help="Print a single JSON example and exit")
    parser.add_argument("--analysis-ratio", type=float, default=ANALYSIS_DEFAULT_RATIO, help="Probability of including analysis [0..1]")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.json_only:
        msgs = generate_one_example(analysis_ratio=args.analysis_ratio)
        print(json.dumps(msgs, ensure_ascii=False, indent=2))
        return

    rows = []
    for _ in range(args.n):
        msgs = generate_one_example(analysis_ratio=args.analysis_ratio)
        rows.append(to_harmony_row(msgs))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_parquet(rows, out_path)
    if not args.no_preview:
        preview_parquet(out_path)

if __name__ == "__main__":
    main()

