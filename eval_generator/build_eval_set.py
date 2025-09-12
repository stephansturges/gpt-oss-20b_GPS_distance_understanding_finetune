#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_eval_set.py
-----------------
Generate an evaluation dataset for GPS distance reasoning:

- Prompts: stable 'developer' (system) + noisy/sloppy 'user'
- Ground truth: 'expected_km' (Haversine, mean Earth radius), plus
  'expected_value' converted into the user's implied/explicit unit
  (default to miles if ambiguous).
- External JSONs in --phrases-dir control user text randomness and noise.

Usage:
  python3 build_eval_set.py --n 1000 --out eval_set.parquet --phrases-dir ./eval_phrases --seed 42

Deps:
  pip install pyarrow pandas
"""

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Constants (keep system stable)
# -----------------------------
EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius
STABLE_SYSTEM = "you are an agent that understands GPS and can calculate distances"

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class Coord:
    lat: float
    lon: float

# -----------------------------
# Utils
# -----------------------------
def choose(seq):
    return random.choice(seq)

def maybe(p: float) -> bool:
    return random.random() < max(0.0, min(1.0, p))

def random_coord() -> Coord:
    # avoid exact poles and ±180 to stay away from edge cases in text/noise
    return Coord(lat=random.uniform(-89.5, 89.5), lon=random.uniform(-179.5, 179.5))

# -----------------------------
# Phrase loader
# -----------------------------
def load_json(dirpath: Path, fname: str, default_obj):
    fp = dirpath / fname
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default_obj

# -----------------------------
# Coordinate formatting + noise
# -----------------------------
def _to_dms_components(value: float) -> Tuple[int,int,float,int]:
    sgn = 1 if value >= 0 else -1
    v = abs(value)
    d = int(v)
    mf = (v - d) * 60.0
    m = int(mf)
    s = round((mf - m) * 60.0, 2)
    if s >= 60: s = 0.0; m += 1
    if m >= 60: m = 0; d += 1
    return d, m, s, sgn

def _to_dmm_components(value: float) -> Tuple[int,float,int]:
    sgn = 1 if value >= 0 else -1
    v = abs(value)
    d = int(v)
    mm = round((v - d)*60.0, 3)
    if mm >= 60.0: mm = 0.0; d += 1
    return d, mm, sgn

def format_dd(c: Coord) -> str:
    return f"{c.lat:.6f}, {c.lon:.6f}"

def format_dms(c: Coord) -> str:
    dlat, mlat, slat, s1 = _to_dms_components(c.lat)
    dlon, mlon, slon, s2 = _to_dms_components(c.lon)
    Hlat = 'N' if s1 >= 0 else 'S'
    Hlon = 'E' if s2 >= 0 else 'W'
    return f"{dlat}° {mlat}′ {slat:.2f}″ {Hlat}, {dlon}° {mlon}′ {slon:.2f}″ {Hlon}"

def format_dmm(c: Coord) -> str:
    dlat, mlat, s1 = _to_dmm_components(c.lat)
    dlon, mlon, s2 = _to_dmm_components(c.lon)
    Hlat = 'N' if s1 >= 0 else 'S'
    Hlon = 'E' if s2 >= 0 else 'W'
    return f"{dlat}° {mlat:.3f}′ {Hlat}, {dlon}° {mlon:.3f}′ {Hlon}"

def noisy_prime_and_deg(text: str, noise: Dict) -> str:
    if maybe(noise.get("p_coord_replace_prime", 0.0)):
        text = text.replace("′", choose(["'", "", "’"]))
    if maybe(noise.get("p_coord_replace_dblprime", 0.0)):
        text = text.replace("″", choose(['"', "", "”"]))
    if maybe(noise.get("p_coord_drop_deg", 0.0)):
        text = text.replace("°", choose(["", " º", " deg"]))
    return text

def noisy_spacing_and_commas(text: str, noise: Dict) -> str:
    if maybe(noise.get("p_coord_extra_spaces", 0.0)):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*,\s*", choose([",", ", ", " ,", "  ,  ", " "]), text)
        text = text.replace("° ", choose(["° ", " °", "  ° ", "°"]))
        text = text.replace(" ′", choose([" ′", "′", "  ′"]))
        text = text.replace(" ″", choose([" ″", "″", "  ″"]))
    if maybe(noise.get("p_coord_remove_comma", 0.0)):
        text = text.replace(",", choose(["", " ", "  "]))
    return text

def maybe_drop_hemi(text: str, noise: Dict) -> str:
    if maybe(noise.get("p_coord_drop_hemi", 0.0)):
        for H in ["N","S","E","W"]:
            if H in text:
                text = text.replace(H, "", 1)
                break
    return text

def render_noisy_coord_string(fmt: str, c: Coord, noise: Dict) -> str:
    if fmt == "DD":
        s = format_dd(c)
        s = noisy_spacing_and_commas(s, noise)
        return s
    elif fmt == "DMS":
        s = format_dms(c)
        s = noisy_prime_and_deg(s, noise)
        s = noisy_spacing_and_commas(s, noise)
        s = maybe_drop_hemi(s, noise)
        return s
    else:
        s = format_dmm(c)
        s = noisy_prime_and_deg(s, noise)
        s = noisy_spacing_and_commas(s, noise)
        s = maybe_drop_hemi(s, noise)
        return s

# -----------------------------
# Units & parsing
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

def normalize_user_unit_guess(raw: Optional[str]) -> Optional[str]:
    if not raw: return None
    s = raw.strip().lower()
    s_comp = re.sub(r"[\s\.\-_/]+", "", s)
    # nautical
    if ("seamile" in s_comp) or ("sea mile" in s): return "nmi"
    if ("naut" in s) or ("nmi" in s_comp) or s_comp == "nm": return "nmi"
    # statute / miles
    if ("statute" in s) or ("statutemile" in s_comp) or ("statutemiles" in s_comp): return "mi"
    if ("mile" in s) or re.search(r"\bmi\b", s) or s_comp in ("m","sm"): return "mi"
    # kilometers
    if ("km" in s_comp) or ("klick" in s) or ("click" in s) or s_comp == "k" or ("meter" in s) or ("metre" in s):
        return "km"
    return None

# -----------------------------
# Haversine ground truth
# -----------------------------
def haversine_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg); lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg); lon2 = math.radians(lon2_deg)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = (math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return EARTH_RADIUS_KM * c

# -----------------------------
# Build user prompt from JSON patterns
# -----------------------------
def build_user_prompt(A_txt: str, B_txt: str, patterns: List[str], unit_variants: Dict[str, List[str]], noise: Dict) -> Tuple[str, Optional[str]]:
    modes = ["normal","short_q","coords_only","fragment","arrow","caps"]
    mode = choose(modes)
    raw_unit_hint: Optional[str] = None

    def pick_variant(canon: str) -> str:
        arr = unit_variants.get(canon, [canon])
        return choose(arr) if arr else canon

    if mode == "normal":
        canonical = choose(["km","mi","mi","mi","nmi"])  # bias miles
        raw_unit_hint = pick_variant(canonical)
        template = choose(patterns)
        text = template.format(A=A_txt, B=B_txt, unit=raw_unit_hint)
    elif mode == "short_q":
        if maybe(0.55):
            canonical = choose(["km","mi","mi","nmi"])
            raw_unit_hint = pick_variant(canonical)
            text = f"how far from {A_txt} to {B_txt} in {raw_unit_hint}?"
        else:
            text = f"how far from {A_txt} to {B_txt}?"
    elif mode == "coords_only":
        sep = choose([" ", "  ", "\n", "  →  "])
        text = f"{A_txt}{sep}{B_txt}"
    elif mode == "fragment":
        canonical = choose(["km","mi","mi","nmi"])
        raw_unit_hint = pick_variant(canonical)
        text = f"distance in {raw_unit_hint}: {A_txt} to {B_txt}"
    elif mode == "arrow":
        canonical = choose(["km","mi","mi","nmi"])
        raw_unit_hint = pick_variant(canonical)
        text = f"{A_txt} -> {B_txt} in {raw_unit_hint}?"
    else:
        canonical = choose(["km","mi","mi","nmi"])
        raw_unit_hint = pick_variant(canonical)
        text = f"DISTANCE??? {A_txt} {B_txt} ({raw_unit_hint})"

    # tiny user typos, not touching digits/signs/hemi letters
    if maybe(noise.get("p_user_typos", 0.0)):
        kept, drop = [], 0
        max_drop = max(1, int(len(text)*0.01))
        for ch in text:
            if ch.isdigit() or ch in "+-." or ch in "NSEW":
                kept.append(ch); continue
            if drop < max_drop and ch.isalpha() and maybe(0.25):
                drop += 1; continue
            kept.append(ch)
        text = "".join(kept)

    return text, raw_unit_hint

# -----------------------------
# One eval row
# -----------------------------
def generate_one_row(phrases_dir: Path) -> Dict:
    # Load phrases once per call (caller can pre-load for speed; fine for N~1e3)
    user_patterns = load_json(phrases_dir, "user_patterns.json", [
        "What is the distance between {A} and {B}, expressed in {unit}?",
        "Measure the distance from {A} to {B}, expressed in {unit}.",
        "How far is it from {A} to {B}, in {unit}?",
        "how far from {A} to {B}?",
        "distance in {unit}: {A} to {B}",
        "{A} -> {B} in {unit}?",
        "{A}\n{B}",
        "DISTANCE??? {A} {B}",
        "pls compute distance {A} {B}  ({unit})",
        "span from {A} to {B} ({unit})"
    ])
    unit_user_variants = load_json(phrases_dir, "unit_user_variants.json", {
        "km": ["km","Km","KM","kms","Kms","k","K","km.","klick","klicks","click","clicks",
               "kilometer","kilometre","kilometers","kilometres","meters","metres","meter","metre","mtrs","mtr"],
        "mi": ["mi","Mi","MI","mile","miles","sm","mi.","m","M","m.","give it to me in m","statute","statute miles"],
        "nmi": ["nmi","nm","NM","nautical mile","nautical miles","sea mile","sea miles","n.m.","n mi"]
    })
    noise_cfg = load_json(phrases_dir, "noise_config.json", {
        "p_coord_drop_deg": 0.25,
        "p_coord_replace_prime": 0.35,
        "p_coord_replace_dblprime": 0.35,
        "p_coord_remove_comma": 0.20,
        "p_coord_extra_spaces": 0.40,
        "p_coord_drop_hemi": 0.05,
        "p_user_typos": 0.08
    })

    # Choose a coordinate format
    fmt = choose(["DD","DMS","DMM"])
    A = random_coord(); B = random_coord()
    A_txt = render_noisy_coord_string(fmt, A, noise_cfg)
    B_txt = render_noisy_coord_string(fmt, B, noise_cfg)

    user_text, raw_unit = build_user_prompt(A_txt, B_txt, user_patterns, unit_user_variants, noise_cfg)

    # Canonical unit (default to miles if ambiguous/absent)
    unit_canonical = normalize_user_unit_guess(raw_unit) or "mi"

    # Ground truth
    d_km = haversine_km(A.lat, A.lon, B.lat, B.lon)
    expected_value, expected_unit = convert_km(d_km, unit_canonical)

    return {
        "developer": STABLE_SYSTEM,      # <- perfect & stable
        "user": user_text,               # <- noisy and randomized
        "expected_km": float(d_km),
        "expected_value": float(expected_value),
        "expected_unit": expected_unit,
        # debug metainfo (optional but handy)
        "fmt": fmt,
        "alat": A.lat, "alon": A.lon, "blat": B.lat, "blon": B.lon
    }

# -----------------------------
# Parquet helpers
# -----------------------------
def write_parquet(rows: List[Dict], out_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        pa.field("developer", pa.string()),
        pa.field("user", pa.string()),
        pa.field("expected_km", pa.float64()),
        pa.field("expected_value", pa.float64()),
        pa.field("expected_unit", pa.string()),
        pa.field("fmt", pa.string()),
        pa.field("alat", pa.float64()),
        pa.field("alon", pa.float64()),
        pa.field("blat", pa.float64()),
        pa.field("blon", pa.float64())
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path, compression="zstd")

def preview_parquet(out_path: Path, n: int = 3):
    import pyarrow.parquet as pq
    import pandas as pd
    df = pq.read_table(out_path).to_pandas()
    print(f"\nWrote: {out_path.resolve()}  rows={len(df)}")
    print(df.head(n).to_string(index=False))

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build evaluation set for GPS distance reasoning.")
    parser.add_argument("--n", type=int, default=1000, help="Number of eval examples")
    parser.add_argument("--out", type=str, default="eval_set.parquet", help="Output Parquet path")
    parser.add_argument("--phrases-dir", type=str, default="eval_phrases", help="Folder with JSON phrase files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-preview", action="store_true", help="Skip preview printout")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    phrases_dir = Path(args.phrases_dir)
    phrases_dir.mkdir(parents=True, exist_ok=True)

    rows = [generate_one_row(phrases_dir) for _ in range(args.n)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, out_path)
    if not args.no_preview:
        preview_parquet(out_path)

if __name__ == "__main__":
    main()

