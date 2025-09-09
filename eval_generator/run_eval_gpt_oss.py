#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_gpt_oss.py
-------------------
Evaluate a gpt-oss model with Transformers on a GPS distance reasoning eval set.

SCORING (explicit & logged):
- We parse the model’s output, preferring a dedicated Harmony “final” channel if present.
  If absent, we consider the whole text; we may also look at the “analysis” channel
  as a fallback when final has no plausible numeric candidates.
- We extract numeric candidates (robust to thousands separators like “26,300” or “9 330”)
  and any nearby unit tokens. For each candidate we compute a heuristic score:
    +5  if a normalized distance unit is present (km/mi/nmi/klick/etc.)
    +3  if near distance keywords (“distance”, “final”, “answer”, “result”, “≈”, “~”, “about”)
    +2  if near “final-ish” terms (“≈”, “~”, “about”, “roughly”, “final”, “answer”, “result”)
    +2  if located toward the tail of the message (tail bias)
    −6  if near latitude/longitude/DMS markers (°, deg, ′, ″, min, sec)
    −6  if near Earth-radius constants (e.g., “R=6371”, “mean radius”)
    −4  if near math-only markers (π, “rad”, “radian”)
    −1  if the magnitude is very small (< 5), to downweight incidental intermediates
  The per-candidate **score breakdown** is logged when --verbose is set.
- We rank candidates by (score desc, position desc, magnitude desc) and pick the top.
  If it lacks an explicit unit, we:
    (a) prefer a unit hinted by the user request (e.g., “sea miles/nm/nmi”, “klicks” => km),
        and log that decision; otherwise
    (b) pick the interpretation among km/mi/nmi whose value is closest to a robust
        reference magnitude formed from the top candidates.
- We compute relative error vs ground-truth kilometers:
      rel = |pred_km − expected_km| / expected_km
  Buckets:
      < 1%       → "perfect"       (10 points)
      < 5%       → "approx"         (3 points)
      < 10%      → "pretty_close"   (1 point)
      ≥ 10%      → "wrong"          (0 points)
- We report bucket counts, total points, and points per sample, and write the
  per-sample details to a Parquet file. The per-sample row includes the chosen
  candidate, normalized/assumed unit, converted km, relative error, and whether
  the analysis fallback was used.

Generation & model notes:
- HF `pipeline("text-generation")` is used with left padding on decoder-only models
  to avoid right-padding warnings. pad_token is set to eos if missing.
- Multi-GPU: you may set `--device-map auto` (default) and optionally `--reserve-gib`
  to leave headroom; the script computes a `max_memory` map when possible.
- Decoding: `--decoding deterministic` (greedy) or `--decoding sampling` with
  `--temperature`, `--top-k`, `--top-p`. When deterministic, sampling params are
  NOT passed to Transformers (prevents “ignored flag” warnings).

"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# -----------------------------
# Logging / console tee
# -----------------------------
class TeeLogger:
    """Writes to stdout and optionally to a file."""
    def __init__(self, log_path: Optional[Path] = None):
        self.file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(log_path, "w", encoding="utf-8")

    def write(self, s: str):
        sys.stdout.write(s)
        sys.stdout.flush()
        if self.file:
            self.file.write(s)
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

# -----------------------------
# Unit helpers
# -----------------------------
UNIT_SYNONYMS = {
    "km": {
        "km", "kms", "k", "kilometer", "kilometers", "kilometre", "kilometres",
        "km.", "klick", "klicks", "click", "clicks"
    },
    "mi": {"mi", "mile", "miles", "statute", "statute miles", "sm"},
    "nmi": {
        "nmi", "nm", "n.m.", "nautical", "nauticalmile", "nauticalmiles",
        "sea", "sea mile", "sea miles", "seamile", "seamiles"
    },
}

def to_km(val: float, unit: str) -> float:
    if unit == "km": return val
    if unit == "mi": return val * 1.609344
    if unit == "nmi": return val * 1.852
    return val

def normalize_unit(tok: Optional[str]) -> Optional[str]:
    if not tok:
        return None
    s = tok.strip().lower()
    s_comp = re.sub(r"[\s\.\-_/]+", "", s)
    if s in UNIT_SYNONYMS["km"] or s_comp in UNIT_SYNONYMS["km"]:
        return "km"
    if s in UNIT_SYNONYMS["mi"] or s_comp in UNIT_SYNONYMS["mi"] or "statute" in s:
        return "mi"
    if s in UNIT_SYNONYMS["nmi"] or s_comp in UNIT_SYNONYMS["nmi"] or "sea" in s:
        return "nmi"
    # Avoid bare 'm' ambiguity (meters vs miles) in noisy outputs.
    return None

def unit_preference_from_user(user_text: str) -> Optional[str]:
    u = user_text.lower()
    if any(t in u for t in ["sea mile", "sea miles", "nmi", "nm", "nautical"]):
        return "nmi"
    if any(t in u for t in ["klick", "klicks", "click", "clicks"]):
        return "km"
    if any(t in u for t in [" mile", " miles", "(mi", "statute"]):
        return "mi"
    if any(t in u for t in [" km", "kilometer", "kilometre"]):
        return "km"
    return None

# -----------------------------
# Fuzzy number parser
# -----------------------------
# Numerical tokens with optional unit word (localized thousands, decimals, NBSPs)
_NUM_RE = re.compile(
    r"(?P<num>[+-]?\d[\d\s\u00A0\u202F,\.]*\d|\d(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>klicks?|clicks?|km(?:s)?|kilometer(?:s)?|kilometre(?:s)?|mi\.?|mile(?:s)?|statute(?:\s+miles?)?|sm|"
    r"nmi|nm|n\.m\.|nautical(?:\s+miles?)?|sea(?:\s+miles?)?))?",
    flags=re.IGNORECASE
)

DEG_MARKERS = ["°", " deg", "deg ", "degrees", "′", "'", "″", "\"", "min", "sec"]
RADIUS_MARKERS = ["r=6371", "r = 6371", "earth radius", "mean radius", "6371.0088", "6371 km", "r≈6371"]
MATH_MARKERS = ["π", "pi", "rad", "radian"]
DISTANCE_MARKERS = ["distance", "final", "answer", "result", "≈", "~", "about", "roughly",
                    "km", "mi", "nmi", "nautical", "sea mile", "statute", "klick", "click"]

def _parse_float_locale(num_str: str) -> Optional[float]:
    """
    Robustly parse floats with thousands groupings (',', '.', spaces, NBSP).
    Heuristics:
      - If both '.' and ',' exist: last punctuation is the decimal mark; the other is thousands.
      - If only ',' exists: treat as thousands if pattern like 12,345 or 1,234,567; else decimal.
      - If only '.' exists: same rule but with '.' as thousands when groups of 3.
      - Also strip thin/regular spaces and apostrophe grouping.
    """
    s = (num_str
         .replace("\u00A0", " ")
         .replace("\u202F", " ")
         .strip())
    # strip quotes/apostrophes also used for grouping in some locales
    s = s.replace("’", "").replace("'", "")
    has_dot = "." in s
    has_com = "," in s

    def only_thousands_sep(txt: str, sep: str) -> bool:
        # e.g., 12,345 or 1,234,567 (no decimals)
        pattern = rf"^[+-]?\d{{1,3}}(?:\{sep}\d{{3}})+$"
        return re.fullmatch(pattern, txt.replace(" ", "")) is not None

    if has_dot and has_com:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_dot > last_com:
            # '.' is decimal; remove commas as thousands
            s_clean = s.replace(",", "").replace(" ", "")
        else:
            # ',' is decimal; remove dots as thousands then replace comma with '.'
            s_clean = s.replace(".", "").replace(" ", "")
            s_clean = s_clean.replace(",", ".")
    elif has_com and not has_dot:
        if only_thousands_sep(s, ","):
            s_clean = s.replace(",", "").replace(" ", "")
        else:
            s_clean = s.replace(" ", "").replace(",", ".")
    elif has_dot and not has_com:
        if only_thousands_sep(s, "."):
            s_clean = s.replace(".", "").replace(" ", "")
        else:
            s_clean = s.replace(" ", "")
    else:
        # no punctuation; remove spaces
        s_clean = s.replace(" ", "")

    # keep only valid float/exponent/sign chars
    s_clean = re.sub(r"[^0-9eE\.\+\-]", "", s_clean)
    if not s_clean:
        return None
    try:
        return abs(float(s_clean))
    except Exception:
        return None

@dataclass
class Candidate:
    raw_value: float
    unit_raw: Optional[str]
    unit_norm: Optional[str]
    start: int
    end: int
    context: str
    score: int
    score_detail: Dict[str, int]
    km_interps: Dict[str, float]

def _near_any(text: str, idx: int, window: int, needles: Sequence[str]) -> bool:
    lo = max(0, idx - window)
    hi = min(len(text), idx + window)
    seg = text[lo:hi].lower()
    return any(n in seg for n in needles)

def extract_candidates_with_context(text: str, tail_bias_chars: int = 1200) -> List[Candidate]:
    out: List[Candidate] = []
    t = text
    for m in _NUM_RE.finditer(t):
        num_s = m.group("num")
        unit_s = m.group("unit")
        val = _parse_float_locale(num_s)
        if val is None:
            continue
        start, end = m.span()
        lo = max(0, start - 40)
        hi = min(len(t), end + 40)
        ctx = t[lo:hi]

        unit_norm = normalize_unit(unit_s) if unit_s else None

        # Score with breakdown
        sd: Dict[str, int] = {}
        score = 0
        if unit_norm:
            sd["unit_bonus"] = 5
            score += 5
        if _near_any(t, start, 80, DISTANCE_MARKERS):
            sd["near_keyword"] = sd.get("near_keyword", 0) + 3
            score += 3
        if _near_any(t, start, 40, ["≈", "~", "about", "roughly", "final", "answer", "result"]):
            sd["finalish"] = sd.get("finalish", 0) + 2
            score += 2
        if start > len(t) - tail_bias_chars:
            sd["tail_bias"] = 2
            score += 2
        if _near_any(t, start, 20, DEG_MARKERS):
            sd["dms_penalty"] = -6
            score -= 6
        if _near_any(t, start, 30, MATH_MARKERS):
            sd["math_penalty"] = -4
            score -= 4
        if _near_any(t, start, 40, RADIUS_MARKERS):
            sd["radius_penalty"] = -6
            score -= 6
        if val < 5.0:
            sd["tiny_penalty"] = -1
            score -= 1

        out.append(Candidate(
            raw_value=val,
            unit_raw=unit_s,
            unit_norm=unit_norm,
            start=start,
            end=end,
            context=ctx,
            score=score,
            score_detail=sd,
            km_interps={
                "km": to_km(val, "km"),
                "mi": to_km(val, "mi"),
                "nmi": to_km(val, "nmi"),
            },
        ))

    # De-duplicate by (value, unit_norm, start,end)
    dedup: Dict[Tuple[float, Optional[str], int, int], Candidate] = {}
    for c in out:
        key = (round(c.raw_value, 12), c.unit_norm, c.start, c.end)
        if key not in dedup:
            dedup[key] = c
    return list(dedup.values())

# -----------------------------
# Harmony / channel extraction
# -----------------------------
FINAL_MARK = "<|channel|>final<|message|>"
ANALYSIS_MARK = "<|channel|>analysis<|message|>"

def _to_text_from_pipeline_obj(obj: Any) -> str:
    """Normalize HF pipeline outputs to a printable string."""
    if isinstance(obj, str):
        return obj
    # List[dict] from some chat pipelines
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def extract_channel_text(full: str, mark: str) -> Optional[str]:
    if mark in full:
        seg = full.split(mark, 1)[-1]
        seg = re.split(r"<\|channel\|>", seg)[0]
        return seg.strip()
    return None

def extract_final_and_analysis(full_obj: Any) -> Tuple[str, Optional[str]]:
    full = _to_text_from_pipeline_obj(full_obj)
    # Harmony channel markers
    final_text = extract_channel_text(full, FINAL_MARK)
    analysis_text = extract_channel_text(full, ANALYSIS_MARK)
    if not final_text:
        # Heuristic: try simple 'analysis...final...' styles seen in logs
        #  e.g., "analysisThe user ... \nassistantfinal**Distance ...**"
        m_final = re.search(r"(assistant)?\s*final\b[:\*]*", full, flags=re.IGNORECASE)
        if m_final:
            final_text = full[m_final.end():].strip()
        else:
            final_text = full.strip()
    return final_text, analysis_text

# -----------------------------
# Scoring buckets
# -----------------------------
def bucket_points(expected_km: float, pred_km: Optional[float]) -> Tuple[str, int, float]:
    if expected_km <= 0 or pred_km is None:
        return ("wrong", 0, float("inf"))
    rel = abs(pred_km - expected_km) / expected_km
    if rel < 0.01: return ("perfect", 10, rel)
    if rel < 0.05: return ("approx", 3, rel)
    if rel < 0.10: return ("pretty_close", 1, rel)
    return ("wrong", 0, rel)

# -----------------------------
# Model loader & batched inference
# -----------------------------
def compute_max_memory_strings(reserve_gib: Optional[float]) -> Optional[Dict[Any, str]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        n = torch.cuda.device_count()
        if n == 0:
            return None
        mm = {}
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            total_gib = props.total_memory / (1024**3)
            leave = max(0.0, float(reserve_gib or 0.0))
            alloc = max(1.0, total_gib - leave)  # leave at least 1 GiB allocable
            mm[i] = f"{int(alloc)}GiB"
        mm["cpu"] = f"{int(96)}GiB"  # large headroom for offload; adjust if desired
        return mm
    except Exception:
        return None

def load_pipeline(model_name: str,
                  device_map: Optional[str] = "auto",
                  reserve_gib: Optional[float] = None):
    """
    Load tokenizer/model/pipeline for gpt-oss Harmony chat.

    - Left padding for decoder-only models.
    - Pad token set to EOS if undefined.
    - device_map and optional max_memory for multi-GPU sharding.
    """
    tok = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": "auto",  # new param; avoids the "torch_dtype deprecated" warning
        "low_cpu_mem_usage": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    max_mem = compute_max_memory_strings(reserve_gib)
    if max_mem:
        model_kwargs["max_memory"] = max_mem

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    gen = pipeline("text-generation", model=model, tokenizer=tok)
    return gen, tok, model_kwargs

def build_messages(system_text: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

def generate_batched(
    gen,
    batch_messages: List[List[Dict[str, str]]],
    decoding: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> List[str]:
    """
    Call the HF pipeline on a batch of Harmony messages.
    Only pass sampling args when decoding == 'sampling' to avoid warnings.
    """
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if decoding == "sampling":
        gen_kwargs.update({
            "do_sample": True,
            "temperature": float(temperature),
        })
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False

    out = gen(batch_messages, **gen_kwargs)
    generated_texts: List[str] = []
    for item in out:
        if isinstance(item, dict) and "generated_text" in item:
            generated_texts.append(item["generated_text"])
        elif isinstance(item, list) and len(item) and isinstance(item[0], dict) and "generated_text" in item[0]:
            generated_texts.append(item[0]["generated_text"])
        else:
            generated_texts.append(_to_text_from_pipeline_obj(item))
    return generated_texts, gen_kwargs

# -----------------------------
# Candidate selection & evaluation
# -----------------------------
def choose_candidate(
    final_text: str,
    analysis_text: Optional[str],
    user_text: str,
    expected_km: float,
    logger: TeeLogger,
    print_candidates: int = 10,
) -> Tuple[Optional[Candidate], Optional[str], Optional[float], bool]:
    """
    Extract and choose a candidate from final_text; if none score > threshold,
    try analysis_text. Returns:
      (candidate, chosen_assumed_unit, chosen_km_value, used_analysis_fallback)
    """
    unit_pref = unit_preference_from_user(user_text)
    used_analysis = False

    def _rank_and_pick(text: str) -> Tuple[Optional[Candidate], Optional[str], Optional[float], List[Candidate]]:
        cands = extract_candidates_with_context(text)
        if not cands:
            return None, None, None, []
        cands_sorted = sorted(cands, key=lambda c: (c.score, c.start, c.raw_value), reverse=True)

        # Pretty print top candidates (value [unit] -> km/mi/nmi)
        if print_candidates > 0:
            logger.write(">> NUMERIC CANDIDATES (value [unit_raw] -> km/mi/nmi)\n")
            for c in cands_sorted[:print_candidates]:
                logger.write(f"  {c.raw_value}  [{'-' if c.unit_raw is None else c.unit_raw}]  "
                             f"->  km:{c.km_interps['km']:.6f}, mi:{c.km_interps['mi']:.6f}, nmi:{c.km_interps['nmi']:.6f}\n")

        top = cands_sorted[0]
        assumed = top.unit_norm
        chosen_km = None

        if top.unit_norm is not None:
            chosen_km = to_km(top.raw_value, top.unit_norm)
            assumed = top.unit_norm
        else:
            if unit_pref in ("km", "mi", "nmi"):
                chosen_km = to_km(top.raw_value, unit_pref)
                assumed = unit_pref
            else:
                # Reference magnitude = median of km-interps over up to 5 best candidates
                pack = cands_sorted[: min(5, len(cands_sorted))]
                ref_vals = [v for c in pack for v in (c.km_interps["km"], c.km_interps["mi"], c.km_interps["nmi"])]
                ref_vals.sort()
                ref = ref_vals[len(ref_vals)//2] if ref_vals else None
                best_u, best_diff = None, float("inf")
                for u in ("km", "mi", "nmi"):
                    km_val = top.km_interps[u]
                    diff = abs(km_val - (ref if ref is not None else km_val))
                    if diff < best_diff:
                        best_diff = diff
                        best_u = u
                        chosen_km = km_val
                assumed = best_u
        return top, assumed, chosen_km, cands_sorted

    cand, assumed_unit, chosen_km, sorted_list = _rank_and_pick(final_text)

    # If none or very weak, try analysis as a fallback
    if cand is None or cand.score <= 0:
        if analysis_text:
            cand2, assumed2, km2, sorted_list2 = _rank_and_pick(analysis_text)
            if cand2 is not None and (cand is None or cand2.score > cand.score):
                cand, assumed_unit, chosen_km = cand2, assumed2, km2
                sorted_list = sorted_list2
                used_analysis = True

    # Last-resort tail of final_text
    if cand is None:
        tail = final_text[-400:]
        cand3, assumed3, km3, _ = _rank_and_pick(tail)
        if cand3 is not None:
            cand, assumed_unit, chosen_km = cand3, assumed3, km3

    # Decision trace for the winner
    if cand is not None:
        logger.write(">> DECISION TRACE (winning candidate)\n")
        logger.write(f"  value={cand.raw_value}  unit_raw={cand.unit_raw!r}  unit_norm={cand.unit_norm!r}\n")
        logger.write(f"  assumed_unit={assumed_unit}  parsed_km={chosen_km}\n")
        logger.write(f"  score={cand.score}  breakdown={cand.score_detail}  span=[{cand.start},{cand.end}]\n")
        logger.write(f"  context=...{cand.context}...\n")
    else:
        logger.write(">> DECISION TRACE: no numeric candidates found\n")

    return cand, assumed_unit, chosen_km, used_analysis

# -----------------------------
# Eval loop
# -----------------------------
def run_eval(
    eval_file: Path,
    model_name: str,
    out_parquet: Path,
    out_summary: Path,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    decoding: str = "deterministic",   # 'deterministic' or 'sampling'
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    limit: Optional[int] = None,
    verbose: bool = False,
    log_path: Optional[Path] = None,
    device_map: Optional[str] = "auto",
    reserve_gib: Optional[float] = None,
    print_limit: int = 1200,
    channel_limit: int = -1,
    print_candidates: int = 10,
) -> Dict:
    logger = TeeLogger(log_path)

    # One-time algorithm summary
    if verbose:
        logger.write("SCORING & SELECTION ALGORITHM\n")
        logger.write("  +5 unit, +3 near distance keywords, +2 final-ish cues, +2 tail bias\n")
        logger.write("  −6 DMS, −6 Earth radius, −4 math-only, −1 tiny magnitude (<5)\n")
        logger.write("  Rank by (score desc, pos desc, magnitude desc); infer unit if missing.\n")
        logger.write("  Buckets: <1% 10pts | <5% 3pts | <10% 1pt | else 0pt\n")
        logger.write("  Fallbacks: analysis channel, then final tail.\n")
        logger.write("─" * 80 + "\n")

    df = pq.read_table(eval_file).to_pandas()
    if limit is not None:
        df = df.head(limit).copy()

    gen, tok, model_kwargs = load_pipeline(model_name, device_map=device_map, reserve_gib=reserve_gib)

    # summarize run config
    if verbose:
        logger.write(f"Loaded {len(df)} eval rows from {eval_file.name}\n")
        logger.write(f"Model: {model_name}\n")
        logger.write(f"Decoding: {decoding} (kwargs will reflect sampling only if sampling)\n")
        if model_kwargs:
            logger.write(f"Multi-GPU: {'auto-shard' if model_kwargs.get('device_map')=='auto' else model_kwargs.get('device_map')} "
                         f"(model_kwargs: {json.dumps(model_kwargs, indent=2)})\n")
        logger.write(f"Tokenizer padding_side={tok.padding_side} pad_token_id={tok.pad_token_id}\n")
        logger.write("─" * 80 + "\n")

    results = []
    buckets = {"perfect": 0, "approx": 0, "pretty_close": 0, "wrong": 0}
    total_points = 0

    n = len(df)
    batches = math.ceil(n / batch_size)

    for b in range(batches):
        lo = b * batch_size
        hi = min(n, (b + 1) * batch_size)
        batch = df.iloc[lo:hi]

        # Build batch of Harmony messages
        batch_messages = [build_messages(row["developer"], row["user"]) for _, row in batch.iterrows()]

        t0 = time.time()
        gen_texts, used_gen_kwargs = generate_batched(
            gen,
            batch_messages,
            decoding=decoding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        dt = time.time() - t0

        if verbose:
            logger.write(f"\nBATCH {b+1}/{batches}  rows {lo}..{hi-1}\n")
            logger.write(f"Generation kwargs: {json.dumps(used_gen_kwargs)}\n")

        for (i, row), full_text in zip(batch.iterrows(), gen_texts):
            system_text = row["developer"]
            user_text = row["user"]
            expected_km = float(row["expected_km"])

            # Extract Harmony channels
            final_text, analysis_text = extract_final_and_analysis(full_text)

            if verbose:
                logger.write("\n" + "═" * 80 + "\n")
                logger.write(f"SAMPLE {i+1}/{n}  (batch {lo}-{hi-1})\n")
                logger.write("─" * 80 + "\n")
                logger.write(">> MESSAGES SENT\n")
                logger.write(f"[DEVELOPER] {system_text}\n")
                logger.write(f"[USER] {user_text}\n")
                logger.write("─" * 80 + "\n")
                # Raw output (possibly huge)
                raw = _to_text_from_pipeline_obj(full_text)
                if print_limit is not None and print_limit >= 0:
                    logger.write(">> RAW PIPELINE OUTPUT\n")
                    snippet = raw if print_limit == -1 else (raw[:print_limit] + ("…" if len(raw) > print_limit else ""))
                    logger.write(snippet + ("\n" if not snippet.endswith("\n") else ""))
                # Channels (full by default as requested)
                logger.write(">> EXTRACTED ANALYSIS TEXT\n")
                if analysis_text:
                    if channel_limit is None or channel_limit < 0:
                        logger.write(analysis_text + ("\n" if not analysis_text.endswith("\n") else ""))
                    else:
                        at = analysis_text[:channel_limit] + ("…" if len(analysis_text) > channel_limit else "")
                        logger.write(at + ("\n" if not at.endswith("\n") else ""))
                else:
                    logger.write("(none)\n")
                logger.write(">> EXTRACTED FINAL TEXT\n")
                if channel_limit is None or channel_limit < 0:
                    logger.write(final_text + ("\n" if not final_text.endswith("\n") else ""))
                else:
                    ft = final_text[:channel_limit] + ("…" if len(final_text) > channel_limit else "")
                    logger.write(ft + ("\n" if not ft.endswith("\n") else ""))

            cand, assumed, pred_km, used_analysis = choose_candidate(
                final_text=final_text,
                analysis_text=analysis_text,
                user_text=user_text,
                expected_km=expected_km,
                logger=logger if verbose else TeeLogger(None),
                print_candidates=print_candidates
            )

            bucket, pts, rel = bucket_points(expected_km, pred_km)

            if verbose:
                logger.write(">> DECISION & SCORE\n")
                logger.write(f"expected_km={expected_km}\n")
                if cand is None:
                    logger.write("parsed: NONE\n")
                else:
                    logger.write(f"parsed_raw: {cand.raw_value} [{cand.unit_raw if cand.unit_raw else '—'}]  "
                                 f"explicit_norm={cand.unit_norm if cand.unit_norm else '—'}\n")
                    logger.write(f"assumed_unit={assumed}  parsed_km={pred_km}  rel_error={rel:.2%}\n")
                    flags = {
                        "near_keyword": "near_keyword" in (cand.score_detail or {}),
                        "radius_context": "radius_penalty" in (cand.score_detail or {})
                    }
                    logger.write(f"flags={flags}  analysis_fallback_used={used_analysis}\n")
                    logger.write(f"context: ...{cand.context}...\n")
                logger.write(f"bucket={bucket}  points={pts}\n")

            results.append({
                "idx": int(i),
                "developer": system_text,
                "user": user_text,
                "model_output_raw": _to_text_from_pipeline_obj(full_text),
                "model_output_final": final_text,
                "model_output_analysis": analysis_text,
                "expected_km": expected_km,
                "parsed_value": None if cand is None else cand.raw_value,
                "parsed_unit_raw": None if cand is None else cand.unit_raw,
                "parsed_unit_norm": None if cand is None else cand.unit_norm,
                "assumed_unit_for_match": assumed,
                "parsed_km": pred_km,
                "rel_error": rel,
                "bucket": bucket,
                "points": pts,
                "used_analysis_fallback": used_analysis,
                "score_breakdown": None if cand is None else cand.score_detail,
                "span_start": None if cand is None else cand.start,
                "span_end": None if cand is None else cand.end,
            })
            if bucket in buckets:
                buckets[bucket] += 1
            total_points += pts

        if verbose:
            logger.write(f"-- batch generation time: {dt:.2f}s\n")

    # write outputs
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(results)), out_parquet, compression="zstd")
    summary = {
        "model": model_name,
        "eval_file": str(eval_file),
        "results_file": str(out_parquet),
        "n": len(results),
        "bucket_counts": buckets,
        "total_points": total_points,
        "points_per_sample": (total_points / len(results)) if results else 0.0
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        logger.write("\n" + "=" * 80 + "\n")
        logger.write(json.dumps(summary, indent=2) + "\n")

    logger.close()
    return summary

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation against gpt-oss (Transformers pipeline, Harmony).")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to eval_set.parquet")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="HF model id")
    parser.add_argument("--out", type=str, default="eval_results.parquet", help="Output Parquet file")
    parser.add_argument("--summary", type=str, default="eval_summary.json", help="Summary JSON")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for pipeline generation")
    parser.add_argument("--max-new-tokens", type=int, default=600)

    # Decoding controls
    parser.add_argument("--decoding", type=str, choices=["deterministic", "sampling"], default="deterministic",
                        help="Greedy vs sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (sampling only)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (sampling only)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (sampling only)")

    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N rows only")

    # Logging & verbosity
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-sample logs")
    parser.add_argument("--log", type=str, default=None, help="Path to logfile; if set, tee console output here")
    parser.add_argument("--print-limit", type=int, default=1200,
                        help="Max chars of RAW pipeline output to print; -1=full")
    parser.add_argument("--channel-limit", type=int, default=-1,
                        help="Max chars of ANALYSIS/FINAL text to print; -1=full")
    parser.add_argument("--print-candidates", type=int, default=10,
                        help="Print top-K numeric candidates per sample (0 to disable)")

    # Device / memory options (compatibility with your previous flags)
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Transformers device_map (e.g., auto, cuda:0).")
    parser.add_argument("--reserve-gib", type=float, default=None,
                        help="Leave this many GiB free per GPU (approx).")
    parser.add_argument("--hf-batch-size", type=int, default=None,
                        help="Alias for --batch-size (if set, overrides).")
    parser.add_argument("--multi-gpu", type=str, default=None,
                        help="Compatibility shim; if 'auto-shard', device_map=auto.")
    # Back-compat no-ops (accepted to avoid argparse errors)
    parser.add_argument("--print-limit-legacy", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # legacy flag wiring
    if args.multi_gpu == "auto-shard" and args.device_map != "auto":
        args.device_map = "auto"
    if args.hf_batch_size is not None:
        args.batch_size = args.hf_batch_size
    if args.print_limit_legacy is not None:
        args.print_limit = args.print_limit_legacy

    log_path = Path(args.log) if args.log else None

    summary = run_eval(
        eval_file=Path(args.eval_file),
        model_name=args.model,
        out_parquet=Path(args.out),
        out_summary=Path(args.summary),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        decoding=args.decoding,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        limit=args.limit,
        verbose=args.verbose,
        log_path=log_path,
        device_map=args.device_map,
        reserve_gib=args.reserve_gib,
        print_limit=args.print_limit,
        channel_limit=args.channel_limit,
        print_candidates=args.print_candidates,
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

