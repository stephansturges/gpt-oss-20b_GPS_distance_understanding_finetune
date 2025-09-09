#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_gpt_oss.py
-------------------
Evaluate a gpt-oss model with Transformers on a GPS distance reasoning eval set.

- Loads Parquet with 'developer' (stable system) + 'user' prompts
- Calls an off-the-shelf 'openai/gpt-oss-*' model via Transformers pipeline
  using the chat template that renders messages in the **Harmony** format
  automatically (final/analysis channels).  (Cookbook + HF docs)  # noqa
- Extracts the Harmony 'final' channel; floats are parsed fuzzily;
  units are normalized and numbers are interpreted as km/mi/nmi to find the
  best match vs ground truth. Scores by buckets (10/3/1/0).

Usage:
  pip install -U transformers accelerate torch pyarrow pandas
  python3 run_eval_gpt_oss.py --eval-file eval_set.parquet \
      --model openai/gpt-oss-20b --out eval_results.parquet \
      --summary eval_summary.json --max-new-tokens 600 --temperature 0.0

References:
- Run gpt-oss with Transformers (pipeline & chat template).  # noqa
- HF model cards (chat template auto-applies Harmony).        # noqa
- OpenAI Harmony channels (final / analysis).                 # noqa
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import pipeline

# -----------------------------
# Unit helpers
# -----------------------------
UNIT_SYNONYMS = {
    "km": {"km","kms","k","kilometer","kilometers","kilometre","kilometres","km.","klick","klicks","click","clicks"},
    "mi": {"mi","mile","miles","statute","sm","m","m."},
    "nmi": {"nmi","nm","n.m.","nautical","nauticalmile","nauticalmiles","sea","seamile","seamiles"}
}

def to_km(val: float, unit: str) -> float:
    if unit == "km": return val
    if unit == "mi": return val * 1.609344
    if unit == "nmi": return val * 1.852
    return val

def normalize_unit(tok: Optional[str]) -> Optional[str]:
    if not tok: return None
    s = tok.strip().lower()
    s_comp = re.sub(r"[\s\.\-_/]+", "", s)
    if s in UNIT_SYNONYMS["km"] or s_comp in UNIT_SYNONYMS["km"]: return "km"
    if s in UNIT_SYNONYMS["mi"] or s_comp in UNIT_SYNONYMS["mi"] or "statute" in s: return "mi"
    if s in UNIT_SYNONYMS["nmi"] or s_comp in UNIT_SYNONYMS["nmi"] or "sea" in s: return "nmi"
    if s == "m": return "mi"  # your dataset convention: ambiguous "m"→ miles
    return None

# -----------------------------
# Fuzzy number parser
# -----------------------------
_NUM_RE = re.compile(
    r"(?P<num>[+-]?\d[\d\s\u00A0\u202F,\.]*\d|\d)"
    r"(?:\s*(?P<unit>km(?:s)?|kilometer(?:s)?|kilometre(?:s)?|mi\.?|mile(?:s)?|statute(?:\s+miles)?|sm|m\b|"
    r"nmi|nm|n\.m\.|nautical(?:\s+miles?)?|sea(?:\s+miles?)?|klicks?|clicks?))?",
    flags=re.IGNORECASE
)

def _parse_float_locale(num_str: str) -> Optional[float]:
    s = num_str.replace("\u00A0", " ").replace("\u202F", " ").strip()
    last_dot = s.rfind(".")
    last_com = s.rfind(",")
    if last_dot == -1 and last_com == -1:
        digits = re.sub(r"[^\d\-+]", "", s)
        if not digits: return None
        try: return float(digits)
        except: return None
    # choose rightmost as decimal; treat others as thousands marks
    if last_dot > last_com:
        s_clean = s.replace(",", "").replace(" ", "")
    else:
        s_clean = s.replace(".", "").replace(" ", "")
        s_clean = s_clean.replace(",", ".")
    s_clean = re.sub(r"[^\d\.\-+eE]", "", s_clean)
    try:
        val = float(s_clean)
        return abs(val)
    except:
        return None

@dataclass
class Candidate:
    raw_value: float
    unit_raw: Optional[str]
    unit_norm: Optional[str]
    km_interps: Dict[str, float]

def extract_candidates(text: str) -> List[Candidate]:
    out: List[Candidate] = []
    for m in _NUM_RE.finditer(text):
        n = _parse_float_locale(m.group("num"))
        if n is None: continue
        u = m.group("unit")
        u_norm = normalize_unit(u) if u else None
        out.append(Candidate(
            raw_value=n,
            unit_raw=u,
            unit_norm=u_norm,
            km_interps={"km": to_km(n,"km"), "mi": to_km(n,"mi"), "nmi": to_km(n,"nmi")}
        ))
    return out

# -----------------------------
# Harmony 'final' extraction
# -----------------------------
FINAL_MARK = "<|channel|>final<|message|>"

def extract_final_text(generated_text: str) -> str:
    if FINAL_MARK in generated_text:
        tail = generated_text.split(FINAL_MARK, 1)[-1]
        # stop before any next channel header if present
        tail = re.split(r"<\|channel\|>", tail)[0]
        return tail.strip()
    return generated_text.strip()

# -----------------------------
# Scoring
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
# Model wrapper (Transformers pipeline)
# -----------------------------
def load_generator(model_name: str, device_map: str = "auto"):
    """
    Use the HF pipeline which understands 'messages=[...]' for gpt-oss models and
    applies the Harmony chat template automatically (per model card/cookbook).
    """
    gen = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map=device_map
    )
    return gen

def call_model(gen, system_text: str, user_text: str, max_new_tokens: int, temperature: float) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    out = gen(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return out[0]["generated_text"]

# -----------------------------
# Eval loop
# -----------------------------
def run_eval(eval_file: Path, model_name: str, out_parquet: Path, out_summary: Path,
             max_new_tokens: int = 512, temperature: float = 0.0, limit: Optional[int] = None) -> Dict:
    df = pq.read_table(eval_file).to_pandas()
    if limit is not None:
        df = df.head(limit).copy()

    gen = load_generator(model_name)

    results = []
    buckets = {"perfect": 0, "approx": 0, "pretty_close": 0, "wrong": 0}
    total_points = 0

    for i, row in df.iterrows():
        system_text = row["developer"]
        user_text = row["user"]
        expected_km = float(row["expected_km"])

        try:
            raw = call_model(gen, system_text, user_text, max_new_tokens, temperature)
        except Exception as e:
            raw = f"[MODEL ERROR: {e}]"

        final_text = extract_final_text(raw)
        cands = extract_candidates(final_text)

        best = None
        best_rel = float("inf")
        best_km = None
        best_unit_assumption = None

        for c in cands:
            for assume, km_val in c.km_interps.items():
                rel = abs(km_val - expected_km) / expected_km if expected_km > 0 else float("inf")
                if rel < best_rel:
                    best_rel = rel
                    best = c
                    best_km = km_val
                    best_unit_assumption = assume

        bucket, pts, rel = bucket_points(expected_km, best_km)
        results.append({
            "idx": int(i),
            "developer": system_text,
            "user": user_text,
            "model_output_raw": raw,
            "model_output_final": final_text,
            "expected_km": expected_km,
            "parsed_value": (None if best is None else best.raw_value),
            "parsed_unit_raw": (None if best is None else best.unit_raw),
            "parsed_unit_norm": (None if best is None else best.unit_norm),
            "assumed_unit_for_match": best_unit_assumption,
            "parsed_km": best_km,
            "rel_error": rel,
            "bucket": bucket,
            "points": pts
        })
        if bucket in buckets: buckets[bucket] += 1
        total_points += pts

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
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation against gpt-oss (Transformers pipeline).")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to eval_set.parquet")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="HF model id")
    parser.add_argument("--out", type=str, default="eval_results.parquet", help="Output Parquet file")
    parser.add_argument("--summary", type=str, default="eval_summary.json", help="Summary JSON")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N rows only")
    args = parser.parse_args()

    summary = run_eval(
        eval_file=Path(args.eval_file),
        model_name=args.model,
        out_parquet=Path(args.out),
        out_summary=Path(args.summary),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

