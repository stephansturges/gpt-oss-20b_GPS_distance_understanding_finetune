#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_gpt_oss.py — strict, verbose, multi‑GPU evaluator with logging
-----------------------------------------------------------------------
- Uses Transformers pipeline + Harmony chat template (GPT‑OSS) with robust output decoding.
- Strict numeric parsing from Harmony 'final' channel (analysis ignored by default).
- Multi‑GPU sharding via device_map="auto" + per‑GPU max_memory (Accelerate).
- Left padding + pad token for decoder‑only models (fixes right‑padding warning).
- Sampling flags are sanitized (no temperature/top_p/top_k unless do_sample=True).
- Verbose tracing and a --log flag to tee all console output to a file.

CLI examples:
  python3 run_eval_gpt_oss.py --eval-file eval_set.parquet \
    --model openai/gpt-oss-20b \
    --decoding deterministic \
    --batch-size 8 --hf-batch-size 8 \
    --max-new-tokens 800 --verbose --print-limit 8 \
    --log run_log.txt
"""

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import pipeline, set_seed, AutoTokenizer

# =============================================================================
# Logging helpers (with optional tee to a file via --log)
# =============================================================================

LOG_FH = None  # file handle for --log

def init_log_file(path: Optional[Path]):
    global LOG_FH
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        LOG_FH = open(path, "w", encoding="utf-8")

def log(s: str, *, file=sys.stdout, flush=True):
    print(s, file=file, flush=flush)
    if LOG_FH is not None:
        print(s, file=LOG_FH, flush=flush)

def hline(char="─", n=80):
    return char * n

# =============================================================================
# Unit helpers
# =============================================================================

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
    if s == "m": return "mi"  # project convention for ambiguous "m"
    return None

# =============================================================================
# Numeric candidate parser (strict)
# =============================================================================

_NUM_RE = re.compile(
    r"(?P<num>[+-]?\d[\d\s\u00A0\u202F,\.]*\d|\d)"
    r"(?:\s*(?P<unit>km(?:s)?|kilometer(?:s)?|kilometre(?:s)?|mi\.?|mile(?:s)?|statute(?:\s+miles)?|sm|m\b|"
    r"nmi|nm|n\.m\.|nautical(?:\s+miles?)?|sea(?:\s+miles?)?|klicks?|clicks?))?",
    flags=re.IGNORECASE
)

KEYWORD_RE = re.compile(
    r"(final|answer|result|distance|dist|km|kilometer|kilometre|mi\b|mile|miles|nmi|nautical|statute|≈|about|approx|~)",
    flags=re.IGNORECASE
)

# constants often present in analysis; we filter unless strong answer context
BAN_CONSTANTS = {6371.0, 1.852, 1.609344, 60.0, 3600.0, 111.0, 69.0}

@dataclass
class Candidate:
    raw_value: float
    unit_raw: Optional[str]
    unit_norm: Optional[str]
    span_start: int
    span_end: int
    chosen_unit: Optional[str] = None
    parsed_km: Optional[float] = None
    context_snippet: Optional[str] = None
    flags: Dict[str, bool] = None

def _parse_float_locale(num_str: str) -> Optional[float]:
    s = num_str.replace("\u00A0", " ").replace("\u202F", " ").strip()
    last_dot = s.rfind("."); last_com = s.rfind(",")
    if last_dot == -1 and last_com == -1:
        digits = re.sub(r"[^\d\-+]", "", s)
        if not digits: return None
        try: return float(digits)
        except: return None
    if last_dot > last_com:
        s_clean = s.replace(",", "").replace(" ", "")
    else:
        s_clean = s.replace(".", "").replace(" ", "").replace(",", ".")
    s_clean = re.sub(r"[^\d\.\-+eE]", "", s_clean)
    try:
        return abs(float(s_clean))
    except:
        return None

def extract_candidates(text: str) -> List[Candidate]:
    out: List[Candidate] = []
    for m in _NUM_RE.finditer(text):
        n = _parse_float_locale(m.group("num"))
        if n is None: continue
        u = m.group("unit")
        u_norm = normalize_unit(u) if u else None
        out.append(Candidate(
            raw_value=n, unit_raw=u, unit_norm=u_norm,
            span_start=m.start(), span_end=m.end(),
            flags={}
        ))
    return out

# =============================================================================
# Harmony extraction
# =============================================================================

FINAL_MARK = "<|channel|>final<|message|>"

def extract_final_from_str(generated_text: str) -> Tuple[str, bool]:
    s = generated_text
    if FINAL_MARK in s:
        tail = s.split(FINAL_MARK, 1)[-1]
        tail = re.split(r"(?:<\|channel\|>|<\|return\|>)", tail)[0]
        return tail.strip(), False
    analysis_like = bool(re.match(r"\s*analysis\b", s, flags=re.IGNORECASE))
    return s.strip(), analysis_like

def extract_final_from_msgs(msgs: List[Any]) -> Tuple[str, bool]:
    if not msgs:
        return "", True
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("channel") == "final":
            return (m.get("content") or "").strip(), False
    assistants = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    if assistants:
        content = (assistants[-1].get("content") or "").strip()
        analysis_like = bool(re.match(r"\s*analysis\b", content, flags=re.IGNORECASE))
        return content, analysis_like
    last = msgs[-1]
    if isinstance(last, dict):
        content = (last.get("content") or "").strip()
        analysis_like = bool(re.match(r"\s*analysis\b", content, flags=re.IGNORECASE))
        return content, analysis_like
    return str(last).strip(), True

def extract_final_text_from_pipeline_output(gen_output_obj: Any) -> Tuple[str, str, bool]:
    if isinstance(gen_output_obj, str):
        text, is_analysis = extract_final_from_str(gen_output_obj)
        return text, "string", is_analysis
    if isinstance(gen_output_obj, list):
        text, is_analysis = extract_final_from_msgs(gen_output_obj)
        return text, "list_of_messages", is_analysis
    s = str(gen_output_obj).strip()
    return s, type(gen_output_obj).__name__, True

# =============================================================================
# Candidate selection (strict)
# =============================================================================

def _context_flags(text: str, c: Candidate) -> Dict[str, bool]:
    start, end = c.span_start, c.span_end
    left = max(0, start - 60); right = min(len(text), end + 60)
    snippet = text[left:right]
    c.context_snippet = snippet
    near_keyword = bool(KEYWORD_RE.search(snippet))
    radius_ctx = bool(re.search(r"\bradius\b|\bR\s*=", snippet, flags=re.IGNORECASE))
    c.flags.update({"near_keyword": near_keyword, "radius_context": radius_ctx})
    return c.flags

def _is_banned_constant(c: Candidate) -> bool:
    for k in BAN_CONSTANTS:
        if abs(c.raw_value - k) < 1e-6:
            return True
    return False

def select_best_candidate(text: str,
                          candidates: List[Candidate],
                          expected_km: float,
                          require_keywords: bool = True,
                          prefer_tail: bool = True,
                          allow_unit_rescue: bool = True) -> Tuple[Optional[Candidate], Optional[str], Optional[float], float]:
    """
    Returns (best_candidate, assumed_unit, parsed_km, rel_error)
    """
    if not candidates:
        return None, None, None, float("inf")

    L = len(text)
    tail_cut = int(L * 0.65) if L > 300 else 0  # prefer last ~35% of text

    filtered: List[Candidate] = []
    for c in candidates:
        _context_flags(text, c)
        # Discard constants if in radius context
        if _is_banned_constant(c) and c.flags.get("radius_context", False):
            c.flags["constant_filtered"] = True
            continue
        if require_keywords and not c.flags.get("near_keyword", False):
            c.flags["keyword_filtered"] = True
            continue
        filtered.append(c)

    pool = filtered if filtered else candidates

    best = None
    best_km = None
    best_unit = None
    best_score = float("inf")
    best_rel = float("inf")

    for c in pool:
        explicit_unit = c.unit_norm is not None
        options: List[Tuple[str, float]] = []

        if explicit_unit:
            # respect explicit unit; do not reinterpret
            km_val = to_km(c.raw_value, c.unit_norm)
            options = [(c.unit_norm, km_val)]
        else:
            if allow_unit_rescue:
                options = [
                    ("km", c.raw_value),
                    ("mi", to_km(c.raw_value, "mi")),
                    ("nmi", to_km(c.raw_value, "nmi")),
                ]
            else:
                options = [("km", c.raw_value)]

        for u, km_val in options:
            rel = abs(km_val - expected_km) / expected_km if expected_km > 0 else float("inf")
            penalty = 0.0

            # prefer tail
            if prefer_tail and c.span_start < tail_cut:
                penalty += 0.15
            # penalize constants without strong keyword context
            if _is_banned_constant(c) and not c.flags.get("near_keyword", False):
                penalty += 2.0

            score = rel + penalty
            if score < best_score:
                best_score = score
                best = c
                best_km = km_val
                best_unit = u
                best_rel = rel

    if best is not None:
        best.chosen_unit = best_unit
        best.parsed_km = best_km
    return best, best_unit, best_km, best_rel

# =============================================================================
# Scoring buckets
# =============================================================================

def bucket_points(expected_km: float, pred_km: Optional[float]) -> Tuple[str, int, float]:
    if expected_km <= 0 or pred_km is None:
        return ("wrong", 0, float("inf"))
    rel = abs(pred_km - expected_km) / expected_km
    if rel < 0.01: return ("perfect", 10, rel)
    if rel < 0.05: return ("approx", 3, rel)
    if rel < 0.10: return ("pretty_close", 1, rel)
    return ("wrong", 0, rel)

# =============================================================================
# Multi‑GPU helpers (Accelerate via device_map='auto')
# =============================================================================

def have_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False

def visible_gpus() -> List[int]:
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []

def max_memory_map(reserve_gib: int = 2) -> Dict[Any, str]:
    mem = {}
    try:
        import torch, psutil
        for i in visible_gpus():
            total = torch.cuda.get_device_properties(i).total_memory
            total_gib = int(total / (1024**3))
            allow = max(1, total_gib - reserve_gib)
            mem[i] = f"{allow}GiB"
        vm = psutil.virtual_memory()
        cpu_allow = max(2, int(vm.available / (1024**3)) - 2)
        mem["cpu"] = f"{cpu_allow}GiB"
    except Exception:
        for i in visible_gpus():
            mem[i] = "20GiB"
        mem["cpu"] = "64GiB"
    return mem

# =============================================================================
# Model loader (tokenizer with left padding; pipeline; sharding)
# =============================================================================

def load_generator(model_name: str,
                   multi_gpu: str = "auto-shard",
                   reserve_gib: int = 2,
                   device_map_override: Optional[str] = None):
    """
    - Builds a tokenizer with left padding and a pad token (EOS for inference if needed).
    - Loads a text-generation pipeline; passes model_kwargs for device_map='auto' and max_memory.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure decoder-only friendly padding for *batched* generation (left-pad)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # inference-only fallback

    model_kwargs = {}
    if multi_gpu == "auto-shard" and have_cuda() and len(visible_gpus()) >= 1:
        dm = device_map_override or "auto"
        model_kwargs.update({
            "device_map": dm,
            "max_memory": max_memory_map(reserve_gib),
            "low_cpu_mem_usage": True
        })

    gen = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tok,
        dtype="auto",
        device_map=None,          # device placement handled via model_kwargs (Accelerate)
        model_kwargs=model_kwargs
    )
    return gen, tok, model_kwargs

# =============================================================================
# Messages (Harmony)
# =============================================================================

def default_system_meta() -> str:
    return (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        "Current date: 2025-09-09\n\n"
        "Reasoning: medium\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )

def build_messages(system_meta: Optional[str], developer_instr: str, user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_meta:
        msgs.append({"role": "system", "content": system_meta})
    msgs.append({"role": "developer", "content": developer_instr})
    msgs.append({"role": "user", "content": user_text})
    return msgs

# =============================================================================
# Decoding config (+ sanitize flags)
# =============================================================================

@dataclass
class DecodeConfig:
    decoding: str
    do_sample: Optional[bool]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    repetition_penalty: Optional[float]
    seed: Optional[int]

def make_decode_kwargs(cfg: DecodeConfig) -> Dict[str, Any]:
    if cfg.decoding == "recommended":
        return dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    if cfg.decoding == "deterministic":
        return dict(do_sample=False)  # greedy
    kwargs: Dict[str, Any] = {}
    if cfg.do_sample is not None: kwargs["do_sample"] = cfg.do_sample
    if cfg.temperature is not None: kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None: kwargs["top_p"] = cfg.top_p
    if cfg.top_k is not None: kwargs["top_k"] = cfg.top_k
    if cfg.repetition_penalty is not None: kwargs["repetition_penalty"] = cfg.repetition_penalty
    return kwargs

def sanitize_decode_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    If not sampling, remove sampling-only flags to avoid warnings:
      'temperature', 'top_p', 'top_k', 'typical_p'.
    """
    # If sampling is explicitly off OR unspecified (defaults to greedy), drop sampling flags
    if kwargs.get("do_sample") is False or "do_sample" not in kwargs:
        for k in ("temperature", "top_p", "top_k", "typical_p"):
            kwargs.pop(k, None)
    return kwargs

# =============================================================================
# Pipeline output unwrapping
# =============================================================================

def safe_get_generated_text(pipeline_item: Any) -> Any:
    if isinstance(pipeline_item, dict):
        return pipeline_item.get("generated_text", pipeline_item)
    if isinstance(pipeline_item, list):
        if pipeline_item and isinstance(pipeline_item[0], dict):
            return pipeline_item[0].get("generated_text", pipeline_item[0])
        return pipeline_item
    return pipeline_item

def extract_final_text_from_pipeline(gen_output_obj: Any) -> Tuple[str, str, bool]:
    return extract_final_text_from_pipeline_output(gen_output_obj)

# =============================================================================
# Eval loop
# =============================================================================

def run_eval(eval_file: Path, model_name: str, out_parquet: Path, out_summary: Path,
             max_new_tokens: int = 512, temperature: float = 0.0, limit: Optional[int] = None,
             batch_size: int = 4, hf_batch_size: int = 4, verbose: bool = False, print_limit: Optional[int] = None,
             decoding: str = "recommended", top_p: Optional[float] = None, top_k: Optional[int] = None,
             do_sample: Optional[bool] = None, repetition_penalty: Optional[float] = None,
             seed: Optional[int] = None, multi_gpu: str = "auto-shard", reserve_gib: int = 2,
             require_final: bool = True, allow_analysis_fallback: bool = False,
             allow_unit_rescue: bool = True) -> Dict:

    df = pq.read_table(eval_file).to_pandas()
    if limit is not None:
        df = df.head(limit).copy()

    system_meta = default_system_meta()
    if seed is not None:
        set_seed(seed)

    cfg = DecodeConfig(
        decoding=decoding,
        do_sample=do_sample,
        temperature=temperature if decoding == "custom" else None,
        top_p=top_p if decoding == "custom" else None,
        top_k=top_k if decoding == "custom" else None,
        repetition_penalty=repetition_penalty,
        seed=seed
    )
    gen_kwargs = sanitize_decode_kwargs(make_decode_kwargs(cfg))
    gen, tok, model_kwargs = load_generator(model_name, multi_gpu=multi_gpu, reserve_gib=reserve_gib)

    log(f"Loaded {len(df)} eval rows from {eval_file}")
    log(f"Model: {model_name}")
    log(f"Decoding: {decoding} (kwargs: {gen_kwargs})")
    log(f"Multi-GPU: {multi_gpu} (model_kwargs: {model_kwargs})")
    log(f"Batches: eval={batch_size}, hf_internal={hf_batch_size}")
    log(f"Tokenizer padding_side={tok.padding_side} pad_token_id={tok.pad_token_id}")
    log(hline())

    n = len(df)
    results = []
    buckets = {"perfect": 0, "approx": 0, "pretty_close": 0, "wrong": 0}
    total_points = 0
    printed = 0

    def make_batch(start: int, end: int):
        msgs_batch, rows_batch = [], []
        for i in range(start, end):
            row = df.iloc[i]
            msgs_batch.append(build_messages(system_meta, row["developer"], row["user"]))
            rows_batch.append((i, row))
        return msgs_batch, rows_batch

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        msgs_batch, rows_batch = make_batch(start, end)

        t0 = time.time()
        outs = gen(
            msgs_batch,
            max_new_tokens=max_new_tokens,
            batch_size=hf_batch_size,
            pad_token_id=tok.pad_token_id,  # ensure correct padding at generate() time
            **gen_kwargs
        )
        dt = time.time() - t0

        for out_item, (i, row) in zip(outs, rows_batch):
            expected_km = float(row["expected_km"])

            gen_payload = safe_get_generated_text(out_item)
            final_text, shape_str, looks_analysis = extract_final_text_from_pipeline(gen_payload)

            analysis_fallback_used = False
            if looks_analysis and require_final and not allow_analysis_fallback:
                best = None; best_km = None; best_rel = float("inf"); best_unit = None
            else:
                if looks_analysis:
                    analysis_fallback_used = True
                cands = extract_candidates(final_text)
                best, best_unit, best_km, best_rel = select_best_candidate(
                    final_text, cands, expected_km,
                    require_keywords=True,
                    prefer_tail=True,
                    allow_unit_rescue=allow_unit_rescue
                )

            bucket, pts, rel = bucket_points(expected_km, best_km)

            results.append({
                "idx": int(i),
                "developer": row["developer"],
                "user": row["user"],
                "model_output_shape": shape_str,
                "model_output_final": final_text,
                "expected_km": expected_km,
                "parsed_value": (None if best is None else best.raw_value),
                "parsed_unit_raw": (None if best is None else best.unit_raw),
                "parsed_unit_norm": (None if best is None else best.unit_norm),
                "assumed_unit_for_match": (None if best is None else best.chosen_unit),
                "parsed_km": (None if best is None else best.parsed_km),
                "rel_error": rel,
                "bucket": bucket,
                "points": pts,
                "analysis_fallback_used": analysis_fallback_used,
                "parser_flags": (None if best is None else best.flags),
                "parser_context": (None if best is None else best.context_snippet)
            })
            if bucket in buckets: buckets[bucket] += 1
            total_points += pts

            # -------- Verbose printing --------
            if verbose and (print_limit is None or printed < print_limit):
                printed += 1
                log(f"\n{hline('=')}\nSAMPLE {i+1}/{n}  (batch {start}-{end-1})  time={dt:.3f}s")
                # Messages
                log(hline()); log(">> MESSAGES SENT")
                msgs = build_messages(system_meta, row["developer"], row["user"])
                for m in msgs:
                    role = m["role"]; content = (m["content"][:500] + "…") if len(m["content"]) > 500 else m["content"]
                    log(f"[{role.upper()}] {content}")
                # Extracted final
                log(hline()); log(">> EXTRACTED FINAL TEXT")
                preview = (final_text[:1000] + "…") if len(final_text) > 1000 else final_text
                log(preview)
                # Candidates
                log(hline()); log(">> NUMERIC CANDIDATES (value [unit_raw] -> km/mi/nmi conversions)")
                cand_list = extract_candidates(final_text)
                if looks_analysis:
                    log("(Note: looks like analysis; final channel missing)")
                if not cand_list:
                    log("(none)")
                else:
                    for c in cand_list[:50]:  # cap print
                        km_map = f"km:{to_km(c.raw_value,'km'):.6f}, mi:{to_km(c.raw_value,'mi'):.6f}, nmi:{to_km(c.raw_value,'nmi'):.6f}"
                        log(f"  {c.raw_value}  [{c.unit_raw or '—'}]  ->  {km_map}")
                # Decision
                log(hline()); log(">> DECISION & SCORE")
                log(f"expected_km={expected_km:.6f}")
                if best is None:
                    log("parsed: (none) -> bucket=wrong, points=0")
                else:
                    log(f"parsed_raw: {best.raw_value} [{best.unit_raw or '—'}]  explicit_norm={best.unit_norm or '—'}")
                    log(f"assumed_unit={best.chosen_unit}  parsed_km={best.parsed_km:.6f}  rel_error={best_rel*100:.2f}%")
                    log(f"flags={best.flags}  analysis_fallback_used={analysis_fallback_used}")
                    ctx = best.context_snippet or ""
                    log("context: ..." + ctx.replace("\n"," ")[:200] + "...")
                    log(f"bucket={bucket}  points={pts}")
                log(hline())

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
    Path(out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strict multi‑GPU evaluation for gpt-oss (Transformers pipeline).")
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--out", type=str, default="eval_results.parquet")
    parser.add_argument("--summary", type=str, default="eval_summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0, help="Used only when --decoding custom")
    parser.add_argument("--top-p", type=float, default=None, help="Used only when --decoding custom")
    parser.add_argument("--top-k", type=int, default=None, help="Used only when --decoding custom")
    parser.add_argument("--do-sample", type=lambda x: str(x).lower() in ("1","true","yes"), default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4, help="Requests per pipeline call")
    parser.add_argument("--hf-batch-size", type=int, default=4, help="Pipeline internal micro-batch per call")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-limit", type=int, default=10)
    parser.add_argument("--decoding", choices=["recommended","deterministic","custom"], default="recommended",
                        help="recommended=temp=1.0,top_p=1.0,top_k=0; deterministic=greedy; custom=flags")
    parser.add_argument("--multi-gpu", choices=["auto-shard","single"], default="auto-shard")
    parser.add_argument("--reserve-gib", type=int, default=2)
    parser.add_argument("--require-final", type=lambda x: str(x).lower() in ("1","true","yes"), default=True,
                        help="Only grade if a Harmony 'final' message is present.")
    parser.add_argument("--allow-analysis-fallback", type=lambda x: str(x).lower() in ("1","true","yes"), default=False,
                        help="If no final, allow parsing from analysis-like text.")
    parser.add_argument("--allow-unit-rescue", type=lambda x: str(x).lower() in ("1","true","yes"), default=True,
                        help="If a number has no unit, try km/mi/nmi assumptions; explicit units are never reinterpreted.")
    parser.add_argument("--log", type=str, default=None,
                        help="If set, tee all console output to this .txt file (e.g., run_log.txt)")
    args = parser.parse_args()

    # init tee logging if requested
    init_log_file(Path(args.log) if args.log else None)

    summary = run_eval(
        eval_file=Path(args.eval_file),
        model_name=args.model,
        out_parquet=Path(args.out),
        out_summary=Path(args.summary),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        batch_size=args.batch_size,
        hf_batch_size=args.hf_batch_size,
        verbose=args.verbose,
        print_limit=args.print_limit,
        decoding=args.decoding,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        multi_gpu=args.multi_gpu,
        reserve_gib=args.reserve_gib,
        require_final=args.require_final,
        allow_analysis_fallback=args.allow_analysis_fallback,
        allow_unit_rescue=args.allow_unit_rescue
    )
    log(json.dumps(summary, indent=2))

    # close log file if used
    if LOG_FH is not None:
        LOG_FH.close()

if __name__ == "__main__":
    main()

