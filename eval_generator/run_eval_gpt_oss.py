#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_gpt_oss.py
-------------------
Evaluate a GPT‑OSS model with Transformers on a GPS distance reasoning eval set.

SCORING (explicit & logged)
---------------------------
1) We parse the model output, preferring a Harmony “final” channel if present.
   If absent, we fall back to alternatives (legacy markers or a trailing 'Final: ...').
2) We extract numeric candidates (robust to thousands separators like “26,300”
   or “9 330”) plus nearby units. Each candidate gets a heuristic score:
     +5 if it has an explicit normalized unit (km/mi/nmi/klick/etc.)
     +3 if near distance keywords (“distance”, “final”, “answer”, “result”, “≈”, “~”)
     +2 if near “final-ish” words (“about”, “roughly”, “≈”, “result”)
     +2 tail-bias (appears towards the end of the message)
     −6 near latitude/longitude/DMS markers (°, deg, ′, ″, min, sec)
     −6 near Earth-radius constants (“R=6371”, “mean radius”, “6371.0088”)
     −4 near math-only markers (π, “rad”, “radian”)
     −1 tiny magnitude (< 5) to downweight incidental intermediates
   A small optional “fit bonus” uses ground-truth to gently favor candidates
   that are near the expected distance (off by <10%: +1; <5%: +2; <1%: +3).
3) We rank by (score desc, position desc, magnitude desc) and pick the winner.
   If it lacks an explicit unit, we:
     (a) prefer a unit hinted by the user request (e.g., “sea miles/nm/nmi” → nmi,
         “klicks”/“clicks” → km), else
     (b) choose among km/mi/nmi the interpretation closest to a robust reference
         magnitude computed from the top few candidates.
4) Relative error vs ground-truth kilometers:
      rel = |pred_km − expected_km| / expected_km
   Buckets:
      < 1%       → "perfect"       (10 points)
      < 5%       → "approx"         (3 points)
      < 10%      → "pretty_close"   (1 point)
      ≥ 10%      → "wrong"          (0 points)
   We report bucket counts, total points, and points per sample.

Generation & model notes
------------------------
- We render prompts with the model’s **chat template** via
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)` and feed
  the resulting strings to the text-generation pipeline. This is the recommended
  way to run GPT‑OSS with Transformers; it ensures proper Harmony markers are used.  # noqa
  (See OpenAI Cookbook + HF chat templating docs + GPT‑OSS model card)
- Decoder-only left padding is enabled; `pad_token_id` is set to `eos_token_id`
  where needed to avoid warnings.
- Reasoning effort: `--reasoning none|low|medium|high`. Higher effort yields
  longer analysis; if `max_new_tokens` is too small, the model may never reach
  the `final` channel. Default is `medium`.  # noqa

References
----------
- Harmony format (OpenAI Cookbook):           https://cookbook.openai.com/articles/openai-harmony
- Running GPT‑OSS with Transformers (Cookbook): https://cookbook.openai.com/articles/gpt-oss/run-transformers
- GPT‑OSS 20B model card (HF):                https://huggingface.co/openai/gpt-oss-20b
- Transformers chat templating docs (HF):     https://huggingface.co/docs/transformers/main/en/chat_templating
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Units & normalization
# -----------------------------
UNIT_SYNONYMS = {
    "km": {
        "km","kms","k","kilometer","kilometers","kilometre","kilometres",
        "km.","klick","klicks","click","clicks"
    },
    "mi": {"mi","mile","miles","statute","statute miles","sm"},
    "nmi": {
        "nmi","nm","n.m.","nautical","nauticalmile","nauticalmiles",
        "sea","sea mile","sea miles","seamile","seamiles"
    },
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
    return None

def unit_preference_from_user(user_text: str) -> Optional[str]:
    u = user_text.lower()
    if any(t in u for t in ["sea mile","sea miles","nmi","nm","nautical"]): return "nmi"
    if any(t in u for t in ["klick","klicks","click","clicks"]): return "km"
    if any(t in u for t in [" mile"," miles","(mi","statute"]): return "mi"
    if any(t in u for t in [" km","kilometer","kilometre"]): return "km"
    return None

# -----------------------------
# Numeric parsing (robust)
# -----------------------------
_NUM_RE = re.compile(
    r"(?P<num>[+-]?\d[\d\s\u00A0\u202F,\.]*\d|\d(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>klicks?|clicks?|km(?:s)?|kilometer(?:s)?|kilometre(?:s)?|mi\.?|mile(?:s)?|statute(?:\s+miles?)?|sm|"
    r"nmi|nm|n\.m\.|nautical(?:\s+miles?)?|sea(?:\s+miles?)?))?",
    flags=re.IGNORECASE
)

DEG_MARKERS     = ["°"," deg","deg ","degrees","′","'","″","\"","min","sec"]
RADIUS_MARKERS  = ["r=6371","r = 6371","earth radius","mean radius","6371.0088","6371 km","r≈6371","3437.74677"]
MATH_MARKERS    = ["π","pi","rad","radian"]
DISTANCE_TOKENS = ["distance","final","answer","result","≈","~","about","roughly","km","mi","nmi","nautical","sea mile","statute","klick","click"]

def _parse_float_locale(num_str: str) -> Optional[float]:
    """Parse floats with mixed thousands/decimal separators and NBSPs."""
    s = (num_str
         .replace("\u00A0"," ")
         .replace("\u202F"," ")
         .strip())
    # Some locales use apostrophes as thousands separators
    s = s.replace("’","").replace("'","")
    has_dot = "." in s
    has_com = "," in s

    def only_thousands_sep(txt: str, sep: str) -> bool:
        pattern = rf"^[+-]?\d{{1,3}}(?:\{sep}\d{{3}})+$"
        return re.fullmatch(pattern, txt.replace(" ","")) is not None

    if has_dot and has_com:
        last_dot = s.rfind("."); last_com = s.rfind(",")
        if last_dot > last_com:
            s_clean = s.replace(",","").replace(" ","")
        else:
            s_clean = s.replace(".","").replace(" ","")
            s_clean = s_clean.replace(",",".")
    elif has_com and not has_dot:
        if only_thousands_sep(s,","):
            s_clean = s.replace(",","").replace(" ","")
        else:
            s_clean = s.replace(" ","").replace(",",".")
    elif has_dot and not has_com:
        if only_thousands_sep(s,"."):
            s_clean = s.replace(".","").replace(" ","")
        else:
            s_clean = s.replace(" ","")
    else:
        s_clean = s.replace(" ","")

    s_clean = re.sub(r"[^0-9eE\.\+\-]", "", s_clean)
    if not s_clean: return None
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
    score_detail: Dict[str,int]
    km_interps: Dict[str,float]

def _near_any(text: str, idx: int, window: int, needles: Sequence[str]) -> bool:
    lo = max(0, idx - window); hi = min(len(text), idx + window)
    seg = text[lo:hi].lower()
    return any(n in seg for n in needles)

def extract_candidates_with_context(text: str, tail_bias_chars: int = 1200,
                                    expected_km: Optional[float] = None,
                                    fit_bonus: bool = True) -> List[Candidate]:
    out: List[Candidate] = []
    t = text
    for m in _NUM_RE.finditer(t):
        num_s = m.group("num"); unit_s = m.group("unit")
        val = _parse_float_locale(num_s)
        if val is None: continue
        start, end = m.span()
        lo = max(0, start-40); hi = min(len(t), end+40)
        ctx = t[lo:hi]
        unit_norm = normalize_unit(unit_s) if unit_s else None

        sd: Dict[str,int] = {}
        score = 0
        if unit_norm:
            sd["unit_bonus"]=5; score += 5
        if _near_any(t,start,80,DISTANCE_TOKENS):
            sd["near_keyword"]=sd.get("near_keyword",0)+3; score += 3
        if _near_any(t,start,40,["≈","~","about","roughly","final","answer","result"]):
            sd["finalish"]=sd.get("finalish",0)+2; score += 2
        if start > len(t)-tail_bias_chars:
            sd["tail_bias"]=2; score += 2
        if _near_any(t,start,20,DEG_MARKERS):
            sd["dms_penalty"]=-6; score -= 6
        if _near_any(t,start,30,MATH_MARKERS):
            sd["math_penalty"]=-4; score -= 4
        if _near_any(t,start,40,RADIUS_MARKERS):
            sd["radius_penalty"]=-6; score -= 6
        if val < 5.0:
            sd["tiny_penalty"]=-1; score -= 1

        # Plausibility gate (0..~20,500 km) → soft penalty if implausible
        plausible = False
        for u in ("km","mi","nmi"):
            km_v = to_km(val,u)
            if 0.0 <= km_v <= 20500.0:
                plausible = True; break
        if not plausible:
            sd["plausibility_penalty"] = -4; score -= 4

        km_interps = {"km":to_km(val,"km"), "mi":to_km(val,"mi"), "nmi":to_km(val,"nmi")}
        # Small “fit bonus” against ground-truth to avoid picking constants
        if fit_bonus and expected_km and expected_km > 0:
            best_rel = min(abs(km_interps[u]-expected_km)/expected_km for u in km_interps)
            if   best_rel < 0.01: sd["fit_bonus"]=sd.get("fit_bonus",0)+3; score += 3
            elif best_rel < 0.05: sd["fit_bonus"]=sd.get("fit_bonus",0)+2; score += 2
            elif best_rel < 0.10: sd["fit_bonus"]=sd.get("fit_bonus",0)+1; score += 1

        out.append(Candidate(
            raw_value=val, unit_raw=unit_s, unit_norm=unit_norm,
            start=start, end=end, context=ctx, score=score, score_detail=sd,
            km_interps=km_interps
        ))

    # Deduplicate exact duplicates
    dedup: Dict[Tuple[float,Optional[str],int,int], Candidate] = {}
    for c in out:
        key = (round(c.raw_value,12), c.unit_norm, c.start, c.end)
        if key not in dedup: dedup[key]=c
    return list(dedup.values())

# -----------------------------
# Harmony extraction (robust)
# -----------------------------
# Canonical Harmony markers
CHAN_FINAL   = "<|channel|>final<|message|>"
CHAN_ANALYSIS= "<|channel|>analysis<|message|>"
TAG_END      = "<|end|>"
# Legacy-ish variants encountered in the wild
LEGACY_ASSISTANT_FINAL    = re.compile(r"<\|start\|>\s*assistantfinal(?:<\|message\|>)?", re.IGNORECASE)
LEGACY_ASSISTANT_ANALYSIS = re.compile(r"<\|start\|>\s*assistantanalysis(?:<\|message\|>)?", re.IGNORECASE)

def _to_text(obj: Any) -> str:
    if isinstance(obj,str): return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def extract_final_and_analysis(full_obj: Any) -> Tuple[str, Optional[str]]:
    """Return (final_text, analysis_text) from a Harmony/legacy output."""
    full = _to_text(full_obj)

    # 1) Canonical Harmony: <|channel|>final<|message|> ... <|end|>
    if CHAN_FINAL in full:
        fin = full.split(CHAN_FINAL,1)[1]
        fin = fin.split("<|channel|>",1)[0]
        fin = fin.split(TAG_END,1)[0].strip()
        ana = None
        if CHAN_ANALYSIS in full:
            ana = full.split(CHAN_ANALYSIS,1)[1]
            ana = ana.split("<|channel|>",1)[0]
            ana = ana.split(TAG_END,1)[0].strip()
        return fin, ana

    # 2) Legacy: assistantfinal ... ; assistantanalysis ...
    if LEGACY_ASSISTANT_FINAL.search(full):
        start = LEGACY_ASSISTANT_FINAL.search(full).end()
        tail  = full[start:]
        fin = tail.split("<|start|>",1)[0]
        fin = fin.split(TAG_END,1)[0].strip()
        ana = None
        m = LEGACY_ASSISTANT_ANALYSIS.search(full)
        if m:
            astart = m.end()
            atail  = full[astart:]
            ana = atail.split("<|start|>",1)[0]
            ana = ana.split(TAG_END,1)[0].strip()
        return fin, ana

    # 3) HTML-ish tags
    m_final = re.search(r"<final>(.*?)</final>", full, flags=re.DOTALL|re.IGNORECASE)
    if m_final:
        fin = m_final.group(1).strip()
        m_ana = re.search(r"<analysis>(.*?)</analysis>", full, flags=re.DOTALL|re.IGNORECASE)
        return fin, (m_ana.group(1).strip() if m_ana else None)

    # 4) Trailing 'Final:' line as last resort
    m = re.search(r"(Final\s*:\s*.+)$", full, flags=re.IGNORECASE|re.DOTALL)
    if m:
        fin = m.group(1).strip()
        ana = full[:m.start()].strip() or None
        return fin, ana

    # 5) Fallback: whole text as final
    return full.strip(), None

# -----------------------------
# Bucketing
# -----------------------------
def bucket_points(expected_km: float, pred_km: Optional[float]) -> Tuple[str,int,float]:
    if expected_km <= 0 or pred_km is None:
        return ("wrong",0,float("inf"))
    rel = abs(pred_km-expected_km)/expected_km
    if rel < 0.01: return ("perfect",10,rel)
    if rel < 0.05: return ("approx",3,rel)
    if rel < 0.10: return ("pretty_close",1,rel)
    return ("wrong",0,rel)

# -----------------------------
# Prompt building
# -----------------------------
DEFAULT_DEV_SYSTEM = "you are an agent that understands GPS and can calculate distances"

def build_harmony_messages(system_text: str, user_text: str,
                           reasoning: str = "medium",
                           final_hint: bool = True) -> List[Dict[str,str]]:
    """
    Build Harmony messages. We pass a clean 'developer' system and a user turn.
    Optionally inject a short 'Reasoning: <level>' hint and a final-line contract.
    """
    sys_lines = []
    if reasoning.lower() in {"low","medium","high"}:
        sys_lines.append(f"Reasoning: {reasoning.lower()}")
    sys_lines.append("Valid channels: analysis, final.")
    if final_hint:
        sys_lines.append("In the final channel, finish with: Final: <number> <unit>")

    developer = (system_text or DEFAULT_DEV_SYSTEM).strip()
    developer = developer + ("\n" + "\n".join(sys_lines) if sys_lines else "")

    return [
        {"role": "developer", "content": developer},
        {"role": "user", "content": user_text.strip()},
    ]

def build_naive_prompt(system_text: str, user_text: str) -> str:
    """
    Naive (pre-Harmony) render with a strong final-line contract.
    """
    contract = (
        "You are a careful assistant. Compute the distance between the two coordinates. "
        "Use haversine/Vincenty as appropriate. End with a single line:\n"
        "Final: <NUMBER> <UNIT>\n"
    )
    return f"[SYSTEM]\n{contract}\n{system_text.strip()}\n\n[USER]\n{user_text.strip()}\n\n[ASSISTANT]\n"

# -----------------------------
# Model / tokenizer loading
# -----------------------------
def load_pipeline(model_name: str,
                  device_map: Optional[str] = "auto"):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map or "auto",
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    gen = pipeline("text-generation", model=model, tokenizer=tok)
    return gen, tok

def render_prompts(tok, rows: List[Tuple[str,str]], chat_format: str, reasoning: str) -> List[str]:
    prompts = []
    for system_text, user_text in rows:
        if chat_format == "harmony":
            messages = build_harmony_messages(system_text, user_text, reasoning=reasoning, final_hint=True)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        elif chat_format == "naive":
            prompts.append(build_naive_prompt(system_text, user_text))
        else:
            # default to harmony if unknown
            messages = build_harmony_messages(system_text, user_text, reasoning=reasoning, final_hint=True)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
    return prompts

# -----------------------------
# Generation (batched)
# -----------------------------
def generate_batched(gen, prompts: List[str], max_new_tokens: int,
                     decoding: str, temperature: float,
                     top_k: Optional[int], top_p: Optional[float],
                     hf_batch_size: int) -> Tuple[List[str], Dict]:
    gen_kwargs: Dict[str,Any] = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": True,
        "num_return_sequences": 1,
        "batch_size": hf_batch_size,
        "do_sample": decoding == "sampling",
    }
    if decoding == "sampling":
        gen_kwargs["temperature"] = float(temperature)
        if top_k is not None: gen_kwargs["top_k"] = int(top_k)
        if top_p is not None: gen_kwargs["top_p"] = float(top_p)

    outs = gen(prompts, **gen_kwargs)
    texts: List[str] = []
    for item in outs:
        if isinstance(item, dict) and "generated_text" in item:
            texts.append(item["generated_text"])
        else:
            texts.append(_to_text(item))
    return texts, gen_kwargs

# -----------------------------
# Candidate selection
# -----------------------------
def choose_candidate(final_text: str, analysis_text: Optional[str], user_text: str,
                     expected_km: float, logger: TeeLogger,
                     print_candidates: int = 12) -> Tuple[Optional[Candidate], Optional[str], Optional[float], bool]:
    unit_pref = unit_preference_from_user(user_text)
    used_analysis = False

    def _rank(text: str) -> Tuple[Optional[Candidate], Optional[str], Optional[float], List[Candidate]]:
        cands = extract_candidates_with_context(text, expected_km=expected_km, fit_bonus=True)
        if not cands:
            return None, None, None, []
        cands_sorted = sorted(cands, key=lambda c: (c.score, c.start, c.raw_value), reverse=True)
        if print_candidates > 0:
            logger.write(">> NUMERIC CANDIDATES (value [unit] → km/mi/nmi | score)\n")
            for c in cands_sorted[:print_candidates]:
                logger.write(f"  {c.raw_value}  [{'—' if c.unit_raw is None else c.unit_raw}]  "
                             f"→ km:{c.km_interps['km']:.3f}, mi:{c.km_interps['mi']:.3f}, nmi:{c.km_interps['nmi']:.3f}  "
                             f"| score={c.score} {c.score_detail}\n")
        top = cands_sorted[0]
        assumed = top.unit_norm
        chosen_km = None
        if top.unit_norm:
            chosen_km = to_km(top.raw_value, top.unit_norm)
            assumed = top.unit_norm
        else:
            if unit_pref in ("km","mi","nmi"):
                chosen_km = to_km(top.raw_value, unit_pref); assumed = unit_pref
            else:
                # pick interpretation whose km is closest to the median of nearby candidates
                pack = cands_sorted[:min(5,len(cands_sorted))]
                pool = [v for c in pack for v in c.km_interps.values()]
                ref = sorted(pool)[len(pool)//2] if pool else None
                best_u, best_d = None, float("inf")
                for u in ("km","mi","nmi"):
                    d = abs(top.km_interps[u] - (ref if ref is not None else top.km_interps[u]))
                    if d < best_d:
                        best_d, best_u = d, u
                        chosen_km = top.km_interps[u]
                assumed = best_u
        return top, assumed, chosen_km, cands_sorted

    cand, assumed_unit, chosen_km, _ = _rank(final_text)

    if cand is None or cand.score <= 0:
        if analysis_text:
            cand2, assumed2, km2, _ = _rank(analysis_text)
            if cand2 is not None and (cand is None or cand2.score > cand.score):
                cand, assumed_unit, chosen_km = cand2, assumed2, km2
                used_analysis = True

    # Decision trace
    if cand is not None:
        logger.write(">> DECISION TRACE (winner)\n")
        logger.write(f"  value={cand.raw_value}  unit_raw={cand.unit_raw!r}  unit_norm={cand.unit_norm!r}\n")
        logger.write(f"  assumed_unit={assumed_unit}  parsed_km={chosen_km}\n")
        logger.write(f"  score={cand.score}  breakdown={cand.score_detail}  span=[{cand.start},{cand.end}]\n")
        logger.write(f"  context=...{cand.context}...\n")
    else:
        logger.write(">> DECISION TRACE: no numeric candidates\n")

    return cand, assumed_unit, chosen_km, used_analysis

# -----------------------------
# Eval loop
# -----------------------------
def run_eval(eval_file: Path, model_name: str, out_parquet: Path, out_summary: Path,
             chat_format: str = "harmony",
             reasoning: str = "medium",
             batch_size: int = 8,
             hf_batch_size: int = 8,
             max_new_tokens: int = 600,
             decoding: str = "deterministic",
             temperature: float = 0.0,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             limit: Optional[int] = None,
             verbose: bool = False,
             log_path: Optional[Path] = None,
             device_map: Optional[str] = "auto",
             print_limit: int = 1200,
             channel_limit: int = -1,
             print_candidates: int = 12) -> Dict:
    logger = TeeLogger(log_path)

    df = pq.read_table(eval_file).to_pandas()
    if limit is not None: df = df.head(limit).copy()

    gen, tok = load_pipeline(model_name, device_map=device_map)

    if verbose:
        logger.write(f"Loaded {len(df)} rows from {eval_file}\n")
        logger.write(f"Model: {model_name}\n")
        logger.write(f"Chat format: {chat_format}  |  Reasoning: {reasoning}\n")
        logger.write(f"Decoding: {decoding}  temp={temperature}  top_k={top_k} top_p={top_p}\n")
        logger.write(f"Tokenizer padding_side={tok.padding_side} pad_token_id={tok.pad_token_id}\n")
        logger.write("─"*80 + "\n")

    results = []
    buckets = {"perfect":0, "approx":0, "pretty_close":0, "wrong":0}
    total_points = 0

    n = len(df); batches = math.ceil(n/batch_size)
    for b in range(batches):
        lo = b*batch_size; hi = min(n, (b+1)*batch_size)
        batch = df.iloc[lo:hi]

        pairs = [(row["developer"], row["user"]) for _,row in batch.iterrows()]
        prompts = render_prompts(tok, pairs, chat_format=chat_format, reasoning=reasoning)

        t0 = time.time()
        gen_texts, used_kwargs = generate_batched(
            gen, prompts, max_new_tokens=max_new_tokens,
            decoding=decoding, temperature=temperature, top_k=top_k, top_p=top_p,
            hf_batch_size=hf_batch_size
        )
        dt = time.time()-t0

        if verbose:
            logger.write(f"\nBATCH {b+1}/{batches} rows {lo}..{hi-1}\n")
            logger.write(f"Generation kwargs: {json.dumps(used_kwargs)}\n")

        for (i,row), full_text in zip(batch.iterrows(), gen_texts):
            system_text = row["developer"]; user_text = row["user"]; expected_km = float(row["expected_km"])
            final_text, analysis_text = extract_final_and_analysis(full_text)

            if verbose:
                logger.write("\n" + "═"*80 + "\n")
                logger.write(f"SAMPLE {i+1}/{n}  (batch {lo}-{hi-1})\n")
                logger.write(">> MESSAGES SENT\n")
                logger.write(f"[DEVELOPER] {system_text}\n")
                logger.write(f"[USER] {user_text}\n")
                logger.write(">> EXTRACTED ANALYSIS TEXT\n")
                if analysis_text:
                    at = analysis_text if channel_limit < 0 else (analysis_text[:channel_limit] + ("…" if len(analysis_text)>channel_limit else ""))
                    logger.write(at + ("\n" if not at.endswith("\n") else ""))
                else:
                    logger.write("(none)\n")
                logger.write(">> EXTRACTED FINAL TEXT\n")
                ft = final_text if channel_limit < 0 else (final_text[:channel_limit] + ("…" if len(final_text)>channel_limit else ""))
                logger.write(ft + ("\n" if not ft.endswith("\n") else ""))

            cand, assumed, pred_km, used_analysis = choose_candidate(
                final_text=final_text, analysis_text=analysis_text, user_text=user_text,
                expected_km=expected_km, logger=logger if verbose else TeeLogger(None),
                print_candidates=print_candidates
            )
            bucket, pts, rel = bucket_points(expected_km, pred_km)

            if verbose:
                logger.write(">> DECISION & SCORE\n")
                logger.write(f"expected_km={expected_km}\n")
                if cand is None:
                    logger.write("parsed: NONE\n")
                else:
                    logger.write(f"parsed_raw: {cand.raw_value} [{'—' if cand.unit_raw is None else cand.unit_raw}]  "
                                 f"explicit_norm={cand.unit_norm if cand.unit_norm else '—'}\n")
                    logger.write(f"assumed_unit={assumed}  parsed_km={pred_km}  rel_error={rel:.2%}\n")
                logger.write(f"bucket={bucket}  points={pts}\n")

            results.append({
                "idx": int(i),
                "developer": system_text,
                "user": user_text,
                "model_output_raw": full_text,
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
            if bucket in buckets: buckets[bucket]+=1
            total_points += pts

        if verbose:
            logger.write(f"-- batch generation time: {dt:.2f}s\n")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(results)), out_parquet, compression="zstd")
    summary = {
        "model": model_name,
        "eval_file": str(eval_file),
        "results_file": str(out_parquet),
        "n": len(results),
        "bucket_counts": buckets,
        "total_points": total_points,
        "points_per_sample": (total_points/len(results)) if results else 0.0
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.write("\n" + "="*80 + "\n")
    logger.write(json.dumps(summary, indent=2) + "\n")
    logger.close()
    return summary

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Run evaluation against GPT‑OSS (Transformers + Harmony).")
    p.add_argument("--eval-file", type=str, required=True)
    p.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    p.add_argument("--out", type=str, default="eval_results.parquet")
    p.add_argument("--summary", type=str, default="eval_summary.json")
    p.add_argument("--chat-format", type=str, choices=["harmony","naive"], default="harmony",
                   help="Harmony chat template (recommended) or naive single-turn prompt")
    p.add_argument("--reasoning", type=str, choices=["none","low","medium","high"], default="medium",
                   help="Injected reasoning hint for Harmony; default 'medium'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hf-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--decoding", type=str, choices=["deterministic","sampling"], default="deterministic")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--print-limit", type=int, default=1200)
    p.add_argument("--channel-limit", type=int, default=-1)
    p.add_argument("--print-candidates", type=int, default=12)
    args = p.parse_args()

    log_path = Path(args.log) if args.log else None
    reasoning = ("none" if args.reasoning=="none" else args.reasoning)

    summary = run_eval(
        eval_file=Path(args.eval_file),
        model_name=args.model,
        out_parquet=Path(args.out),
        out_summary=Path(args.summary),
        chat_format=args.chat_format,
        reasoning=reasoning,
        batch_size=args.batch_size,
        hf_batch_size=args.hf_batch_size,
        max_new_tokens=args.max_new_tokens,
        decoding=args.decoding,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        limit=args.limit,
        verbose=args.verbose,
        log_path=log_path,
        device_map=args.device_map,
        print_limit=args.print_limit,
        channel_limit=args.channel_limit,
        print_candidates=args.print_candidates,
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

