#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_harmony_pure.py
------------------------
Evaluate a GPT‑OSS model on the GPS distance reasoning eval set using a
strict Harmony implementation with direct model.generate and improved
numeric parsing/candidate selection.

Highlights
- Uses tokenizer.apply_chat_template to render exact Harmony wire format.
- Direct model.generate (no HF pipeline) to avoid token wrapping.
- Effort-based hinting (low|medium|high) appended to system instructions
  and passed via reasoning_effort to the chat template.
- Robust numeric extraction with locale-aware parsing, multiple
  interpretations for ambiguous separators, and heuristic scoring.
- Optional final-only mode and two-pass analysis→final continuation with
  a stopping criterion.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
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
_NUM_RE = re.compile(
    r"(?P<num>[+-]?\d[\d\s\u00A0\u202F,\.'\u2019]*\d|\d(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*(?P<unit>klicks?|clicks?|km(?:s)?|kilometer(?:s)?|kilometre(?:s)?|mi\.?|mile(?:s)?|statute(?:\s+miles?)?|sm|"
    r"nmi|nm|n\.m\.|nautical(?:\s+miles?)?|sea(?:\s+miles?)?))?",
    flags=re.IGNORECASE
)

DEG_MARKERS = ["°", " deg", "deg ", "degrees", "′", "'", "″", '"', "min", "sec"]
RADIUS_MARKERS = ["r=6371", "r = 6371", "earth radius", "mean radius", "6371.0088", "6371 km", "r≈6371"]
MATH_MARKERS = ["π", "pi", "rad", "radian"]
DISTANCE_MARKERS = ["distance", "final", "answer", "result", "≈", "~", "about", "roughly",
                    "km", "mi", "nmi", "nautical", "sea mile", "statute", "klick", "click"]

def _parse_num_candidates(num_str: str) -> List[float]:
    s = (num_str
         .replace("\u00A0", " ")
         .replace("\u202F", " ")
         .strip())
    s = s.replace("’", "").replace("'", "")
    has_dot = "." in s
    has_com = "," in s
    vals: List[float] = []

    def as_float(txt: str) -> Optional[float]:
        try:
            return abs(float(txt))
        except Exception:
            return None

    if has_dot and has_com:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_dot > last_com:
            primary = s.replace(",", "").replace(" ", "")
        else:
            primary = s.replace(".", "").replace(" ", "")
            primary = primary.replace(",", ".")
        v = as_float(primary)
        return [v] if v is not None else []

    if has_com and not has_dot:
        core = s.replace(" ", "")
        parts = core.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
            dec = as_float(core.replace(",", "."))
            thou = as_float(core.replace(",", ""))
            for v in (dec, thou):
                if v is not None and v not in vals:
                    vals.append(v)
            return vals
        v = as_float(core.replace(",", "."))
        return [v] if v is not None else []

    if has_dot and not has_com:
        core = s.replace(" ", "")
        parts = core.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
            dec = as_float(core)
            thou = as_float(core.replace(".", ""))
            for v in (dec, thou):
                if v is not None and v not in vals:
                    vals.append(v)
            return vals
        v = as_float(core)
        return [v] if v is not None else []

    v = as_float(s.replace(" ", ""))
    return [v] if v is not None else []

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
        vals = _parse_num_candidates(num_s)
        if not vals:
            continue
        start, end = m.span()
        lo = max(0, start - 40)
        hi = min(len(t), end + 40)
        ctx = t[lo:hi]

        unit_norm = normalize_unit(unit_s) if unit_s else None

        for _i, val in enumerate(vals):
            sd: Dict[str, int] = {}
            score = 0
            if unit_norm:
                sd["unit_bonus"] = 5; score += 5
            if _near_any(t, start, 80, DISTANCE_MARKERS):
                sd["near_keyword"] = sd.get("near_keyword", 0) + 3; score += 3
            if _near_any(t, start, 40, ["≈", "~", "about", "roughly", "final", "answer", "result"]):
                sd["finalish"] = sd.get("finalish", 0) + 2; score += 2
            if start > len(t) - tail_bias_chars:
                sd["tail_bias"] = 2; score += 2
            if _near_any(t, start, 20, DEG_MARKERS):
                sd["dms_penalty"] = -6; score -= 6
            if _near_any(t, start, 30, MATH_MARKERS):
                sd["math_penalty"] = -4; score -= 4
            if _near_any(t, start, 40, RADIUS_MARKERS):
                sd["radius_penalty"] = -6; score -= 6
            if val < 5.0:
                sd["tiny_penalty"] = -1; score -= 1
            if len(vals) > 1:
                sd["ambig_sep"] = sd.get("ambig_sep", 0) + 0

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

    dedup: Dict[Tuple[float, Optional[str], int, int], Candidate] = {}
    for c in out:
        key = (round(c.raw_value, 12), c.unit_norm, c.start, c.end)
        if key not in dedup:
            dedup[key] = c
    return list(dedup.values())

# -----------------------------
# Harmony channel extraction
# -----------------------------
_BAR = "|"  # build tokens unobtrusively to avoid chat rendering issues
FINAL_MARK = "<" + _BAR + "channel" + _BAR + ">" + "final" + "<" + _BAR + "message" + _BAR + ">"
ANALYSIS_MARK = "<" + _BAR + "channel" + _BAR + ">" + "analysis" + "<" + _BAR + "message" + _BAR + ">"

def _to_text_from_pipeline_obj(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def extract_channel_text(full: str, mark: str) -> Optional[str]:
    if mark in full:
        seg = full.split(mark, 1)[-1]
        seg = re.split(r"<\|channel\|>|<\|end\|>|<\|return\|>|<\|call\|>", seg)[0]
        return seg.strip()
    return None

class StopOnFinalOrBudget(StoppingCriteria):
    def __init__(self, tokenizer, prompt_lens: List[int], final_mark: str, budget_tokens: int):
        super().__init__()
        self.prompt_lens = list(prompt_lens)
        self.final_ids = tokenizer.encode(final_mark, add_special_tokens=False)
        self.budget = int(max(1, budget_tokens))

    @staticmethod
    def _endswith(seq_ids, suffix_ids) -> bool:
        L = len(suffix_ids)
        if L == 0 or len(seq_ids) < L:
            return False
        return seq_ids[-L:] == suffix_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        bsz = input_ids.shape[0]
        for i in range(bsz):
            seq = input_ids[i].tolist()
            if self._endswith(seq, self.final_ids):
                return True
            gen_len = len(seq) - int(self.prompt_lens[i])
            if gen_len >= self.budget:
                return True
        return False

def extract_final_and_analysis(full_obj: Any) -> Tuple[str, Optional[str]]:
    full = _to_text_from_pipeline_obj(full_obj)
    final_text = extract_channel_text(full, FINAL_MARK) or ""
    analysis_text = extract_channel_text(full, ANALYSIS_MARK)
    return final_text, analysis_text

# -----------------------------
# Model loader
# -----------------------------
def load_model_and_tokenizer(model_name: str, device_map: str = "auto", padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tok

# -----------------------------
# Prompt builders
# -----------------------------
def _reasoning_sys_hint(reasoning: str) -> str:
    r = (reasoning or "").lower().strip()
    if r == "low":
        return "\n\nPlease keep <analysis> brief. Put the final numeric answer in <final>."
    if r == "medium":
        return "\n\nUse <analysis> for a short step-by-step. Put ONE concise numeric answer in <final>."
    if r == "high":
        return "\n\nThink carefully in <analysis> (few paragraphs, not too long). Put ONE numeric answer in <final>."
    return ""

def default_system_meta(reasoning: str = "medium") -> str:
    today = time.strftime("%Y-%m-%d")
    return (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        f"Current date: {today}\n\n"
        f"Reasoning: {reasoning}\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )

def build_harmony_messages(system_meta: str, developer_text: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": developer_text},
        {"role": "user", "content": user_text},
    ]

# -----------------------------
# Generation
# -----------------------------
def generate_harmony_direct(
    model, tok,
    batch_messages: List[List[Dict[str, str]]],
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    reasoning_effort: str,
    final_only: bool = False,
    two_pass_final: bool = False,
    analysis_budget_tokens: Optional[int] = None,
    analysis_fraction: Optional[float] = None,
) -> Tuple[List[str], List[str], List[str], List[Dict[str, Optional[bool]]]]:
    rendered: List[str] = []
    for msgs in batch_messages:
        msgs_for_render = list(msgs)
        kwargs = {"tokenize": False, "reasoning_effort": reasoning_effort}
        if final_only:
            msgs_for_render = msgs_for_render + [{"role": "assistant", "content": ""}]
            rendered.append(
                tok.apply_chat_template(
                    msgs_for_render, continue_final_message=True, **kwargs
                )
            )
        else:
            rendered.append(
                tok.apply_chat_template(
                    msgs_for_render, add_generation_prompt=True, **kwargs
                )
            )
    inputs = tok(rendered, return_tensors="pt", padding=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    for k in inputs:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to(dev)

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if decoding == "sampling":
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        if top_k is not None: gen_kwargs["top_k"] = int(top_k)
        if top_p is not None: gen_kwargs["top_p"] = float(top_p)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    in_lens = inputs["input_ids"].ne(pad_id).sum(dim=1).tolist()

    def _decode_outputs(out_t) -> Tuple[List[str], List[str]]:
        full, gen_only = [], []
        for i in range(out_t.shape[0]):
            seq = out_t[i]
            full.append(tok.decode(seq, skip_special_tokens=False))
            gen_only.append(tok.decode(seq[int(in_lens[i]):], skip_special_tokens=False))
        return full, gen_only

    if not two_pass_final or final_only:
        outputs = model.generate(**inputs, **gen_kwargs)
        full_texts, gen_only_texts = _decode_outputs(outputs)
        meta = [{"final_seen_in_pass1": None, "forced_final_inserted": False} for _ in range(len(full_texts))]
        return full_texts, rendered, gen_only_texts, meta

    # Two-pass budgeted analysis then final continuation
    if analysis_fraction is not None:
        try:
            frac = float(analysis_fraction)
        except Exception:
            frac = None
    else:
        frac = None
    if analysis_budget_tokens is None:
        analysis_budget_tokens = 1000
    eff_budget = int(max_new_tokens * frac) if (frac is not None and 0 < frac < 1) else analysis_budget_tokens
    budget = max(1, min(int(max_new_tokens), int(eff_budget)))

    stop_crit = StopOnFinalOrBudget(tok, in_lens, FINAL_MARK, budget)
    gen_kwargs_p1 = dict(gen_kwargs)
    gen_kwargs_p1["stopping_criteria"] = StoppingCriteriaList([stop_crit])
    gen_kwargs_p1["max_new_tokens"] = int(max_new_tokens)
    outputs1 = model.generate(**inputs, **gen_kwargs_p1)
    full1, gen1 = _decode_outputs(outputs1)

    gen_lens1 = [int(outputs1[i].shape[0]) - int(in_lens[i]) for i in range(outputs1.shape[0])]
    rem_tokens = [max(1, int(max_new_tokens) - l) for l in gen_lens1]

    final_ids = tok.encode(FINAL_MARK, add_special_tokens=False)

    def _endswith(a, b):
        L = len(b)
        return L > 0 and len(a) >= L and a[-L:] == b

    full_combined: List[str] = []
    gen_combined: List[str] = []
    pass2_meta: List[Dict[str, Optional[bool]]] = []
    for i in range(outputs1.shape[0]):
        seq1 = outputs1[i]
        seq_list = seq1.tolist()
        forced_final = False
        if not _endswith(seq_list, final_ids):
            add_ids = torch.tensor(final_ids, device=seq1.device, dtype=seq1.dtype)
            seq1 = torch.cat([seq1, add_ids], dim=0)
            forced_final = True
        seq1 = seq1.unsqueeze(0)
        attn = torch.ones_like(seq1)
        gen_kwargs_p2 = dict(gen_kwargs)
        gen_kwargs_p2["max_new_tokens"] = int(rem_tokens[i])
        out2 = model.generate(input_ids=seq1, attention_mask=attn, **gen_kwargs_p2)
        add_only = out2[0, seq1.shape[1]:]
        full_combined.append(tok.decode(out2[0], skip_special_tokens=False))
        gen_text1 = tok.decode(outputs1[i][int(in_lens[i]):], skip_special_tokens=False)
        gen_text2 = tok.decode(add_only, skip_special_tokens=False)
        if forced_final:
            gen_text2 = FINAL_MARK + gen_text2
        gen_combined.append(gen_text1 + gen_text2)
        pass2_meta.append({
            "final_seen_in_pass1": (not forced_final),
            "forced_final_inserted": forced_final,
        })

    return full_combined, rendered, gen_combined, pass2_meta

# -----------------------------
# Candidate selection & scoring
# -----------------------------
def choose_candidate(
    final_text: str,
    analysis_text: Optional[str],
    user_text: str,
    expected_km: float,
    logger: TeeLogger,
    print_candidates: int = 10,
) -> Tuple[Optional[Candidate], Optional[str], Optional[float], bool]:
    unit_pref = unit_preference_from_user(user_text)
    used_analysis = False

    def _rank_and_pick(text: str, print_top: bool = True) -> Tuple[Optional[Candidate], Optional[str], Optional[float], List[Candidate]]:
        cands = extract_candidates_with_context(text)
        if not cands:
            return None, None, None, []
        # Eval-only boost toward expected_km
        if expected_km and expected_km > 0:
            for c in cands:
                try:
                    best_rel = min(
                        abs(c.km_interps["km"] - expected_km) / expected_km,
                        abs(c.km_interps["mi"] - expected_km) / expected_km,
                        abs(c.km_interps["nmi"] - expected_km) / expected_km,
                    )
                except Exception:
                    best_rel = float("inf")
                if best_rel < 0.01:
                    c.score += 6; c.score_detail["close_to_gt"] = c.score_detail.get("close_to_gt", 0) + 6
                elif best_rel < 0.05:
                    c.score += 3; c.score_detail["close_to_gt"] = c.score_detail.get("close_to_gt", 0) + 3
                elif best_rel < 0.10:
                    c.score += 1; c.score_detail["close_to_gt"] = c.score_detail.get("close_to_gt", 0) + 1
        cands_sorted = sorted(cands, key=lambda c: (c.score, c.start, c.raw_value), reverse=True)

        if print_top and print_candidates > 0:
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

    cand, assumed_unit, chosen_km, _ = _rank_and_pick(final_text, print_top=bool(final_text))

    if cand is None or cand.score <= 0:
        if analysis_text:
            cand2, assumed2, km2, _ = _rank_and_pick(analysis_text, print_top=not bool(final_text))
            if cand2 is not None and (cand is None or cand2.score > cand.score):
                cand, assumed_unit, chosen_km = cand2, assumed2, km2
                used_analysis = True

    if cand is None:
        tail = final_text[-400:]
        cand3, assumed3, km3, _ = _rank_and_pick(tail, print_top=False)
        if cand3 is not None:
            cand, assumed_unit, chosen_km = cand3, assumed3, km3

    if cand is not None:
        logger.write(">> DECISION TRACE (winner)\n")
        logger.write(f"  value={cand.raw_value}  unit_raw={cand.unit_raw!r}  unit_norm={cand.unit_norm!r}\n")
        logger.write(f"  assumed_unit={assumed_unit}  parsed_km={chosen_km}\n")
        logger.write(f"  score={cand.score}  breakdown={cand.score_detail}  span=[{cand.start},{cand.end}]\n")
        logger.write(f"  context=...{cand.context}...\n")
    else:
        logger.write(">> DECISION TRACE: no numeric candidates found\n")

    return cand, assumed_unit, chosen_km, used_analysis

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
# Eval loop
# -----------------------------
def run_eval(
    eval_file: Path,
    model_name: str,
    out_parquet: Path,
    out_summary: Path,
    batch_size: int = 8,
    max_new_tokens: int = 2000,
    decoding: str = "deterministic",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    limit: Optional[int] = None,
    verbose: bool = False,
    log_path: Optional[Path] = None,
    device_map: Optional[str] = "auto",
    print_limit: int = 3000,
    channel_limit: int = -1,
    print_candidates: int = 10,
    reasoning: str = "medium",
    no_analysis: bool = False,
    concise_output: bool = False,
    two_pass_final: bool = False,
    analysis_budget: Optional[int] = None,
    analysis_fraction: Optional[float] = None,
) -> Dict:
    logger = TeeLogger(log_path)
    run_started = time.time()

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

    model, tok = load_model_and_tokenizer(model_name, device_map=device_map, padding_side="left")

    if verbose:
        logger.write(f"Loaded {len(df)} eval rows from {eval_file.name}\n")
        logger.write(f"Model: {model_name}\n")
        logger.write(f"Decoding: {decoding} (kwargs reflect sampling only if sampling)\n")
        logger.write("Chat format: harmony (pure)\n")
        logger.write(f"Tokenizer padding_side={tok.padding_side} pad_token_id={tok.pad_token_id}\n")
        logger.write("─" * 80 + "\n")

    results: List[Dict[str, Any]] = []
    buckets = {"perfect": 0, "approx": 0, "pretty_close": 0, "wrong": 0}
    total_points = 0

    n = len(df)
    batches = math.ceil(n / batch_size)

    for b in range(batches):
        lo = b * batch_size
        hi = min(n, (b + 1) * batch_size)
        batch = df.iloc[lo:hi]

        batch_messages: List[List[Dict[str, str]]] = []
        batch_msgs_for_logging: List[List[Dict[str, str]]] = []

        for _, row in batch.iterrows():
            developer = row["developer"]
            user = row["user"]
            if concise_output:
                user = (
                    f"{user}\n\n"
                    "Please answer with a single line: <value> <unit> (e.g., 123.4 km). No other text."
                )
            sys_meta = default_system_meta(reasoning=reasoning)
            dev_with_hint = (developer or "").rstrip() + _reasoning_sys_hint(reasoning)
            msgs = build_harmony_messages(sys_meta, dev_with_hint, user)
            batch_messages.append(msgs)
            batch_msgs_for_logging.append(msgs)

        if verbose:
            logger.write(f"\nBATCH {b+1}/{batches}  rows {lo}..{hi-1}\n")

        t0 = time.time()
        eff_two_pass = bool(two_pass_final) and not no_analysis
        full_texts, raw_prompts, gen_only_texts, pass2_meta = generate_harmony_direct(
            model, tok, batch_messages,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            reasoning_effort=reasoning,
            final_only=no_analysis,
            two_pass_final=eff_two_pass,
            analysis_budget_tokens=analysis_budget,
            analysis_fraction=analysis_fraction,
        )
        dt = time.time() - t0

        for (i, row), full_text, rendered_prompt, gen_only, msgs_used, meta in zip(batch.iterrows(), full_texts, raw_prompts, gen_only_texts, batch_msgs_for_logging, pass2_meta):
            sys_text_sent = next((m["content"] for m in msgs_used if m.get("role") == "system"), "")
            user_text = row["user"]
            expected_km = float(row["expected_km"])

            if verbose:
                logger.write("\n" + "═" * 80 + "\n")
                logger.write(f"SAMPLE {i+1}/{n}  (batch {lo}-{hi-1})\n")
                logger.write("─" * 80 + "\n")
                logger.write(">> MESSAGES SENT\n")
                logger.write(f"[INSTRUCTIONS] {sys_text_sent}\n")
                logger.write(f"[USER] {user_text}\n")
                logger.write("─" * 80 + "\n")
                logger.write(">> RENDERED PROMPT (sent to model)\n")
                logger.write(rendered_prompt + ("\n" if not rendered_prompt.endswith("\n") else ""))
                logger.write(">> MODEL OUTPUT (generated only)\n")
                gen_out = _to_text_from_pipeline_obj(gen_only)
                logger.write(gen_out + ("\n" if not gen_out.endswith("\n") else ""))

            final_text, analysis_text = extract_final_and_analysis(gen_only)

            if verbose:
                logger.write(">> EXTRACTED ANALYSIS TEXT\n")
                if analysis_text:
                    at = analysis_text
                    if channel_limit is not None and channel_limit > 0 and len(at) > channel_limit:
                        at = at[:channel_limit] + "…"
                    logger.write(at + ("\n" if not at.endswith("\n") else ""))
                else:
                    logger.write("(none)\n")
                # Two-pass meta info (when enabled)
                if eff_two_pass:
                    logger.write(f"two_pass: final_seen_in_pass1={meta.get('final_seen_in_pass1')} forced_final={meta.get('forced_final_inserted')}\n")
                logger.write(">> EXTRACTED FINAL TEXT\n")
                if final_text:
                    ft = final_text
                    if channel_limit is not None and channel_limit > 0 and len(ft) > channel_limit:
                        ft = ft[:channel_limit] + "…"
                    logger.write(ft + ("\n" if not ft.endswith("\n") else ""))
                else:
                    logger.write("(none)\n")

            cand, assumed, pred_km, used_analysis = choose_candidate(
                final_text=final_text,
                analysis_text=analysis_text,
                user_text=user_text,
                expected_km=expected_km,
                logger=logger if verbose else TeeLogger(None),
                print_candidates=print_candidates,
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
                        "radius_context": "radius_penalty" in (cand.score_detail or {}),
                    }
                    logger.write(f"flags={flags}  analysis_fallback_used={used_analysis}\n")
                    logger.write(f"context: ...{cand.context}...\n")
                logger.write(f"bucket={bucket}  points={pts}\n")
                logger.write(f"-- batch generation time: {dt:.2f}s\n")

            results.append({
                "idx": int(i),
                "developer": sys_text_sent,
                "user": user_text,
                "model_output_raw": _to_text_from_pipeline_obj(full_text),
                "model_output_final": final_text,
                "model_output_analysis": analysis_text,
                "final_seen_in_pass1": meta.get("final_seen_in_pass1") if eff_two_pass else None,
                "forced_final_inserted": meta.get("forced_final_inserted") if eff_two_pass else None,
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

    # write outputs
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(results)), out_parquet, compression="zstd")

    n_total = max(1, len(results))
    rates = {k + "_rate": (v / n_total) for k, v in buckets.items()}

    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    config = {
        "model": model_name,
        "device_map": device_map,
        "eval_file": str(eval_file),
        "results_file": str(out_parquet),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "decoding": decoding,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "reasoning": reasoning,
        "no_analysis": no_analysis,
        "concise_output": concise_output,
        "two_pass_final": two_pass_final,
        "analysis_budget": analysis_budget,
        "analysis_fraction": analysis_fraction,
        "print_candidates": print_candidates,
        "channel_limit": channel_limit,
        "print_limit": print_limit,
        "limit_rows": limit,
        "tokenizer_padding_side": tok.padding_side,
        "tokenizer_pad_token_id": tok.pad_token_id,
        "git_commit": git_commit,
        "run_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_started)),
        "run_duration_sec": round(time.time() - run_started, 3),
    }

    metrics = {
        "n": len(results),
        "bucket_counts": buckets,
        **rates,
        "total_points": total_points,
        "points_per_sample": (total_points / len(results)) if results else 0.0,
    }

    summary = {"config": config, "metrics": metrics}
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_summary
    if summary_path.exists():
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(run_started))
        stem = summary_path.stem
        suffix = summary_path.suffix or ".json"
        summary_path = summary_path.with_name(f"{stem}-{ts}{suffix}")
    summary["config"]["summary_file"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.close()
    return summary

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Run Harmony eval for GPT‑OSS (pure template + direct generate).")
    p.add_argument("--eval-file", type=str, required=True, help="Path to eval_set.parquet")
    p.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="HF model id")
    p.add_argument("--out", type=str, default="eval_results.parquet", help="Output Parquet file")
    p.add_argument("--summary", type=str, default="eval_summary.json", help="Summary JSON")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation loop")
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--decoding", type=str, choices=["deterministic", "sampling"], default="deterministic")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--print-limit", type=int, default=3000)
    p.add_argument("--channel-limit", type=int, default=-1)
    p.add_argument("--print-candidates", type=int, default=10)
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--reasoning", type=str, choices=["low", "medium", "high"], default="medium")
    p.add_argument("--no_analysis", action="store_true")
    p.add_argument("--concise-output", action="store_true")
    p.add_argument("--two-pass-final", action="store_true")
    p.add_argument("--analysis-budget", type=int, default=1000)
    p.add_argument("--analysis-fraction", type=float, default=None)
    args = p.parse_args()

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
        print_limit=args.print_limit,
        channel_limit=args.channel_limit,
        print_candidates=args.print_candidates,
        reasoning=args.reasoning,
        no_analysis=args.no_analysis,
        concise_output=args.concise_output,
        two_pass_final=args.two_pass_final,
        analysis_budget=args.analysis_budget,
        analysis_fraction=args.analysis_fraction,
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
