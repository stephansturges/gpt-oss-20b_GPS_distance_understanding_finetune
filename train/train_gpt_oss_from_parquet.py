
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gpt_oss_from_parquet.py
-----------------------------
Fine-tune `openai/gpt-oss-20b` on a Harmony-format Parquet dataset (with a `messages` column)
using TRL's SFTTrainer and PEFT LoRA, following the OpenAI Cookbook tutorial.

Key facts (from the tutorial):
- GPT-OSS models expect Harmony-formatted conversations; TRL will apply the model's chat template
  over your `messages` list automatically during SFT.  (OpenAI Cookbook)
- We load the base model with MXFP4 config (dequantize=True) and dtype=bfloat16, with
  device_map="auto" for multi-GPU sharding.  (OpenAI Cookbook)
- For the MoE architecture, in addition to "all-linear", we include expert projection layers
  in LoRA via `target_parameters`.  (OpenAI Cookbook)

Citations:
- OpenAI Cookbook: Fine-tuning with gpt-oss and Hugging Face Transformers
  https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers

Usage:
  python3 train_gpt_oss_from_parquet.py \
      --parquet data/gps_train.parquet \
      --model openai/gpt-oss-20b \
      --output-dir runs/gptoss-gps-lora \
      --epochs 1 --per-device-train-batch-size 2 --grad-accum-steps 8 \
      --max-length 2048 --learning-rate 2e-4 --warmup-ratio 0.03 \
      --push-to-hub false

After training, the adapter is saved to --output-dir. You can also push to Hub
(--push-to-hub true) if `huggingface-cli login` has been run.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import datasets
from datasets import load_dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Mxfp4Config,
)

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model


def peek_dataset(ds, n: int = 2):
    """Print a couple of rows to confirm shape & messages format."""
    print("\n=== Dataset sample ===")
    for i in range(min(n, len(ds))):
        row = ds[i]
        keys = list(row.keys())
        print(f"\nRow {i} keys: {keys}")
        # Prefer the 'messages' field (Harmony)
        msgs = row.get("messages")
        if msgs:
            # Show a compact view
            preview = []
            for msg in msgs[:5]:
                role = msg.get("role")
                content = (msg.get("content") or "")[:120].replace("\n", " ")
                thinking_present = "thinking" in msg and msg["thinking"] is not None
                preview.append({"role": role, "content": content, "thinking?": thinking_present})
            print("messages (first 5):", json.dumps(preview, ensure_ascii=False))
        else:
            print("No 'messages' field found! The SFTTrainer expects Harmony-format messages.")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune openai/gpt-oss-20b on a Harmony Parquet dataset with TRL+PEFT.")
    ap.add_argument("--parquet", type=str, required=True, help="Path to Parquet file (with a 'messages' column).")
    ap.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="Base model id.")
    ap.add_argument("--output-dir", type=str, default="gptoss-gps-lora", help="Output directory for adapter/checkpoints.")
    ap.add_argument("--seed", type=int, default=42)

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", type=str, default="cosine_with_min_lr")
    ap.add_argument("--min-lr-rate", type=float, default=0.1, help="For cosine_with_min_lr scheduler.")

    ap.add_argument("--max-length", type=int, default=2048, help="Max sequence length (tokenized).")
    ap.add_argument("--packing", action="store_true", help="Enable sequence packing for efficiency.")

    # LoRA
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)

    # Hub & logging
    ap.add_argument("--push-to-hub", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--hub-repo-id", type=str, default=None)
    ap.add_argument("--report-to", type=str, default="none", help="e.g., 'trackio' if installed, else 'none'")

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # -----------------------------
    # Load dataset from Parquet
    # -----------------------------
    print(f"Loading Parquet: {args.parquet}")
    ds = load_dataset("parquet", data_files=args.parquet, split="train")
    if "messages" not in ds.column_names:
        raise ValueError("Parquet must include a 'messages' column (Harmony list of dicts).")

    peek_dataset(ds, n=2)

    # -----------------------------
    # Tokenizer (chat template is in the model card)
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        # Right padding is fine for training; the chat template will be applied by TRL.
        padding_side="right",
        trust_remote_code=True,
    )
    # Ensure a pad token exists; if not, mirror eos.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Base model (MXFP4-aware load, dequantize to bf16)
    # -----------------------------
    quant_cfg = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,         # per Transformers deprecation notice, use `dtype` not `torch_dtype`
        quantization_config=quant_cfg,
        use_cache=False,              # disable cache for gradient checkpointing
        device_map="auto",            # shard across available GPUs
        low_cpu_mem_usage=True,
    )
    print("Loading base model with kwargs:", {k: (str(v) if not isinstance(v, dict) else v) for k, v in model_kwargs.items()})
    base_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # -----------------------------
    # LoRA (include MoE expert projections)
    # -----------------------------
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",  # standard linears
        # Include a subset of expert projections as in the tutorial; adjust as needed.
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # -----------------------------
    # TRL SFT config and trainer
    # -----------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate} if args.scheduler == "cosine_with_min_lr" else None,
        logging_steps=10,
        max_seq_length=args.max_length,
        packing=args.packing,
        report_to=args.report_to if args.report_to.lower() != "none" else [],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo_id,
        save_safetensors=True,
    )

    # SFTTrainer will:
    #  • read the `messages` field
    #  • apply the model's chat template
    #  • tokenize & pack (if enabled)
    #  • run the supervised fine-tuning loop
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving adapter and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print("Pushing to Hugging Face Hub...")
        trainer.push_to_hub()

    print("\nDone.")
    print(f"Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

