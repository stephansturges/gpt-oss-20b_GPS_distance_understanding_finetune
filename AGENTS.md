# Repository Guidelines

## Project Structure & Module Organization
- `data_generator/` – Dataset utilities and configs (e.g., `build_parquet_dataset.py`, `gps_example_generator.py`, JSON noise/config files).
- `eval_generator/` – Evaluation tooling: `build_eval_set.py`, `run_eval_gpt_oss.py`, phrase JSONs, and generated artifacts (`eval_set.parquet`, logs).
- `train/` – Fine-tuning entry point: `train_gpt_oss_from_parquet.py` (TRL + PEFT LoRA).
- Root files – `requirements.txt`, `.gitignore`, `readme.txt`. A local `.venv/` is expected but ignored.

## Setup, Build, and Development Commands
- Create env and install deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt` (install a matching PyTorch CUDA wheel per your system).
- Generate eval set (example):
  - `python3 eval_generator/build_eval_set.py --n 1000 --out eval_generator/eval_set.parquet --seed 42`
- Run evaluation (Harmony template recommended):
  - `python3 eval_generator/run_eval_gpt_oss.py --eval-file eval_generator/eval_set.parquet --model openai/gpt-oss-20b --out eval_generator/eval_results.parquet --summary eval_generator/eval_summary.json --reasoning high --decoding deterministic --batch-size 8 --chat-format harmony --log eval_generator/eval_run_high.txt`
- Start training (Parquet with `messages` list required):
  - `python3 train/train_gpt_oss_from_parquet.py --parquet data/gps_train.parquet --output-dir runs/gptoss-gps-lora --epochs 1 --per-device-train-batch-size 2 --grad-accum-steps 8 --max-length 2048`

## Coding Style & Naming Conventions
- Python ≥ 3.10, PEP 8, 4‑space indentation.
- Names: modules/functions `snake_case`, classes `CapWords`. Keep files small and cohesive.
- Prefer type hints and module docstrings; keep pure helpers in `data_generator/` and eval-specific logic in `eval_generator/`.
- Chat formatting: default to `--chat-format harmony`; keep the developer/system text stable.

## Testing Guidelines
- No formal unit test suite yet. Treat `eval_generator` runs as functional tests: verify bucket counts and points in `eval_summary.json`.
- If adding tests, use `pytest`, place under `tests/` with `test_*.py`. Cover numeric parsing, unit normalization, and Haversine helpers.

## Commit & Pull Request Guidelines
- History favors short, present‑tense subjects (e.g., `fix parsing`, `update readme`). Use imperative mood; keep bodies concise with rationale.
- PRs should include: purpose, summary of changes, sample commands to reproduce, and before/after eval metrics. Link related issues.

## Security & Configuration Tips
- Authenticate for private models or pushes: `huggingface-cli login`.
- GPU tips: prefer `--device-map auto`; set `--max-new-tokens` conservatively; monitor VRAM.
- Don’t commit secrets or large artifacts. Store outputs under `runs/` or `eval_generator/` and clean logs as needed. Prefer Parquet with zstd compression.

