EVAL
First step: generate a table of evaluation data:
python3 build_eval_set.py --n 1000

Then run the eval with the settings you want:
python3 run_eval_gpt_oss.py   --eval-file eval_set.parquet   --model openai/gpt-oss-20b   --out eval_results.parquet   --summary eval_summary.json   --reasoning high --max-new-tokens 2000 --decoding deterministic --verbose --log eval_run_high.txt --batch-size 8 --chat-format harmony

--> In my experience for now Harmony format really does make things a lot better for this model, as does "high" reasoning. Set the batch size to whatever works on your GPU(s). Max new tokens 2000 is plenty for "high" reasoning to work it's magic, if you reduce to "medium" you can drop to 1000 or less.

TRAINING

(coming soon)


list of open points
1. It would be cool to add orientation to the dataset, so that our LLM learns to predict something like "X kilometers towards the SE"
2. There's no reason not to add a LOT more diversity including different languages to the basic prompt structure
3. We could include different reference formats for the GPS points and different ways to calculate distance (acknowledging this goes beyond the idea of building a dataset that tries to teach _intuition_ about distance, so a little out of scope).
