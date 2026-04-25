# Alpamayo Demo

This demo shows the thinnest safe way to prototype an Alpamayo-style Qwen3-VL setup inside `verl` without modifying rollout or reward core code.

The demo package provides:

- `prepare_demo_asset.py`: prepares a local Qwen3-VL-2B-Instruct asset with Alpamayo demo tokens and a `generation_config` whose `eos_token_id` includes both the original EOS token and `<|traj_future_start|>`.
- `dataset.py`: a custom dataset that supports both already-fused historical trajectory input and raw Alpamayo-style `clip_id` samples.
- `reward_fn.py`: a custom reward function that checks whether decoded output still contains `<|cot_start|>` and `<|cot_end|>`.
- `run_qwen3_vl_2b_demo.sh`: a minimal GRPO run script that uses the custom dataset and custom reward hooks.
- `data/train.jsonl` and `data/val.jsonl`: tiny example files for a smoke-test style run.

## Quick start

```bash
bash examples/alpamayo_demo/run_qwen3_vl_2b_demo.sh
```

Environment variables:

- `ASSET_DIR`: where the prepared local demo model should be written.
- `BASE_MODEL`: base model to clone locally. Defaults to `Qwen/Qwen3-VL-2B-Instruct`.
- `ENGINE`: rollout engine. Defaults to `hf` for the simplest path.
- `TRAIN_FILE` / `VAL_FILE`: override demo data paths.

## Dataset contract

Each record may either:

1. provide a ready-made `prompt`, or
2. provide `fused_history` plus optional `instruction`, or
3. provide `clip_id` plus optional raw-loader fields such as `t0_us`, `num_frames`, `camera_features`, `ncore_manifest_path`, `ncore_root`, and `extract_cache_dir`.

When `fused_history` is used, the dataset wraps it into a single user message that includes:

- `<|traj_history_start|>` and `<|traj_history_end|>` around the fused history text
- instructions asking the model to emit visible `<|cot_start|>` / `<|cot_end|>` markers
- a request to begin future trajectory output with `<|traj_future_start|>`

When `clip_id` is used, the dataset loads raw driving data, tokenizes the history trajectory with Alpamayo's delta tokenizer, replaces the trajectory placeholder chain with discrete `<iN>` tokens, and emits a verl-compatible multimodal `raw_prompt` containing images plus text.

This keeps the demo on verl's dataset interface while making the input path much closer to the Alpamayo reference pipeline.
