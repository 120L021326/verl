# Alpamayo Demo

This demo shows the thinnest safe way to prototype an Alpamayo-style Qwen3-VL setup inside `verl` without modifying rollout or reward core code.

The demo package provides:

- `prepare_demo_asset.py`: prepares a local Qwen3-VL-2B-Instruct asset with Alpamayo demo tokens and a `generation_config` whose `eos_token_id` includes both the original EOS token and `<|traj_future_start|>`.
- `dataset.py`: a custom dataset that loads raw Alpamayo-style `clip_id` samples and converts them into verl-compatible multimodal prompts.
- `reward_fn.py`: a custom reward function that checks whether decoded output still contains `<|cot_start|>` and `<|cot_end|>`.
- `run_qwen3_vl_alpamayo_demo.sh`: a minimal GRPO run script that uses the custom dataset and custom reward hooks.
- `data/train.jsonl` and `data/val.jsonl`: tiny example files for a smoke-test style run.

## Quick start

```bash
bash examples/alpamayo_demo/run_qwen3_vl_alpamayo_demo.sh
```

Environment variables:

- `PROJECT_DIR`: repository root. Defaults to the current working directory.
- `ENGINE`: rollout engine. Defaults to `vllm`.

The run script currently uses these paths by default:

- `EXAMPLE_DIR=/workspace/verl/examples/alpamayo_demo`
- `MODEL_DIR=/workspace/Alpamayo-R1-10B-vlm`
- `TRAIN_FILE=/workspace/ncore_10clips/train.jsonl`
- `VAL_FILE=/workspace/ncore_10clips/val.jsonl`

Edit the script or pass Hydra overrides at the end of the command if your model or data are stored elsewhere.

## Dataset contract

Each record must provide a `clip_id`. Optional raw-loader fields include:

- `t0_us`: timestamp used as the current frame anchor. Defaults to `5100000`.
- `num_frames`: number of image frames to load. Defaults to `1`.
- `camera_features`: list of camera names/features. Defaults to `["camera_front_wide_120fov"]`.
- `ncore_manifest_path`: local NCore manifest JSON path.
- `ncore_root`: local NCore clip root.
- `extract_cache_dir`: directory used to cache extracted `.itar` stores.
- `ground_truth`, `reward_model.ground_truth`, `answer`, `coc`, or `target`: optional target value passed through to reward computation.
- `extra_info`: optional metadata carried through rollout and reward computation.

Example:

```json
{"clip_id":"100ae358-f548-49b8-af4d-c0afdbcfe9ed","ncore_manifest_path":"/workspace/ncore_10clips/clips/00ae358-f548-49b8-af4d-c0afdbcfe9ed/pai_100ae358-f548-49b8-af4d-c0afdbcfe9ed.json","ncore_root":"/workspace/ncore_10clips/clips/00ae358-f548-49b8-af4d-c0afdbcfe9ed","extract_cache_dir":"/tmp/alpamayo_ncore_extract/00ae358-f548-49b8-af4d-c0afdbcfe9ed"}
```

The dataset loads raw driving data, tokenizes the history trajectory with Alpamayo's delta tokenizer, replaces the trajectory placeholder chain with discrete `<iN>` tokens, and emits a verl-compatible multimodal `raw_prompt` containing images plus text.

Local NCore loading is used when `ncore_manifest_path` is present in the record or `ALPAMAYO_NCORE_MANIFEST_PATH` is set. Otherwise, the loader falls back to the official `physical_ai_av` interface.

This keeps the demo on verl's dataset interface while making the input path much closer to the Alpamayo reference pipeline.
