from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    alpamayo_src = repo_root / "alpamayo" / "src"
    alpamayo_finetune = repo_root / "alpamayo" / "finetune"
    alpamayo_rl_models = repo_root / "alpamayo" / "finetune" / "rl" / "models"
    for path in (repo_root, alpamayo_src, alpamayo_finetune, alpamayo_rl_models):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _build_prompt(model_path: str, prompt: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {
            "role": "system",
            "content": "You are a driving assistant that generates safe and accurate actions.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "<|cot_start|>"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def _validate_training_checkpoint(model_path: str) -> None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing config.json under --model path: {model_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if model_type == "alpamayo_r1":
        raise SystemExit(
            "The --model path points to the raw Alpamayo release checkpoint "
            "(model_type='alpamayo_r1'). This smoke test expects the training-ready "
            "ReasoningVLA checkpoint produced by alpamayo/scripts/"
            "convert_release_config_to_training.py, whose config has "
            "model_type='alpamayo_reasoning_vla'."
        )
    if model_type != "alpamayo_reasoning_vla":
        raise SystemExit(
            f"Unsupported --model config model_type={model_type!r}. Expected "
            "'alpamayo_reasoning_vla'."
        )


def main() -> None:
    _add_repo_paths()

    parser = argparse.ArgumentParser(
        description="Smoke-test loading an official Alpamayo ReasoningVLA checkpoint in vLLM."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ALPAMAYO_MODEL_DIR"),
        help="Official training-ready ReasoningVLA checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default="output the chain-of-thought reasoning of the driving process.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("Set --model or ALPAMAYO_MODEL_DIR to a ReasoningVLA checkpoint.")
    _validate_training_checkpoint(args.model)

    os.environ.setdefault("HF_ALLOW_CODE_EXECUTION", "1")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    from vllm import LLM, SamplingParams

    from examples.alpamayo_demo.register_reasoning_vla_vllm import (
        reasoning_vla_hf_overrides,
        register_reasoning_vla_for_vllm,
    )

    register_reasoning_vla_for_vllm()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        hf_overrides=reasoning_vla_hf_overrides,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_mm_preprocessor_cache=True,
    )

    sampling_params = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        detokenize=True,
    )
    if hasattr(sampling_params, "skip_special_tokens"):
        sampling_params.skip_special_tokens = False

    prompt = _build_prompt(args.model, args.prompt)
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    for output in outputs:
        for completion in output.outputs:
            print(completion.text)


if __name__ == "__main__":
    main()
