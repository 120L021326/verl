import argparse
import sys
from pathlib import Path
from typing import Any

from transformers import AutoModelForVision2Seq, AutoProcessor

WORKSPACE_ROOT = Path("/workspace")
ALPAMAYO_SRC = WORKSPACE_ROOT / "alpamayo" / "src"
if str(ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO_SRC))

TRAJ_FUTURE_START_TOKEN = "<|traj_future_start|>"

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN

def configure_generation_stop(
    generation_config: Any,
    traj_future_start_token_id: int,
) -> list[int]:
    eos_token_ids: list[int] = []
    original_eos_token_id = generation_config.eos_token_id
    if isinstance(original_eos_token_id, int):
        eos_token_ids.append(original_eos_token_id)
    elif isinstance(original_eos_token_id, (list, tuple)):
        for token_id in original_eos_token_id:
            if isinstance(token_id, int):
                eos_token_ids.append(token_id)
    if traj_future_start_token_id not in eos_token_ids:
        eos_token_ids.append(traj_future_start_token_id)
    generation_config.eos_token_id = eos_token_ids

def export_from_alpamayo_checkpoint(
    alpamayo_ckpt: str,
    output_dir: str,
    dtype: str,
) -> None:
    model = AlpamayoR1.from_pretrained(alpamayo_ckpt, dtype=dtype)
    vlm = model.vlm
    generation_config = vlm.generation_config
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)

    new_vocab_size = len(tokenizer)
    print(f"Original vocab size: {vlm.config.vocab_size}, new vocab size after loading Alpamayo checkpoint: {new_vocab_size}")
    vlm.resize_token_embeddings(new_vocab_size)
    if hasattr(vlm.config, "text_config"):
        vlm.config.text_config.vocab_size = new_vocab_size
    vlm.config.vocab_size = new_vocab_size

    traj_future_start_token_id = tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN)
    if not isinstance(traj_future_start_token_id, int) or traj_future_start_token_id < 0:
        raise ValueError(f"Failed to resolve token id for {TRAJ_FUTURE_START_TOKEN} after loading Alpamayo checkpoint")
    configure_generation_stop(generation_config, traj_future_start_token_id)
    print(f"Configured generation eos_token_id: {generation_config.eos_token_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vlm.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print(f"Exported Alpamayo VLM to: {output_path}")
    print(f"Tokenizer vocab size: {new_vocab_size}")


def export_from_base_model(
    base_model: str,
    output_dir: str,
    traj_vocab_size: int,
    add_special_tokens: bool,
) -> None:
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    discrete_tokens = [f"<i{i}>" for i in range(traj_vocab_size)]
    tokenizer.add_tokens(discrete_tokens)

    if add_special_tokens:
        tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    else:
        tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    model.config.vocab_size = new_vocab_size

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print(f"Saved Alpamayo-ready Qwen3-VL model to: {output_path}")
    print(f"Tokenizer vocab size: {new_vocab_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standalone VLM checkpoint for RLinf. "
        "Preferred path: export directly from a full Hugging Face Alpamayo checkpoint."
    )
    parser.add_argument(
        "--alpamayo-ckpt",
        default="/workspace/.cache/modelscope/hub/models/nv-community/Alpamayo-R1-10B",
        help="Full Hugging Face Alpamayo checkpoint path. Preferred when available.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Fallback base Qwen3-VL model path or HF id.",
    )
    parser.add_argument("--output-dir", help="Export directory",default="/workspace/Alpamayo-R1-10B-vlm")
    parser.add_argument("--dtype", default="bfloat16", help="dtype passed to AlpamayoR1.from_pretrained")
    parser.add_argument("--traj-vocab-size", type=int, default=4000)
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()

    if args.alpamayo_ckpt:
        export_from_alpamayo_checkpoint(
            alpamayo_ckpt=args.alpamayo_ckpt,
            output_dir=args.output_dir,
            dtype=args.dtype,
        )
        return

    if args.base_model:
        export_from_base_model(
            base_model=args.base_model,
            output_dir=args.output_dir,
            traj_vocab_size=args.traj_vocab_size,
            add_special_tokens=args.add_special_tokens,
        )
        return

    raise ValueError("You must provide either --alpamayo-ckpt or --base-model.")


if __name__ == "__main__":
    main()
