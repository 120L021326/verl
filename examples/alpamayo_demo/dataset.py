from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

import einops
import torch
from PIL import Image
from torch.utils.data import Dataset

from examples.alpamayo_demo.ref.delta_tokenizer import DeltaTrajectoryTokenizer
from examples.alpamayo_demo.ref.load_physical_aiavdataset import load_physical_aiavdataset


MIN_PIXELS = 163840
MAX_PIXELS = 196608
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
DEFAULT_T0_US = 5_100_000
DEFAULT_NUM_FRAMES = 1
DEFAULT_CAMERA_FEATURES = ["camera_front_wide_120fov"]

def resolve_ground_truth(row_dict: dict[str, Any]) -> Any:
    if "ground_truth" in row_dict:
        return row_dict["ground_truth"]
    reward_model = row_dict.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model["ground_truth"]
    for candidate_key in ("answer", "coc", "target"):
        if candidate_key in row_dict:
            return row_dict[candidate_key]
    return ""


def create_message(frames: list[Any]) -> list[dict[str, Any]]:
    num_traj_token = 48
    hist_traj_placeholder = (
        f"{TRAJ_TOKEN['history_start']}{TRAJ_TOKEN['history'] * num_traj_token}{TRAJ_TOKEN['history_end']}"
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


def tokenize_history_trajectory(tokenizer: Any, traj_data: dict[str, Any]) -> Any:
    history_xyz = traj_data["ego_history_xyz"]
    history_rot = traj_data["ego_history_rot"]
    if history_xyz.ndim != 4:
        raise ValueError(f"ego_history_xyz must be 4D [B, n_traj, T, 3], got shape {tuple(history_xyz.shape)}")

    batch_size = history_xyz.shape[0]
    flat_history_xyz = history_xyz.flatten(start_dim=0, end_dim=1)
    flat_history_rot = history_rot.flatten(start_dim=0, end_dim=1)
    history_token_ids = tokenizer.encode(
        hist_xyz=flat_history_xyz[:, :1],
        hist_rot=flat_history_rot[:, :1],
        fut_xyz=flat_history_xyz,
        fut_rot=flat_history_rot,
    )
    if not isinstance(history_token_ids, torch.Tensor):
        history_token_ids = torch.as_tensor(history_token_ids)
    return einops.rearrange(history_token_ids, "(b n_traj) n -> b (n_traj n)", b=batch_size)


def frame_tensor_to_pil(frame_chw: Any) -> Any:
    frame_hwc = frame_chw.detach().cpu().permute(1, 2, 0).numpy()
    if str(frame_hwc.dtype) != "uint8":
        frame_hwc = frame_hwc.clip(0, 255).astype("uint8")
    return Image.fromarray(frame_hwc)


def replace_history_placeholder(messages: list[dict[str, Any]], history_token_ids: list[int]) -> None:
    num_traj_token = 48

    replacement = (
        f"{TRAJ_TOKEN['history_start']}"
        + "".join(f"<i{token_id+3000}>" for token_id in history_token_ids)
        + f"{TRAJ_TOKEN['history_end']}"
    )
    placeholder = (
        f"{TRAJ_TOKEN['history_start']}{TRAJ_TOKEN['history'] * num_traj_token}{TRAJ_TOKEN['history_end']}"
    )

    user_content = messages[1]["content"]
    for item in user_content:
        if item.get("type") == "text":
            item["text"] = item["text"].replace(placeholder, replacement, 1)
            return
    raise ValueError("Failed to find history placeholder text in Alpamayo prompt")


class AlpamayoDemoDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: Any,
        config: Any,
        processor: Optional[Any] = None,
        max_samples: int = -1,
    ):
        del processor

        if isinstance(data_files, str):
            data_files = [data_files]

        self.data_files = [str(Path(data_file).expanduser()) for data_file in data_files]
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        self.prompt_key = config.get("prompt_key", "prompt")
        self.clip_id_key = config.get("clip_id_key", "clip_id")
        self.default_data_source = config.get("default_data_source", "alpamayo_demo")
        self.default_t0_us = int(config.get("t0_us", DEFAULT_T0_US))
        self.default_num_frames = int(config.get("num_frames", DEFAULT_NUM_FRAMES))
        self.default_camera_features = config.get("camera_features", DEFAULT_CAMERA_FEATURES)
        self.default_ncore_manifest_path = config.get("ncore_manifest_path")
        self. default_ncore_root = config.get("ncore_root")
        self.default_extract_cache_dir = config.get("extract_cache_dir")
        self._read_files()

    def _read_one_file(self, data_file: str) -> Any:
        if data_file.endswith(".jsonl"):
            return [json.loads(line) for line in Path(data_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if data_file.endswith(".json"):
            payload = json.loads(Path(data_file).read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"JSON demo data must be a list of records: {data_file}")
            return payload
        raise ValueError(f"Unsupported file format: {data_file}")

    def _read_files(self) -> None:
        self.dataframe: list[dict[str, Any]] = []
        for data_file in self.data_files:
            self.dataframe.extend(self._read_one_file(data_file))
        if 0 < self.max_samples < len(self.dataframe):
            self.dataframe = self.dataframe[: self.max_samples]

    def __len__(self) -> int:
        return len(self.dataframe)

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: Any,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        del config
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )
        return images, videos

    def _build_messages_from_clip(self, row_dict: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        clip_id = row_dict.get(self.clip_id_key)
        if not clip_id:
            raise ValueError(f"Each Alpamayo raw sample must contain '{self.clip_id_key}'.")

        loaded_sample = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=int(row_dict.get("t0_us", self.default_t0_us)),
            camera_features=row_dict.get("camera_features", self.default_camera_features),
            num_frames=int(row_dict.get("num_frames", self.default_num_frames)),
            ncore_manifest_path=row_dict.get("ncore_manifest_path", self.default_ncore_manifest_path),
            ncore_root=row_dict.get("ncore_root", self.default_ncore_root),
            extract_cache_dir=row_dict.get("extract_cache_dir", self.default_extract_cache_dir),
        )

        frames = loaded_sample["image_frames"].flatten(0, 1)
        messages = create_message([frame_tensor_to_pil(frame) for frame in frames])

        history_tokenizer = DeltaTrajectoryTokenizer()
        history_tokens = tokenize_history_trajectory(
            history_tokenizer,
            {
                "ego_history_xyz": loaded_sample["ego_history_xyz"],
                "ego_history_rot": loaded_sample["ego_history_rot"],
            },
        ).squeeze(0)
        history_token_ids = [int(token_id) for token_id in history_tokens.tolist()]

        replace_history_placeholder(messages, history_token_ids)
        clip_meta = {
            "clip_id": loaded_sample.get("clip_id", clip_id),
            "t0_us": loaded_sample.get("t0_us"),
            "hist_token_count": len(history_token_ids),
        }
        return messages, clip_meta

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_dict = copy.deepcopy(self.dataframe[item])
        row_dict["raw_prompt"], clip_meta = self._build_messages_from_clip(row_dict)
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        row_dict.setdefault("data_source", self.default_data_source)
        row_dict["ground_truth"] = resolve_ground_truth(row_dict)
        reward_model = copy.deepcopy(row_dict.get("reward_model") or {})
        reward_model.setdefault("ground_truth", row_dict["ground_truth"])
        row_dict["reward_model"] = reward_model

        extra_info = copy.deepcopy(row_dict.get("extra_info") or {})
        extra_info.setdefault("index", item)
        extra_info.update({key: value for key, value in clip_meta.items() if value is not None})
        row_dict["extra_info"] = extra_info
        row_dict["index"] = extra_info["index"]
        row_dict["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        return row_dict
