# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dataset_module = load_module("alpamayo_demo_dataset_test", "examples/alpamayo_demo/dataset.py")
reward_module = load_module("alpamayo_demo_reward_test", "examples/alpamayo_demo/reward_fn.py")


class TrajectoryAwareTokenizerStub:
    def convert_tokens_to_ids(self, token_name: str) -> int:
        if token_name.startswith("<i") and token_name.endswith(">"):
            return 1
        return -1


class HistoryTokenizerStub:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
        del hist_xyz, hist_rot, fut_xyz, fut_rot
        return [[i for i in range(48)]]


class HistoryTokensStub:
    def __init__(self, values):
        self._values = list(values)

    def squeeze(self, dim):
        assert dim == 0
        return self

    def tolist(self):
        return list(self._values)


class FakeImageFrames:
    def __init__(self, frames):
        self.frames = list(frames)

    def flatten(self, start_dim, end_dim):
        assert (start_dim, end_dim) == (0, 1)
        return list(self.frames)


def test_dataset_accepts_prebuilt_prompt_messages(tmp_path):
    dataset_file = tmp_path / "demo_prompt.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "already built"}],
                "extra_info": {"index": 7},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = dataset_module.AlpamayoDemoDataset(
        data_files=str(dataset_file),
        tokenizer=None,
        processor=None,
        config={},
    )

    sample = dataset[0]
    assert sample["raw_prompt"] == [{"role": "user", "content": "already built"}]
    assert sample["index"] == 7


def test_dataset_normalizes_reference_style_answer_fields(tmp_path):
    dataset_file = tmp_path / "demo_answer.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "reference-style prompt"}],
                "answer": "reference-answer",
                "extra_info": {"index": 3},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = dataset_module.AlpamayoDemoDataset(
        data_files=str(dataset_file),
        tokenizer=None,
        processor=None,
        config={},
    )

    sample = dataset[0]
    assert sample["ground_truth"] == "reference-answer"
    assert sample["reward_model"] == {"ground_truth": "reference-answer"}
    assert sample["raw_prompt"] == [{"role": "user", "content": "reference-style prompt"}]


def test_dataset_supports_raw_clip_samples_with_history_token_replacement(tmp_path):
    dataset_file = tmp_path / "demo_clip.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "clip_id": "clip-001",
                "answer": "clip-ground-truth",
                "instruction": "Prefer a conservative plan.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = dataset_module.AlpamayoDemoDataset(
        data_files=str(dataset_file),
        tokenizer=TrajectoryAwareTokenizerStub(),
        processor=None,
        config={},
    )

    def fake_loader(**kwargs):
        assert kwargs["clip_id"] == "clip-001"
        return {
            "clip_id": "clip-001",
            "t0_us": kwargs["t0_us"],
            "image_frames": FakeImageFrames(["frame-0", "frame-1"]),
            "camera_indices": [1],
            "relative_timestamps": [[0.0, 0.1]],
            "ego_history_xyz": "unused-history-xyz",
            "ego_history_rot": "unused-history-rot",
        }

    original_tokenize_history_trajectory = getattr(dataset_module, "tokenize_history_trajectory")
    original_frame_tensor_to_pil = getattr(dataset_module, "frame_tensor_to_pil")
    original_loader = getattr(dataset_module, "load_physical_aiavdataset")
    original_history_tokenizer_cls = getattr(dataset_module, "DeltaTrajectoryTokenizer")
    setattr(dataset_module, "load_physical_aiavdataset", fake_loader)
    setattr(dataset_module, "DeltaTrajectoryTokenizer", HistoryTokenizerStub)
    setattr(dataset_module, "tokenize_history_trajectory", lambda tokenizer, traj_data: HistoryTokensStub(list(range(48))))
    setattr(dataset_module, "frame_tensor_to_pil", lambda frame: f"image-for-{frame}")

    try:
        sample = dataset[0]
    finally:
        setattr(dataset_module, "load_physical_aiavdataset", original_loader)
        setattr(dataset_module, "DeltaTrajectoryTokenizer", original_history_tokenizer_cls)
        setattr(dataset_module, "tokenize_history_trajectory", original_tokenize_history_trajectory)
        setattr(dataset_module, "frame_tensor_to_pil", original_frame_tensor_to_pil)

    user_content = sample["raw_prompt"][1]["content"]

    assert sample["ground_truth"] == "clip-ground-truth"
    assert sample["reward_model"] == {"ground_truth": "clip-ground-truth"}
    assert sample["extra_info"]["clip_id"] == "clip-001"
    assert sample["extra_info"]["hist_token_count"] == 48
    assert sample["raw_prompt"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
    }
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "image"
    assert user_content[0]["image"] == "image-for-frame-0"
    assert user_content[1]["image"] == "image-for-frame-1"
    assert user_content[-1]["type"] == "text"
    assert "<i0><i1><i2><i3><i4><i5>" in user_content[-1]["text"]
    assert "<|traj_history|>" not in user_content[-1]["text"]
    assert sample["raw_prompt"][2] == {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]}


def test_reward_requires_visible_cot_markers():
    assert reward_module.compute_score("alpamayo_demo", "<|cot_start|>reasoning<|cot_end|>", "") == 1.0
    assert reward_module.compute_score("alpamayo_demo", "missing markers", "") == 0.0
    assert reward_module.compute_score("alpamayo_demo", "<|cot_end|>bad order<|cot_start|>", "") == 0.0
