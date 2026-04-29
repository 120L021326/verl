from __future__ import annotations

import re
from functools import cached_property
from typing import Any

import torch


def clear_weight_name(name: str) -> str:
    """Small local replacement for cosmos_rl.utils.util.clear_weight_name."""
    for prefix in ("module.", "_orig_mod."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


class StandaloneHFModelWeightMapper:
    """Cosmos-free subset of HFModelWeightMapper needed for vLLM load_weights.

    The reference Cosmos mapper also handles distributed policy-to-rollout
    synchronization. For the demo smoke path we only need:
    - rollout-side key normalization
    - packed qkv / gate_up parameter splitting into writable tensor views
    - vocab-padding trimming for embedding/lm_head receive views
    """

    _WEIGHT_MAPPER_BACKEND_SUPPORTED = {"vllm", "trtllm"}

    def __init__(self, hf_config: Any):
        self.config = hf_config
        self.backend = "vllm"
        self.kv_head_ratio = 1
        self.head_dim = 1

        if getattr(self.config, "num_key_value_heads", None) is not None:
            self.kv_head_ratio = self.config.num_attention_heads // self.config.num_key_value_heads
            self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        elif getattr(self.config, "text_config", None) is not None:
            text_config = self.config.text_config
            self.kv_head_ratio = (
                text_config.num_attention_heads // text_config.num_key_value_heads
            )
            self.head_dim = text_config.hidden_size // text_config.num_attention_heads
        elif getattr(self.config, "llm_config", None) is not None:
            text_config = self.config.llm_config
            self.kv_head_ratio = (
                text_config.num_attention_heads // text_config.num_key_value_heads
            )
            self.head_dim = text_config.hidden_size // text_config.num_attention_heads

        self.is_vlm = getattr(self.config, "vision_config", None) is not None
        self.reverse_hf_conversion_mapping = None
        self.map_to_unsplited_weight_name: dict[str, str] = {}

    def setup_rollout_backend(self, backend: str) -> None:
        if backend not in self._WEIGHT_MAPPER_BACKEND_SUPPORTED:
            raise ValueError(f"Backend {backend} is not supported by weight mapper.")
        self.backend = backend

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = clear_weight_name(name)
        if self.is_vlm:
            if self.reverse_hf_conversion_mapping:
                for pattern, replacement in self.reverse_hf_conversion_mapping.items():
                    if re.match(pattern, name):
                        return re.sub(pattern, replacement, name)
            return name
        if name != "lm_head.weight" and not name.startswith("model."):
            name = "model." + name
        return name

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        model_type = getattr(self.config, "model_type", "")
        if model_type == "qwen3_vl":
            if rollout_weight_name.startswith("language_model.model."):
                rollout_weight_name = rollout_weight_name.replace(
                    "language_model.model.", "language_model.", 1
                )
            if (
                not rollout_weight_name.startswith("model.")
                and "lm_head" not in rollout_weight_name
            ):
                rollout_weight_name = "model." + rollout_weight_name
            if rollout_weight_name.startswith("language_model.lm_head."):
                rollout_weight_name = rollout_weight_name.replace(
                    "language_model.lm_head.", "lm_head.", 1
                )
            return rollout_weight_name
        return self.policy_map_local_key_to_hf_key(rollout_weight_name)

    def _rollout_split_qkv_weight(
        self, name: str, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "visual" in name or "vision_tower" in name:
            if weight.shape[0] % 3 != 0:
                raise ValueError(f"Visual qkv tensor has incompatible shape: {tuple(weight.shape)}")
            unit_dim = weight.shape[0] // 3
            return weight[:unit_dim], weight[unit_dim : unit_dim * 2], weight[unit_dim * 2 :]

        shares = self.kv_head_ratio + 2
        if weight.shape[0] % shares != 0:
            raise ValueError(
                f"QKV tensor has incompatible shape {tuple(weight.shape)} for shares={shares}"
            )
        unit_dim = weight.shape[0] // shares
        q_weight = weight[: unit_dim * self.kv_head_ratio]
        k_weight = weight[unit_dim * self.kv_head_ratio : unit_dim * (self.kv_head_ratio + 1)]
        v_weight = weight[unit_dim * (self.kv_head_ratio + 1) :]
        return q_weight, k_weight, v_weight

    @staticmethod
    def _split_gate_proj_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if weight.shape[0] % 2 != 0:
            raise ValueError(f"gate_up tensor has incompatible shape: {tuple(weight.shape)}")
        dim_0 = weight.shape[0]
        return weight[: dim_0 // 2], weight[dim_0 // 2 :]

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        compatible_key = self.rollout_map_local_key_to_hf_key(param_name)
        group_keys: list[tuple[str, torch.Tensor]] = []

        if ("qkv_proj" in compatible_key) or ("qkv" in compatible_key and not self.is_vlm):
            rule = "qkv_proj" if "qkv_proj" in compatible_key else "qkv"
            q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(
                compatible_key, param
            )
            group_keys.append((compatible_key.replace(rule, "q_proj"), q_weight))
            group_keys.append((compatible_key.replace(rule, "k_proj"), k_weight))
            group_keys.append((compatible_key.replace(rule, "v_proj"), v_weight))
        elif "gate_up_proj" in compatible_key:
            gate_proj_weight, up_proj_weight = self._split_gate_proj_weight(param)
            group_keys.append(
                (compatible_key.replace("gate_up_proj", "gate_proj"), gate_proj_weight)
            )
            group_keys.append((compatible_key.replace("gate_up_proj", "up_proj"), up_proj_weight))
        elif "qkv" in compatible_key:
            q_weight, k_weight, v_weight = self._rollout_split_qkv_weight(
                compatible_key, param
            )
            group_keys.append((compatible_key.replace("qkv", "q"), q_weight))
            group_keys.append((compatible_key.replace("qkv", "k"), k_weight))
            group_keys.append((compatible_key.replace("qkv", "v"), v_weight))
        else:
            group_keys.append((compatible_key, param))

        return self._trim_vocab_padding(group_keys)

    def _trim_vocab_padding(
        self, group: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None:
            return group

        trimmed: list[tuple[str, torch.Tensor]] = []
        for name, tensor in group:
            if (
                name in ("model.embed_tokens.weight", "lm_head.weight")
                and tensor.ndim == 2
                and tensor.shape[0] > vocab_size
                and tensor.shape[1] > 0
            ):
                trimmed.append((name, tensor[:vocab_size]))
            else:
                trimmed.append((name, tensor))
        return trimmed

    def rollout_prepare_recv(
        self,
        rollout_model: torch.nn.Module,
    ) -> tuple[dict[str, torch.Tensor], list[list[tuple[str, int]]]]:
        recv_key_n_rank_list: list[list[tuple[str, int]]] = []
        inplace_view_map: dict[str, torch.Tensor] = {}
        self.map_to_unsplited_weight_name = {}

        for param_name, param in rollout_model.named_parameters():
            unsplited_weight_name = self.rollout_map_local_key_to_hf_key(param_name)
            group_keys_n_params = self.rollout_split_local_key_n_param_to_hf_key_n_param(
                param_name, param
            )
            recv_key_n_rank_list.append([(key, weight.ndim) for key, weight in group_keys_n_params])
            for key, weight in group_keys_n_params:
                inplace_view_map[key] = weight
            if len(group_keys_n_params) > 1:
                for key, _ in group_keys_n_params:
                    self.map_to_unsplited_weight_name[key] = unsplited_weight_name

        return inplace_view_map, recv_key_n_rank_list

    @cached_property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        return {
            "qkv": ["q", "k", "v"],
            "gate_up_proj": ["gate_proj", "up_proj"],
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }


class ReasoningVLAWeightMapper(StandaloneHFModelWeightMapper):
    """Weight-name mapper for Alpamayo ReasoningVLA vLLM loading."""

    def __init__(self, hf_config: Any):
        llm_config = hf_config.get_llm_config() if hasattr(hf_config, "get_llm_config") else hf_config
        super().__init__(llm_config)
        self.orig_config = hf_config

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        name = clear_weight_name(name)
        for old, new in (
            ("reasoning_vla.", ""),
            ("vlm.", ""),
            ("model.language_model.", "model."),
            ("model.visual.", "visual."),
        ):
            if name.startswith(old):
                name = name.replace(old, new, 1)
        return super().policy_map_local_key_to_hf_key(name)

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        name = rollout_weight_name
        if name.startswith("llm.model."):
            name = name.replace("llm.model.", "model.", 1)
        elif name.startswith("llm.lm_head."):
            name = name.replace("llm.lm_head.", "lm_head.", 1)
        elif name.startswith("model.vlm.model.visual."):
            name = name.replace("model.vlm.model.visual.", "visual.", 1)
        elif name.startswith("vlm.model.visual."):
            name = name.replace("vlm.model.visual.", "visual.", 1)
        elif name.startswith("vlm."):
            name = name[len("vlm.") :]

        if name.startswith("language_model.model."):
            name = name.replace("language_model.model.", "model.", 1)
        elif name.startswith("language_model.lm_head."):
            name = name.replace("language_model.lm_head.", "lm_head.", 1)
        elif name.startswith("language_model."):
            name = name.replace("language_model.", "model.", 1)

        if name.startswith("visual.") and ".attn.qkv_proj." in name:
            name = name.replace(".attn.qkv_proj.", ".attn.qkv.", 1)

        return self.policy_map_local_key_to_hf_key(name)

