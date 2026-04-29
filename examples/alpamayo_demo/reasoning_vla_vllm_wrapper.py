from __future__ import annotations

"""vLLM wrapper for loading Alpamayo ReasoningVLA checkpoints in verl demos."""

import logging
from typing import Iterable, Optional, Union

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from vllm.config import MultiModalConfig, VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsMRoPE,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

try:
    from vllm.v1.sample.metadata import SamplingMetadata
except Exception:  # pragma: no cover - vLLM version compatibility
    SamplingMetadata = None

try:
    from examples.alpamayo_demo.reasoning_vla_vllm_weight_mapper import ReasoningVLAWeightMapper
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from reasoning_vla_vllm_weight_mapper import ReasoningVLAWeightMapper


class ReasoningVLAModelForVLLM(torch.nn.Module, SupportsMultiModal):
    """vLLM model wrapper that hosts the VLM module of ReasoningVLA."""

    merge_by_field_config = True
    supports_mrope = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        orig_cfg = vllm_config.model_config.hf_config
        if not hasattr(orig_cfg, "get_llm_config"):
            raise TypeError(
                "ReasoningVLAModelForVLLM expects a ReasoningVLA config with get_llm_config()."
            )
        llm_cfg = orig_cfg.get_llm_config()

        self._orig_hf_config_for_mapper = orig_cfg

        if hasattr(llm_cfg, "get_text_config"):
            llm_text_cfg = llm_cfg.get_text_config()
        else:
            llm_text_cfg = getattr(llm_cfg, "text_config", llm_cfg)

        ckpt_vocab = getattr(orig_cfg, "vocab_size", None)
        if ckpt_vocab is not None:
            if getattr(llm_cfg, "text_config", None) is not None:
                llm_cfg.text_config.vocab_size = ckpt_vocab
                llm_cfg.text_config.pad_vocab_size_multiple = 1
            llm_cfg.vocab_size = ckpt_vocab
            llm_cfg.pad_vocab_size_multiple = 1
            vllm_config.model_config.vocab_size = ckpt_vocab

        model_config = vllm_config.model_config
        model_config.hf_config = llm_cfg
        model_config.hf_text_config = llm_text_cfg
        model_config.task = "generate"
        model_config.multimodal_config = (
            getattr(model_config, "multimodal_config", None) or MultiModalConfig()
        )

        self.vlm = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "vlm"),
            architectures=["Qwen3VLForConditionalGeneration"],
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        del i
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids and inputs_embeds cannot be None at the same time")
        return self.vlm(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states, sampling_metadata=None):
        try:
            if sampling_metadata is None:
                return self.vlm.compute_logits(hidden_states)
            return self.vlm.compute_logits(hidden_states, sampling_metadata)
        except TypeError:
            return self.vlm.compute_logits(hidden_states)

    def get_language_model(self):
        return self.vlm.get_language_model()
    
    def get_input_embeddings(
        self, input_ids: torch.Tensor, multimodal_embeddings=None
    ) -> torch.Tensor:
        return self.vlm.get_input_embeddings(input_ids, multimodal_embeddings)

    def embed_multimodal(self, **kwargs: object):
        """Compute multimodal embeddings across vLLM method-name versions."""
        kwargs = {
            key: self._move_grid_tensors_to_cpu(value)
            if key.endswith("grid_thw") or isinstance(value, (dict, list, tuple))
            else value
            for key, value in kwargs.items()
        }
        if hasattr(self.vlm, "embed_multimodal"):
            return self.vlm.embed_multimodal(**kwargs)
        if hasattr(self.vlm, "get_multimodal_embeddings"):
            return self.vlm.get_multimodal_embeddings(**kwargs)
        raise AttributeError(
            f"{self.vlm.__class__.__name__} exposes neither embed_multimodal() "
            "nor get_multimodal_embeddings()."
        )

    def get_multimodal_embeddings(self, **kwargs: object):
        return self.embed_multimodal(**kwargs)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Delegate M-RoPE position construction to the underlying Qwen3VL model."""
        return self.vlm.get_mrope_input_positions(input_tokens, mm_features)
    
    @staticmethod
    def _move_grid_tensors_to_cpu(value: object) -> object:
        """Keep Qwen3-VL grid metadata on CPU for visual code that calls numpy()."""
        if isinstance(value, torch.Tensor):
            return value.cpu() if value.is_cuda else value
        if isinstance(value, dict):
            return {
                key: ReasoningVLAModelForVLLM._move_grid_tensors_to_cpu(item)
                if key.endswith("grid_thw")
                else item
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                ReasoningVLAModelForVLLM._move_grid_tensors_to_cpu(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                ReasoningVLAModelForVLLM._move_grid_tensors_to_cpu(item)
                if isinstance(item, dict)
                else item
                for item in value
            )
        return value

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = ReasoningVLAWeightMapper(self._orig_hf_config_for_mapper)
        mapper.setup_rollout_backend("vllm")
        inplace_map, _ = mapper.rollout_prepare_recv(self.vlm)

        def normalize(hf_name: str) -> list[str]:
            name = hf_name.replace("language_model.", "")
            candidates: list[str] = [name]
            if name.startswith("vlm."):
                candidates.append(name.replace("vlm.", "", 1))
                candidates.append(name.replace("vlm.model.", "model."))
                if name.startswith("vlm.model.visual."):
                    candidates.append(name.replace("vlm.model.visual.", "visual."))
                if name.startswith("vlm.visual."):
                    candidates.append(name.replace("vlm.visual.", "visual."))
            if name.startswith("model."):
                candidates.append(f"vlm.{name}")
                candidates.append(name.replace("model.", "vlm.model."))
                if name.startswith("model.visual."):
                    candidates.append(name.replace("model.visual.", "visual."))
            if name.startswith("visual."):
                candidates.append(f"vlm.model.{name}")
                candidates.append(f"model.{name}")
                candidates.append(f"vlm.{name}")

            more: list[str] = []
            for candidate in candidates:
                idx = candidate.find("visual.")
                if idx != -1:
                    more.append(f"vlm.{candidate[idx:]}")
            candidates.extend(item for item in more if item not in candidates)

            extra: list[str] = []
            for candidate in list(candidates):
                if ".attn.proj." in candidate:
                    extra.append(candidate.replace(".attn.proj.", ".attn.out_proj."))
                if ".attn.out_proj." in candidate:
                    extra.append(candidate.replace(".attn.out_proj.", ".attn.proj."))
                if ".attn.qkv_proj." in candidate:
                    extra.append(candidate.replace(".attn.qkv_proj.", ".attn.qkv."))
                if ".attn.qkv." in candidate:
                    extra.append(candidate.replace(".attn.qkv.", ".attn.qkv_proj."))
            candidates.extend(item for item in extra if item not in candidates)

            result: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    result.append(candidate)

            if "qkv" in name and "visual" in name:
                result.append(name.replace("vlm.model.visual.", "visual.", 1).replace("qkv", "q"))
                result.append(name.replace("vlm.model.visual.", "visual.", 1).replace("qkv", "k"))
                result.append(name.replace("vlm.model.visual.", "visual.", 1).replace("qkv", "v"))
            return result

        unused_ckpt_keys: set[str] = set()
        for raw_name, tensor in weights:
            copied = False
            for key in normalize(raw_name):
                dst = inplace_map.get(key)
                if dst is None:
                    continue
                target = dst if isinstance(dst, torch.Tensor) else dst()

                if "qkv" in raw_name and "qkv" not in key:
                    if ".q." in key:
                        target.data.copy_(
                            tensor[: tensor.shape[0] // 3].to(
                                dtype=target.dtype, device=target.device
                            )
                        )
                        copied = True
                    elif ".k." in key:
                        target.data.copy_(
                            tensor[tensor.shape[0] // 3 : tensor.shape[0] // 3 * 2].to(
                                dtype=target.dtype, device=target.device
                            )
                        )
                        copied = True
                    elif ".v." in key:
                        target.data.copy_(
                            tensor[tensor.shape[0] // 3 * 2 :].to(
                                dtype=target.dtype, device=target.device
                            )
                        )
                        copied = True
                    if copied:
                        continue

                if target.shape != tensor.shape:
                    continue
                target.data.copy_(tensor.to(dtype=target.dtype, device=target.device))
                copied = True
                break

            if not copied:
                unused_ckpt_keys.add(raw_name)

        logging.info("[ReasoningVLAModelForVLLM] Unused checkpoint keys: %s", unused_ckpt_keys)
        return {name for name, _ in self.named_parameters()}


class ReasoningVLAProcessingInfo(Qwen3VLProcessingInfo):
    """Processing info that resolves the underlying Qwen3-VL config."""

    def get_hf_config(self):
        try:
            return self.ctx.get_hf_config(Qwen3VLConfig)
        except TypeError:
            cfg = self.ctx.model_config.hf_config
            if hasattr(cfg, "get_llm_config"):
                return cfg.get_llm_config()
            return super().get_hf_config()


MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ReasoningVLAProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)(ReasoningVLAModelForVLLM)
