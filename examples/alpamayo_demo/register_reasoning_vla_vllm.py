from __future__ import annotations

"""Registration helpers for loading ReasoningVLA checkpoints with vLLM."""

from typing import Any


def reasoning_vla_hf_overrides(cfg: Any) -> Any:
    """Make vLLM route an Alpamayo ReasoningVLA config to the local wrapper."""
    if hasattr(cfg, "get_llm_config"):
        setattr(cfg, "text_config", cfg.get_llm_config())
    architectures = list(getattr(cfg, "architectures", []) or [])
    for name in ("ReasoningVLA", "REASONING_VLA"):
        if name not in architectures:
            architectures.append(name)
    setattr(cfg, "architectures", architectures)
    return cfg


def register_reasoning_vla_for_vllm() -> None:
    """Register the local ReasoningVLA vLLM wrapper.

    This function must run in the same process that constructs the vLLM LLM
    engine.
    """
    from transformers import AutoConfig
    from vllm import ModelRegistry

    try:
        from reasoning_vla.config import RLWrapperReasoningVLAConfig
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import reasoning_vla.config.RLWrapperReasoningVLAConfig. "
            "Add alpamayo/finetune/rl/models to PYTHONPATH before constructing vLLM."
        ) from exc

    try:
        AutoConfig.register("alpamayo_reasoning_vla", RLWrapperReasoningVLAConfig)
    except ValueError as exc:
        if "already" not in str(exc).lower() and "exist" not in str(exc).lower():
            raise

    try:
        from examples.alpamayo_demo.reasoning_vla_vllm_wrapper import ReasoningVLAModelForVLLM
    except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
        from reasoning_vla_vllm_wrapper import ReasoningVLAModelForVLLM

    for arch in ("ReasoningVLA", "REASONING_VLA"):
        try:
            ModelRegistry.register_model(arch, ReasoningVLAModelForVLLM)
        except Exception as exc:
            # vLLM may raise if a name is already registered in a reused process.
            if "already" not in str(exc).lower() and "exist" not in str(exc).lower():
                raise
