# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import transformers.models.qwen3_vl.modeling_qwen3_vl as hf_qwen3_vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)

from verl.utils.transformers_compat import unpack_visual_output

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _is_qwen3_vl_debug_enabled() -> bool:
    return os.getenv("VERL_DEBUG_QWEN3_VL", "0").lower() in {"1", "true", "yes"}


def _get_pid_rank() -> tuple[int, Optional[int]]:
    rank = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    return os.getpid(), rank


def _shape_or_none(value):
    if value is None:
        return None
    return tuple(value.shape) if hasattr(value, "shape") else type(value).__name__


def _tensor_preview(value: Optional[torch.Tensor], limit: int = 8):
    if value is None or not hasattr(value, "shape"):
        return None
    if value.numel() == 0:
        return []
    flat = value.detach().reshape(-1)
    limit = min(limit, flat.numel())
    return flat[:limit].cpu().tolist()


def _stride_or_none(value):
    if value is None or not hasattr(value, "stride"):
        return None
    return tuple(value.stride())


def _should_force_rotary_log(x, position_ids=None, cos=None) -> bool:
    x_shape = _shape_or_none(x)
    pos_shape = _shape_or_none(position_ids)
    cos_shape = _shape_or_none(cos)

    if pos_shape is not None and isinstance(pos_shape, tuple) and len(pos_shape) >= 1:
        if pos_shape[-1] <= 16:
            return True
    if cos_shape is not None and isinstance(cos_shape, tuple) and len(cos_shape) >= 1:
        if cos_shape[-1] <= 16 or (len(cos_shape) >= 2 and cos_shape[1] <= 16):
            return True
    if (
        x_shape is not None
        and pos_shape is not None
        and isinstance(x_shape, tuple)
        and isinstance(pos_shape, tuple)
        and len(x_shape) >= 2
        and len(pos_shape) >= 1
        and x_shape[1] != pos_shape[-1]
    ):
        return True
    if (
        x_shape is not None
        and cos_shape is not None
        and isinstance(x_shape, tuple)
        and isinstance(cos_shape, tuple)
        and len(x_shape) >= 2
        and len(cos_shape) >= 2
        and x_shape[1] != cos_shape[1]
    ):
        return True
    return False


def _maybe_patch_rotary_debug(language_model) -> None:
    if getattr(language_model, "_verl_rotary_debug_patched", False):
        return

    original_rotary_forward = language_model.rotary_emb.forward

    @functools.wraps(original_rotary_forward)
    def rotary_forward_with_debug(*args, **kwargs):
        position_ids = kwargs.get("position_ids")
        x = kwargs.get("x")
        if x is None and len(args) >= 1:
            x = args[0]
        if position_ids is None and len(args) >= 2:
            position_ids = args[1]

        pid, rank = _get_pid_rank()
        logger.warning(
            "Qwen3-VL rotary debug enter: pid=%s rank=%s x=%s x_preview=%s position_ids=%s position_ids_preview=%s",
            pid,
            rank,
            _shape_or_none(x),
            _tensor_preview(x),
            _shape_or_none(position_ids),
            _tensor_preview(position_ids),
        )
        if _should_force_rotary_log(x, position_ids=position_ids):
            enter_message = (
                "Qwen3-VL rotary debug suspicious enter: "
                f"pid={pid} rank={rank} "
                f"x={_shape_or_none(x)} x_stride={_stride_or_none(x)} "
                f"position_ids={_shape_or_none(position_ids)} position_ids_stride={_stride_or_none(position_ids)} "
                f"x_preview={_tensor_preview(x)} position_ids_preview={_tensor_preview(position_ids)}"
            )
            logger.error(enter_message)
            print(enter_message, file=sys.stderr, flush=True)
        cos, sin = original_rotary_forward(*args, **kwargs)
        logger.warning(
            "Qwen3-VL rotary debug exit: pid=%s rank=%s cos=%s sin=%s cos_preview=%s",
            pid,
            rank,
            _shape_or_none(cos),
            _shape_or_none(sin),
            _tensor_preview(cos),
        )
        if _should_force_rotary_log(x, position_ids=position_ids, cos=cos):
            exit_message = (
                "Qwen3-VL rotary debug suspicious exit: "
                f"pid={pid} rank={rank} "
                f"x={_shape_or_none(x)} x_stride={_stride_or_none(x)} "
                f"position_ids={_shape_or_none(position_ids)} position_ids_stride={_stride_or_none(position_ids)} "
                f"cos={_shape_or_none(cos)} cos_stride={_stride_or_none(cos)} "
                f"sin={_shape_or_none(sin)} sin_stride={_stride_or_none(sin)} "
                f"position_ids_preview={_tensor_preview(position_ids)} cos_preview={_tensor_preview(cos)}"
            )
            logger.error(exit_message)
            print(exit_message, file=sys.stderr, flush=True)
        return cos, sin

    language_model.rotary_emb.forward = rotary_forward_with_debug
    language_model._verl_rotary_debug_patched = True

    if not getattr(hf_qwen3_vl, "_verl_apply_rotary_debug_patched", False):
        original_apply_rotary = hf_qwen3_vl.apply_rotary_pos_emb

        @functools.wraps(original_apply_rotary)
        def apply_rotary_pos_emb_with_debug(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            pid, rank = _get_pid_rank()
            logger.warning(
                "Qwen3-VL apply_rotary debug enter: pid=%s rank=%s q=%s k=%s cos=%s sin=%s q_preview=%s cos_preview=%s unsqueeze_dim=%s",
                pid,
                rank,
                _shape_or_none(q),
                _shape_or_none(k),
                _shape_or_none(cos),
                _shape_or_none(sin),
                _tensor_preview(q),
                _tensor_preview(cos),
                unsqueeze_dim,
            )
            try:
                return original_apply_rotary(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
            except RuntimeError:
                error_message = (
                    "Qwen3-VL apply_rotary debug error: "
                    f"pid={pid} rank={rank} "
                    f"q={_shape_or_none(q)} q_stride={_stride_or_none(q)} "
                    f"k={_shape_or_none(k)} k_stride={_stride_or_none(k)} "
                    f"cos={_shape_or_none(cos)} cos_stride={_stride_or_none(cos)} "
                    f"sin={_shape_or_none(sin)} sin_stride={_stride_or_none(sin)} "
                    f"position_ids={_shape_or_none(position_ids)} "
                    f"position_ids_preview={_tensor_preview(position_ids)} "
                    f"q_preview={_tensor_preview(q)} "
                    f"cos_preview={_tensor_preview(cos)} "
                    f"unsqueeze_dim={unsqueeze_dim}"
                )
                logger.error(error_message)
                print(error_message, file=sys.stderr, flush=True)
                raise

        hf_qwen3_vl.apply_rotary_pos_emb = apply_rotary_pos_emb_with_debug
        hf_qwen3_vl._verl_apply_rotary_debug_patched = True


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    # Since we use timestamps to separate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0 because llm_grid_t is always 1
            # (we use timestamps to encode the temporal information for videos)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def _get_input_embeds(
    model: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, video_mask = None, None
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, deepstack_image_embeds = unpack_visual_output(model.visual(pixel_values, grid_thw=image_grid_thw))
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if _is_qwen3_vl_debug_enabled():
            logger.warning(
                "Qwen3-VL forward debug image alignment: input_ids=%s pixel_values=%s image_grid_thw=%s n_image_tokens=%s n_image_features=%s",
                tuple(input_ids.shape),
                _shape_or_none(pixel_values),
                _shape_or_none(image_grid_thw),
                n_image_tokens,
                n_image_features,
            )
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = unpack_visual_output(
            model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        )
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = unpack_visual_output(
            model.visual(pixel_values, grid_thw=image_grid_thw)
        )
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds = inputs_embeds + 0.0 * emb.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


@dataclass
class Qwen3VLCausalLMOutputForPPO(Qwen3VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def qwen3_vl_base_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    kwargs.update(input_kwargs)
    if _is_qwen3_vl_debug_enabled():
        _maybe_patch_rotary_debug(self.language_model)
        pid, rank = _get_pid_rank()
    if _is_qwen3_vl_debug_enabled():
        logger.warning(
            "Qwen3-VL forward debug before language_model: pid=%s rank=%s input_ids=%s attention_mask=%s position_ids=%s position_ids_preview=%s pixel_values=%s pixel_values_videos=%s image_grid_thw=%s video_grid_thw=%s",
            pid,
            rank,
            tuple(input_ids.shape) if input_ids is not None else None,
            _shape_or_none(attention_mask),
            _shape_or_none(kwargs.get("position_ids")),
            _tensor_preview(kwargs.get("position_ids")),
            _shape_or_none(pixel_values),
            _shape_or_none(pixel_values_videos),
            _shape_or_none(image_grid_thw),
            _shape_or_none(video_grid_thw),
        )
    return self.language_model(
        input_ids=None,
        **kwargs,
    )


def forward_with_normal_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def patch_qwen3_vl_moe_sparse_moe_block_forward():
    """
    Monkey patch to fix a bug in transformers 4.57.3 where Qwen3VLMoeTextSparseMoeBlock.forward
    incorrectly uses torch.zeros_like(hidden_states) instead of torch.zeros_like(router_logits)
    when creating router_weights (line 148 in modeling_qwen3_vl_moe.py).

    This is a minimal fix that only changes the problematic line while keeping the rest of the
    original implementation intact.
    """
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
    except ImportError:
        # Model not available, skip patching
        return

    # Store the original forward method for reference
    original_forward = Qwen3VLMoeTextSparseMoeBlock.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # BUG FIX: Original code incorrectly uses hidden_states here, should use router_logits
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

    # Apply the patch
    Qwen3VLMoeTextSparseMoeBlock.forward = patched_forward
    logger.info("Monkey patched Qwen3VLMoeTextSparseMoeBlock.forward to fix router_weights bug")
