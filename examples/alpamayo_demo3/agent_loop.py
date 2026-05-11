from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids


class AlpamayoPrefillAgentLoop(SingleTurnAgentLoop):
    """Single-turn Alpamayo rollout that continues the final assistant message."""

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        apply_kwargs: dict[str, Any] = dict(self.apply_chat_template_kwargs)
        apply_kwargs.pop("add_generation_prompt", None)
        apply_kwargs.pop("continue_final_message", None)

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False,
                    **apply_kwargs,
                ),
            )

            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=True,
                    **apply_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids
