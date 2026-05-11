from __future__ import annotations

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

TRAJ_FUTURE_START_TOKEN = "<|traj_future_start|>"
TRAJ_FUTURE_END_TOKEN = "<|traj_future_end|>"


def _token_ids(tokenizer, token: str) -> list[int]:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    return [int(token_id) for token_id in token_ids]


class AlpamayoSpecialTokenRewardManager(RewardManagerBase):
    """Reward manager for Alpamayo text that must preserve generated special tokens."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["response_token_ids"] = [int(token_id) for token_id in valid_response_ids.tolist()]
        extra_info["traj_future_start_token_ids"] = _token_ids(self.tokenizer, TRAJ_FUTURE_START_TOKEN)
        extra_info["traj_future_end_token_ids"] = _token_ids(self.tokenizer, TRAJ_FUTURE_END_TOKEN)

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
