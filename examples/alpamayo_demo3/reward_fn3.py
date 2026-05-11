from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import aiohttp
import torch

COT_START_TOKEN = "<|cot_start|>"
COT_END_TOKEN = "<|cot_end|>"
TRAJ_FUTURE_START_TOKEN = "<|traj_future_start|>"
TRAJ_FUTURE_END_TOKEN = "<|traj_future_end|>"
TRAJ_TOKEN_PATTERN = re.compile(r"<i(\d+)>")
TRAJ_VOCAB_SIZE = 3000
TRAJ_TOKEN_ID_BASE = 151669
TRAJ_TOKEN_ID_END = TRAJ_TOKEN_ID_BASE + TRAJ_VOCAB_SIZE - 1
REASON_JUDGE_ENDPOINT = "v1/chat/completions"
REASON_JUDGE_MAX_RETRIES = 3
MAX_ABS_MAG_JERK: Final[float] = 8.37
MAX_ABS_LAT_ACCEL: Final[float] = 4.89
MAX_LON_ACCEL: Final[float] = 2.40
MIN_LON_ACCEL: Final[float] = -4.05
MAX_ABS_YAW_ACCEL: Final[float] = 1.93
MAX_ABS_LON_JERK: Final[float] = 4.13
MAX_ABS_YAW_RATE: Final[float] = 0.95
PLANNING_FREQ: Final[float] = 10.0
COMFORT_METRIC_CONFIG_DICT = {
    "comfort/lon_accel": ("ego_dv_lon", MIN_LON_ACCEL, MAX_LON_ACCEL),
    "comfort/lat_accel": ("ego_dv_lat", -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
    "comfort/jerk": ("ego_jerk", -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
    "comfort/lon_jerk": ("ego_jerk_lon", -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
    "comfort/yaw_accel": ("ego_yaw_accel", -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
    "comfort/yaw_rate": ("ego_yaw_rate", -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
}


def has_visible_cot_markers(solution_str: str) -> bool:
    cot_start_index = solution_str.find(COT_START_TOKEN)
    cot_end_index = solution_str.find(COT_END_TOKEN)
    if cot_end_index < 0:
        return False
    # The prompt may already end with <|cot_start|>, so the generated response
    # can validly start directly with the CoT body and only include <|cot_end|>.
    return cot_start_index < 0 or cot_end_index > cot_start_index


def has_required_response_format(solution_str: str) -> bool:
    cot_start_index = solution_str.find(COT_START_TOKEN)
    cot_end_index = solution_str.find(COT_END_TOKEN)
    future_start_index = solution_str.find(TRAJ_FUTURE_START_TOKEN)
    future_end_index = solution_str.find(TRAJ_FUTURE_END_TOKEN)
    if cot_end_index < 0 or future_start_index < 0 or future_end_index < 0:
        return False
    if cot_start_index >= 0 and cot_start_index > cot_end_index:
        return False
    return cot_end_index < future_start_index < future_end_index


def extract_reasoning_trace(solution_str: str) -> str:
    cot_start_index = solution_str.find(COT_START_TOKEN)
    if cot_start_index >= 0:
        start_index = cot_start_index + len(COT_START_TOKEN)
    else:
        # AlpamayoPrefillAgentLoop continues an assistant message that already
        # ends with <|cot_start|>, so the decoded response may begin with the
        # reasoning body directly.
        start_index = 0

    cot_end_index = solution_str.find(COT_END_TOKEN, start_index)
    if cot_end_index < 0:
        return ""
    return solution_str[start_index:cot_end_index].strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _resolve_gt_reasoning(ground_truth: Any, extra_info: dict[str, Any]) -> str:
    text = _normalize_text(ground_truth)
    if text:
        return text
    for key in ("ground_truth", "gt_cot", "coc", "cot", "reasoning"):
        text = _normalize_text(extra_info.get(key))
        if text:
            return text
    return ""


def _build_reasoning_judge_prompt(pred_reasoning: str, gt_reasoning: str, extra_info: dict[str, Any]) -> str:
    context_keys = (
        "clip_id",
        "t0_us",
    )
    context_lines = []
    for key in context_keys:
        value = _normalize_text(extra_info.get(key))
        if value:
            context_lines.append(f"{key}: {value}")
    sample_context = "\nSample metadata:\n" + "\n".join(context_lines) + "\n" if context_lines else ""

    return f"""You are an expert evaluator for autonomous driving reasoning traces.
The reasoning trace describes what the ego vehicle should be doing and the reasons and factors that lead to the behavior.
Score how well PRED aligns with GT in terms of behavior consistency and causal reasoning quality.

Scoring rubric:
5: Behavior and causal reasoning are fully consistent.
4: Behavior is correct; causal reasoning is mostly consistent.
3: Behavior is roughly correct, but reasoning is incomplete or slightly incorrect.
2: Behavior is partially incorrect or reasoning is largely inconsistent.
1: Behavior is wrong or contradicts GT.
0: Completely unrelated or opposite.
{sample_context}
GT:
{gt_reasoning}

PRED:
{pred_reasoning}

Return only a JSON object with this schema:
{{"score": <integer from 0 to 5>, "reason": "<brief explanation>"}}"""


def _extract_judge_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if choices:
        first_choice = choices[0]
        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = first_choice.get("text")
        if isinstance(text, str):
            return text
    data = response_json.get("data") or []
    if data and isinstance(data[0], dict):
        text = data[0].get("text") or data[0].get("content")
        if isinstance(text, str):
            return text
    return json.dumps(response_json)


def _parse_reasoning_judge_score(text: str) -> tuple[float, str]:
    # print(f"raw reason: {text}")
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match is not None:
        try:
            payload = json.loads(json_match.group(0))
            score = float(payload["score"])
            reason = _normalize_text(payload.get("reason"))
            return min(5.0, max(0.0, score)), reason
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    score_match = re.search(r'"?score"?\s*[:=]\s*([0-5](?:\.\d+)?)', text, flags=re.IGNORECASE)
    if score_match is None:
        score_match = re.search(r"\b([0-5](?:\.\d+)?)\b", text)
    if score_match is None:
        raise ValueError(f"could not parse judge score from response: {text[:500]}")
    score = float(score_match.group(1))
    return min(5.0, max(0.0, score)), text.strip()[:500]


async def _post_reward_model(
    reward_router_address: str,
    endpoint: str,
    payload: dict[str, Any],
    max_retries: int = REASON_JUDGE_MAX_RETRIES,
) -> dict[str, Any]:
    url = f"http://{reward_router_address}/{endpoint.lstrip('/')}"
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json(content_type=None)
        except Exception as exc:
            last_exception = exc
            if attempt == max_retries - 1:
                break
    assert last_exception is not None
    raise last_exception


async def compute_reasoning_quality_reward(
    pred_reasoning: str,
    gt_reasoning: str,
    extra_info: dict[str, Any],
    *,
    reward_router_address: str | None,
    reason_judge_endpoint: str = REASON_JUDGE_ENDPOINT,
    reason_judge_max_tokens: int = 256,
) -> dict[str, Any]:
    if not pred_reasoning:
        return {
            "reasoning_quality_score": 0.0,
            "reasoning_quality_reward": 0.0,
            "reasoning_judge_reason": "missing predicted reasoning trace",
            "reasoning_judge_error": "",
        }
    if not gt_reasoning:
        return {
            "reasoning_quality_score": 0.0,
            "reasoning_quality_reward": 0.0,
            "reasoning_judge_reason": "missing ground-truth reasoning trace",
            "reasoning_judge_error": "",
        }
    if not reward_router_address:
        return {
            "reasoning_quality_score": 0.0,
            "reasoning_quality_reward": 0.0,
            "reasoning_judge_reason": "reward model router is not enabled",
            "reasoning_judge_error": "",
        }

    prompt = _build_reasoning_judge_prompt(pred_reasoning, gt_reasoning, extra_info)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(reason_judge_max_tokens),
    }
    response_json = await _post_reward_model(reward_router_address, reason_judge_endpoint, payload)
    judge_text = _extract_judge_text(response_json)
    score, reason = _parse_reasoning_judge_score(judge_text)
    return {
        "reasoning_quality_score": score,
        "reasoning_quality_reward": score / 5.0,
        "reasoning_judge_reason": reason,
        "reasoning_judge_error": "",
    }


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_trajectory_xyz(value: Any, name: str) -> torch.Tensor:
    tensor = _as_float_tensor(value)
    if tensor.ndim == 4:
        tensor = tensor[:, -1]
    if tensor.ndim == 3:
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor[-1]
    if tensor.ndim != 2 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must resolve to [T, 3], got shape {tuple(tensor.shape)}")
    return tensor


def _as_trajectory_rot(value: Any, name: str) -> torch.Tensor:
    tensor = _as_float_tensor(value)
    if tensor.ndim == 5:
        tensor = tensor[:, -1]
    if tensor.ndim == 4:
        tensor = tensor[0] if tensor.shape[0] == 1 else tensor[-1]
    if tensor.ndim != 3 or tensor.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must resolve to [T, 3, 3], got shape {tuple(tensor.shape)}")
    return tensor


def _as_history_xyz(value: Any, name: str) -> torch.Tensor:
    tensor = _as_float_tensor(value)
    if tensor.ndim == 4:
        tensor = tensor[:, -1]
    elif tensor.ndim == 3 and tensor.shape[0] != 1:
        tensor = tensor[-1:].contiguous()
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must resolve to [B, T, 3], got shape {tuple(tensor.shape)}")
    return tensor


def _as_history_rot(value: Any, name: str) -> torch.Tensor:
    tensor = _as_float_tensor(value)
    if tensor.ndim == 5:
        tensor = tensor[:, -1]
    elif tensor.ndim == 4 and tensor.shape[0] != 1:
        tensor = tensor[-1:].contiguous()
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must resolve to [B, T, 3, 3], got shape {tuple(tensor.shape)}")
    return tensor


def calculate_ade(pred_trajectory: torch.Tensor, gt_trajectory: torch.Tensor) -> float:
    if pred_trajectory.shape != gt_trajectory.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred_trajectory.shape)} vs gt {tuple(gt_trajectory.shape)}")
    pred_xy = pred_trajectory[..., :2]
    gt_xy = gt_trajectory[..., :2]
    distances = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    return float(distances.mean().item())


def _diff(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-1] < 2:
        return torch.zeros_like(tensor)
    delta = tensor[..., 1:] - tensor[..., :-1]
    last_delta = delta[..., -1:].clone()
    return torch.cat([delta, last_delta], dim=-1) * PLANNING_FREQ


def _diff_yaw(yaw: torch.Tensor) -> torch.Tensor:
    if yaw.shape[-1] < 2:
        return torch.zeros_like(yaw)
    yaw_diff = torch.diff(yaw, dim=-1)
    yaw_diff = torch.where(yaw_diff > torch.pi, yaw_diff - 2 * torch.pi, yaw_diff)
    yaw_diff = torch.where(yaw_diff < -torch.pi, yaw_diff + 2 * torch.pi, yaw_diff)
    yaw_rate = yaw_diff * PLANNING_FREQ
    last_yaw_rate = yaw_rate[..., -1:].clone()
    return torch.cat((yaw_rate, last_yaw_rate), dim=-1)


def _within_bound(metric: torch.Tensor, min_bound: float, max_bound: float) -> torch.Tensor:
    metric_within_bound = (metric > min_bound) & (metric < max_bound)
    return torch.all(metric_within_bound, axis=-1).float()


def gather_dynamics(pred_xyz: torch.Tensor, pred_rot: torch.Tensor) -> dict[str, torch.Tensor]:
    ego_x = pred_xyz[..., 0]
    ego_y = pred_xyz[..., 1]
    ego_h = torch.atan2(pred_rot[..., 1, 0], pred_rot[..., 0, 0])
    ego_dx = _diff(ego_x)
    ego_dy = _diff(ego_y)
    ego_yaw_rate = ego_dh = _diff_yaw(ego_h)
    ego_v = torch.linalg.norm(torch.stack([ego_dx, ego_dy], dim=-1), dim=-1)
    ego_dv = _diff(ego_v)
    ego_jerk = _diff(ego_dv)
    ego_yaw_accel = _diff(ego_dh)
    ego_v_lon = ego_dx * torch.cos(ego_h) + ego_dy * torch.sin(ego_h)
    ego_dv_lon = _diff(ego_v_lon)
    ego_jerk_lon = _diff(ego_dv_lon)
    ego_dv_lat = _diff(-ego_dx * torch.sin(ego_h) + ego_dy * torch.cos(ego_h))
    return {
        "ego_yaw_rate": ego_yaw_rate,
        "ego_dv": ego_dv,
        "ego_jerk": ego_jerk,
        "ego_yaw_accel": ego_yaw_accel,
        "ego_dv_lon": ego_dv_lon,
        "ego_jerk_lon": ego_jerk_lon,
        "ego_dv_lat": ego_dv_lat,
    }


def compute_comfort(pred_xyz: torch.Tensor, pred_rot: torch.Tensor) -> dict[str, torch.Tensor]:
    ego_dynamics = gather_dynamics(pred_xyz, pred_rot)
    comfort_metric_dict = {}
    for name, (dyn_key, lo, hi) in COMFORT_METRIC_CONFIG_DICT.items():
        comfort_metric_dict[name] = _within_bound(ego_dynamics[dyn_key], lo, hi).mean(2)
    for name in comfort_metric_dict:
        comfort_metric_dict[name] = comfort_metric_dict[name].mean(dim=-1)
    return comfort_metric_dict


def _extract_future_segment(solution_str: str) -> tuple[str, bool]:
    start_index = solution_str.find(TRAJ_FUTURE_START_TOKEN)
    if start_index < 0:
        raise ValueError(f"response is missing {TRAJ_FUTURE_START_TOKEN}")
    start_index += len(TRAJ_FUTURE_START_TOKEN)
    end_index = solution_str.find(TRAJ_FUTURE_END_TOKEN, start_index)
    if end_index < 0:
        return solution_str[start_index:], True
    return solution_str[start_index:end_index], False


def _find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    last_start = len(values) - len(pattern)
    for index in range(start, last_start + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def _extract_future_token_ids_from_response_ids(extra_info: dict[str, Any]) -> list[int]:
    response_token_ids = [int(token_id) for token_id in extra_info.get("response_token_ids", [])]
    future_start_token_ids = [int(token_id) for token_id in extra_info.get("traj_future_start_token_ids", [])]
    future_end_token_ids = [int(token_id) for token_id in extra_info.get("traj_future_end_token_ids", [])]

    start_index = _find_subsequence(response_token_ids, future_start_token_ids)
    if start_index < 0:
        return []

    scan_start = start_index + len(future_start_token_ids)
    end_index = _find_subsequence(response_token_ids, future_end_token_ids, start=scan_start)
    scan_end = end_index if end_index >= 0 else len(response_token_ids)

    token_ids = []
    for token_id in response_token_ids[scan_start:scan_end]:
        if TRAJ_TOKEN_ID_BASE <= token_id <= TRAJ_TOKEN_ID_END:
            token_ids.append(token_id - TRAJ_TOKEN_ID_BASE)
    return token_ids


def _extract_future_token_ids(solution_str: str, extra_info: dict[str, Any]) -> tuple[list[int], bool, str]:
    future_segment, missing_future_end = _extract_future_segment(solution_str)

    if missing_future_end:
        token_ids_from_response = _extract_future_token_ids_from_response_ids(extra_info)
        if token_ids_from_response:
            return token_ids_from_response, True, "response_token_ids_missing_future_end"

    token_ids = []
    for raw_token_id in TRAJ_TOKEN_PATTERN.findall(future_segment):
        token_id = int(raw_token_id)
        if 0 <= token_id < TRAJ_VOCAB_SIZE:
            token_ids.append(token_id)
    decode_source = "decoded_traj_future_tokens_missing_future_end" if missing_future_end else "decoded_traj_future_tokens"
    return token_ids, missing_future_end, decode_source


def _pad_or_truncate_tokens(token_ids: list[int], expected_len: int) -> list[int]:
    if len(token_ids) < expected_len:
        return token_ids + [0] * (expected_len - len(token_ids))
    return token_ids[:expected_len]


def _ensure_alpamayo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    alpamayo_src = repo_root / "alpamayo" / "src"
    if alpamayo_src.exists() and str(alpamayo_src) not in sys.path:
        sys.path.insert(0, str(alpamayo_src))


def _load_config_json(model_path: str) -> SimpleNamespace:
    config_path = Path(model_path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model config does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    # Some copied Hugging Face pages include page chrome around the raw JSON.
    # Keep this fallback narrow: parse from the first JSON object to the last.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"model config is not JSON-like: {config_path}")
    payload = json.loads(text[start : end + 1])
    return SimpleNamespace(**payload)


def _load_model_config(model_path: str) -> Any:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except (OSError, ValueError) as exc:
        return _load_config_json(model_path)


@lru_cache(maxsize=4)
def _load_discrete_trajectory_tokenizer(model_path: str) -> tuple[Any, Any]:
    _ensure_alpamayo_src_on_path()
    import hydra.utils as hyu

    config = _load_model_config(model_path)
    traj_tokenizer_cfg = getattr(config, "traj_tokenizer_cfg", None)
    if traj_tokenizer_cfg is None:
        raise ValueError(f"model config has no traj_tokenizer_cfg: {model_path}")

    traj_tokenizer = hyu.instantiate(traj_tokenizer_cfg)
    if traj_tokenizer.__class__.__name__ != "DiscreteTrajectoryTokenizer":
        raise TypeError(
            "reward_fn3 expects alpamayo_r1.action_space.discrete_action_space."
            f"DiscreteTrajectoryTokenizer, got {traj_tokenizer.__class__.__qualname__}"
        )
    return config, traj_tokenizer


def _expected_future_token_count(config: Any, traj_tokenizer: Any) -> int:
    if getattr(config, "tokens_per_future_traj", None) is not None:
        return int(config.tokens_per_future_traj)
    token_count = 1
    for dim in traj_tokenizer.action_space.get_action_space_dims():
        token_count *= int(dim)
    return token_count


def _decode_prediction_from_solution(
    solution_str: str,
    extra_info: dict[str, Any],
    model_path: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int, str, bool]:
    config, traj_tokenizer = _load_discrete_trajectory_tokenizer(model_path)
    token_ids, missing_future_end, decode_source = _extract_future_token_ids(solution_str, extra_info)
    if not token_ids:
        raise ValueError("response contains no discrete future trajectory tokens")

    expected_len = int(extra_info.get("future_token_count") or _expected_future_token_count(config, traj_tokenizer))
    raw_token_count = len(token_ids)
    token_ids = _pad_or_truncate_tokens(token_ids, expected_len)
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    history_xyz = _as_history_xyz(extra_info["ego_history_xyz"], "ego_history_xyz")
    history_rot = _as_history_rot(extra_info["ego_history_rot"], "ego_history_rot")
    pred_xyz, pred_rot, _ = traj_tokenizer.decode(
        hist_xyz=history_xyz,
        hist_rot=history_rot,
        tokens=tokens.to(history_xyz.device),
    )
    return pred_xyz[0], pred_rot[0], raw_token_count, expected_len, decode_source, missing_future_end


def _resolve_model_path(extra_info: dict[str, Any], model_path: str | None) -> str:
    resolved_model_path = model_path or extra_info.get("model_path") or os.environ.get("ALPAMAYO_MODEL_DIR") or os.environ.get("MODEL_DIR")
    if not resolved_model_path:
        raise ValueError(
            "model_path is required to instantiate DiscreteTrajectoryTokenizer. "
            "Pass reward.custom_reward_function.reward_kwargs.model_path=... "
            "or set ALPAMAYO_MODEL_DIR."
        )
    return str(resolved_model_path)


def _trajectory_reward(pred_xyz: torch.Tensor, pred_rot: torch.Tensor, gt_xyz: torch.Tensor) -> tuple[float, float, float]:
    ade = calculate_ade(pred_xyz, gt_xyz)
    comfort_dict = compute_comfort(pred_xyz[None, None, None], pred_rot[None, None, None])
    comfort = float((sum(value.mean() for value in comfort_dict.values()) / len(comfort_dict)).item()) - 1.0
    reward = -1.0 if ade >= 3.0 else -0.4 * (ade / 3.0) + 0.1 * comfort
    return reward, ade, comfort


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    model_path: str | None = None,
    reward_router_address: str | None = None,
    reason_judge_endpoint: str = REASON_JUDGE_ENDPOINT,
    reason_judge_max_tokens: int = 256,
    enable_reasoning_reward: bool = True,
    traj_weight: float = 1.0,
    reason_weight: float = 1.0,
    reason_judge_fail_reward: float = 0.0,
    missing_trajectory_reward: float = -1.0,
    cot_marker_weight: float = 0.0,
    format_weight: float = 0.0,
    **_: Any,
) -> dict[str, Any]:
    del data_source, missing_trajectory_reward
    #print(solution_str)
    extra_info = extra_info or {}
    cot_marker_score = float(has_visible_cot_markers(solution_str))
    format_score = float(has_required_response_format(solution_str))
    pred_reasoning = extract_reasoning_trace(solution_str)
    gt_reasoning = _resolve_gt_reasoning(ground_truth, extra_info)

    resolved_model_path = _resolve_model_path(extra_info, model_path)
    gt_xyz = _as_trajectory_xyz(extra_info["ego_future_xyz"], "ego_future_xyz")
    pred_xyz, pred_rot, raw_token_count, expected_token_count, decode_source, missing_future_end = _decode_prediction_from_solution(
        solution_str,
        extra_info,
        resolved_model_path,
    )
    reward, ade, comfort = _trajectory_reward(pred_xyz, pred_rot, gt_xyz)

    if enable_reasoning_reward:
        try:
            reason_info = await compute_reasoning_quality_reward(
                pred_reasoning,
                gt_reasoning,
                extra_info,
                reward_router_address=reward_router_address,
                reason_judge_endpoint=reason_judge_endpoint,
                reason_judge_max_tokens=reason_judge_max_tokens,
            )
        except Exception as exc:
            reason_info = {
                "reasoning_quality_score": 0.0,
                "reasoning_quality_reward": float(reason_judge_fail_reward),
                "reasoning_judge_reason": "",
                "reasoning_judge_error": str(exc),
            }
    else:
        reason_info = {
            "reasoning_quality_score": 0.0,
            "reasoning_quality_reward": 0.0,
            "reasoning_judge_reason": "disabled",
            "reasoning_judge_error": "",
        }

    reason_reward = float(reason_info["reasoning_quality_reward"])
    score = (
        float(traj_weight) * reward
        + float(reason_weight) * reason_reward
        + cot_marker_weight * cot_marker_score
        + format_weight * format_score
    )
    # print(f"solution_str: {solution_str}")
    # print(f"reasoning reward struct: {reason_info}")
    return {
        "score": float(score),
        "traj_L2": ade,
        "comfort_reward": comfort,
        "cot_marker_score": cot_marker_score,
        "format_score": format_score,
        "trajectory_reward": reward,
        "discrete_action_token_count": raw_token_count,
        "expected_discrete_action_token_count": expected_token_count,
        "reward_decode_source": decode_source,
        "missing_traj_future_end": float(missing_future_end),
        "reward_decode_error": "",
        "pred_reasoning_length": len(pred_reasoning),
        "gt_reasoning_length": len(gt_reasoning),
        "gt_reasoning_missing": float(not bool(gt_reasoning)),
        "traj_weight": float(traj_weight),
        "reason_weight": float(reason_weight),
        **reason_info,
    }
