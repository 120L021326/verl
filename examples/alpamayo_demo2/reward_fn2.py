from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch

COT_START_TOKEN = "<|cot_start|>"
COT_END_TOKEN = "<|cot_end|>"
TRAJ_FUTURE_START_TOKEN = "<|traj_future_start|>"
TRAJ_FUTURE_END_TOKEN = "<|traj_future_end|>"
TRAJ_TOKEN_PATTERN = re.compile(r"<i(\d+)>")
TRAJ_VOCAB_SIZE = 3000
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


def _extract_future_segment(solution_str: str) -> str:
    start_index = solution_str.find(TRAJ_FUTURE_START_TOKEN)
    if start_index < 0:
        raise ValueError(f"response is missing {TRAJ_FUTURE_START_TOKEN}")
    start_index += len(TRAJ_FUTURE_START_TOKEN)
    end_index = solution_str.find(TRAJ_FUTURE_END_TOKEN, start_index)
    if end_index < 0:
        raise ValueError(f"response is missing {TRAJ_FUTURE_END_TOKEN}")
    return solution_str[start_index:end_index]


def _extract_future_token_ids(solution_str: str) -> list[int]:
    future_segment = _extract_future_segment(solution_str)
    token_ids = []
    for raw_token_id in TRAJ_TOKEN_PATTERN.findall(future_segment):
        token_id = int(raw_token_id)
        if 0 <= token_id < TRAJ_VOCAB_SIZE:
            token_ids.append(token_id)
    return token_ids


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
            "reward_fn2 expects alpamayo_r1.action_space.discrete_action_space."
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
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    config, traj_tokenizer = _load_discrete_trajectory_tokenizer(model_path)
    token_ids = _extract_future_token_ids(solution_str)
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
    return pred_xyz[0], pred_rot[0], raw_token_count, expected_len


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
    reward = -1.0 if ade >= 3.0 else -(ade / 3.0) + 0.1 * comfort
    return reward, ade, comfort


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    model_path: str | None = None,
    missing_trajectory_reward: float = -1.0,
    cot_marker_weight: float = 0.0,
    format_weight: float = 0.0,
    **_: Any,
) -> dict[str, Any]:
    del data_source, ground_truth
    #print(solution_str)
    extra_info = extra_info or {}
    cot_marker_score = float(has_visible_cot_markers(solution_str))
    format_score = float(has_required_response_format(solution_str))

    resolved_model_path = _resolve_model_path(extra_info, model_path)
    gt_xyz = _as_trajectory_xyz(extra_info["ego_future_xyz"], "ego_future_xyz")
    pred_xyz, pred_rot, raw_token_count, expected_token_count = _decode_prediction_from_solution(
        solution_str,
        extra_info,
        resolved_model_path,
    )
    reward, ade, comfort = _trajectory_reward(pred_xyz, pred_rot, gt_xyz)
    score = reward + cot_marker_weight * cot_marker_score + format_weight * format_score
    return {
        "score": float(score),
        "traj_L2": ade,
        "comfort_reward": comfort,
        "cot_marker_score": cot_marker_score,
        "format_score": format_score,
        "trajectory_reward": reward,
        "discrete_action_token_count": raw_token_count,
        "expected_discrete_action_token_count": expected_token_count,
        "reward_decode_source": "decoded_traj_future_tokens",
        "reward_decode_error": "",
    }
