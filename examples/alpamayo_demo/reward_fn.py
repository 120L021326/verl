from __future__ import annotations

from typing import Any

COT_START_TOKEN = "<|cot_start|>"
COT_END_TOKEN = "<|cot_end|>"


def has_visible_cot_markers(solution_str: str) -> bool:
    cot_start_index = solution_str.find(COT_START_TOKEN)
    cot_end_index = solution_str.find(COT_END_TOKEN)
    return cot_start_index >= 0 and cot_end_index > cot_start_index


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    del data_source, ground_truth, extra_info
    return float(has_visible_cot_markers(solution_str))
