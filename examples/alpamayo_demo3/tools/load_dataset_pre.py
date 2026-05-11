# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build Alpamayo Demo3 metadata parquet files.

This is a standalone metadata builder for demo3. In addition to the normal
clip-index path, it can read an OOD reasoning parquet with an ``events`` column,
explode every event into one training row, keep the event ``coc`` text, and set
``t0_us`` from the event start time.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MANDATORY_PATTERNS = [
    "features.csv",
    "clip_index.parquet",
    "metadata/**",
]
OPTIONAL_COMPONENTS = ("camera", "calibration", "labels", "lidar", "radar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Alpamayo metadata parquet files from local PAI data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nvidia/PhysicalAI-Autonomous-Vehicles"),
        help="Local Physical AI AV dataset root.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call Hugging Face snapshot_download; only use already-local data.",
    )
    parser.add_argument(
        "--metadata-output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where train.parquet and val.parquet index files are written.",
    )
    parser.add_argument(
        "--clip-ids-file",
        type=Path,
        default=None,
        help="Optional JSON list of clip ids or objects with clip_id/t0_us.",
    )
    parser.add_argument(
        "--coc-parquet",
        type=Path,
        default=None,
        help="Optional OOD reasoning parquet with clip_id and events columns.",
    )
    parser.add_argument(
        "--events-column",
        type=str,
        default="events",
        help="Column in --coc-parquet containing JSON event lists.",
    )
    parser.add_argument(
        "--default-t0-us",
        type=int,
        default=5_100_000,
        help="Timestamp used when no event timestamp/frame is available.",
    )
    parser.add_argument(
        "--event-frame-rate",
        type=float,
        default=10.0,
        help="Fallback frame rate for converting event_start_frame to t0_us.",
    )
    parser.add_argument(
        "--t0-source",
        choices=("event_timestamp", "event_frame"),
        default="event_timestamp",
        help="Use event_start_timestamp or event_start_frame to derive t0_us.",
    )
    parser.add_argument(
        "--event-time-offset-us",
        type=int,
        default=0,
        help="Offset added to event-derived t0_us. Use negative values to sample before event start.",
    )
    parser.add_argument(
        "--min-t0-us",
        type=int,
        default=1_600_001,
        help="Drop rows whose final t0_us is not large enough for the default 1.6s history window.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of index rows reserved for validation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional final number of metadata rows to keep after filtering.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=11,
        help="Random seed used when --num-samples is set.",
    )
    parser.add_argument(
        "--chunk-ids",
        type=str,
        default=None,
        help="Chunk IDs. Supports single '0', multi '0 1', or range '0-3' with exclusive end.",
    )
    parser.add_argument(
        "--camera",
        nargs="+",
        default=None,
        help="Camera subparts, e.g. camera_front_wide_120fov camera_cross_left_120fov.",
    )
    parser.add_argument(
        "--calibration",
        nargs="+",
        default=None,
        help="Calibration subparts, e.g. camera_intrinsics sensor_extrinsics.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels subparts, e.g. egomotion.",
    )
    parser.add_argument(
        "--lidar",
        nargs="+",
        default=None,
        help="Lidar subparts, e.g. lidar_top_360fov.",
    )
    parser.add_argument(
        "--radar",
        nargs="+",
        default=None,
        help="Radar subparts, e.g. radar_front_center_mrr_2.",
    )
    return parser.parse_args()


def _parse_chunk_ids(raw_chunk_ids: str | None) -> list[int]:
    if raw_chunk_ids is None or raw_chunk_ids == "":
        return []
    if " " in raw_chunk_ids:
        return [int(chunk) for chunk in raw_chunk_ids.split(" ") if chunk]
    if "-" in raw_chunk_ids:
        start, end = raw_chunk_ids.split("-", 1)
        return list(range(int(start), int(end)))
    return [int(raw_chunk_ids)]


def parse_component_subparts(args: argparse.Namespace) -> list[tuple[str, str]]:
    component_pairs: list[tuple[str, str]] = []
    for component in OPTIONAL_COMPONENTS:
        subparts = getattr(args, component) or []
        for subpart in subparts:
            cleaned = subpart.strip().strip("/")
            if cleaned:
                component_pairs.append((component, cleaned))
    return component_pairs


def build_allow_patterns(
    component_pairs: list[tuple[str, str]],
    chunk_ids: list[int] | None,
) -> list[str]:
    patterns: list[str] = list(MANDATORY_PATTERNS)
    normalized_chunks = [f"chunk_{int(chunk):04d}" for chunk in (chunk_ids or [])]

    for component, subpart in component_pairs:
        if normalized_chunks:
            for chunk in normalized_chunks:
                patterns.append(f"{component}/{subpart}/{subpart}.{chunk}.*")
        else:
            patterns.append(f"{component}/{subpart}/{subpart}.*")
    return list(dict.fromkeys(patterns))


def main() -> None:
    args = parse_args()
    args.chunk_ids = _parse_chunk_ids(args.chunk_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("using chunks: ", args.chunk_ids if args.chunk_ids else "all")

    component_pairs = parse_component_subparts(args)
    allow_patterns = build_allow_patterns(component_pairs, args.chunk_ids)
    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is not installed. Install with: pip install huggingface_hub"
            ) from exc

        print("download patterns", allow_patterns)
        downloaded_path = snapshot_download(
            repo_id=DEFAULT_REPO_ID,
            repo_type="dataset",
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        print(f"Downloaded dataset snapshot to: {downloaded_path}")
    else:
        print("skip download: using existing local dataset only")
        print(f"local dataset dir: {args.output_dir}")

    write_index_parquets(args)


def _load_clip_rows_from_json(path: Path, default_t0_us: int) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("clips", raw.get("clip_ids", []))

    rows = []
    for item in raw:
        if isinstance(item, str):
            rows.append({"clip_id": item, "t0_us": default_t0_us})
        elif isinstance(item, dict):
            clip_id = item.get("clip_id") or item.get("id")
            if not clip_id:
                raise SystemExit(f"clip entry is missing clip_id: {item}")
            rows.append({"clip_id": str(clip_id), "t0_us": int(item.get("t0_us", default_t0_us))})
        else:
            raise SystemExit(f"unsupported clip id entry: {item!r}")
    if not rows:
        raise SystemExit(f"no clip ids found in {path}")
    return rows


def _load_filtered_clip_index(output_dir: Path, chunk_ids: list[int] | None) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read local PAI clip_index.parquet.") from exc

    clip_index_path = output_dir / "clip_index.parquet"
    if not clip_index_path.exists():
        raise SystemExit(f"local clip index does not exist: {clip_index_path}")

    clip_index = pd.read_parquet(clip_index_path)
    if chunk_ids:
        clip_index = clip_index.loc[clip_index["chunk"].isin(chunk_ids)]
    if clip_index.empty:
        raise SystemExit(f"no clips found in local index after chunk filtering: {clip_index_path}")
    return clip_index


def _load_clip_rows_from_local_index(
    output_dir: Path,
    default_t0_us: int,
    chunk_ids: list[int] | None,
) -> list[dict[str, object]]:
    clip_index = _load_filtered_clip_index(output_dir, chunk_ids)

    rows: list[dict[str, object]] = []
    for clip_id, row in clip_index.iterrows():
        t0_us = default_t0_us
        event_t0s = row.get("event_t0s", None)
        if event_t0s is not None and len(event_t0s) > 0:
            t0_us = int(event_t0s[0])
        rows.append({"clip_id": str(clip_id), "t0_us": t0_us, "chunk": int(row["chunk"])})
    return rows


def _parse_events(raw_events: object) -> list[dict[str, object]]:
    if raw_events is None:
        return []
    if isinstance(raw_events, str):
        raw_events = raw_events.strip()
        if not raw_events:
            return []
        parsed = json.loads(raw_events)
    elif isinstance(raw_events, float) and raw_events != raw_events:
        return []
    else:
        parsed = raw_events

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, Iterable):
        events = []
        for event in parsed:
            if not isinstance(event, dict):
                raise SystemExit(f"event entry must be a dict, got: {event!r}")
            events.append(event)
        return events
    raise SystemExit(f"unsupported events payload: {raw_events!r}")


def _event_t0_us(
    event: dict[str, object],
    default_t0_us: int,
    event_frame_rate: float,
    event_time_offset_us: int,
    t0_source: str,
) -> int:
    timestamp = event.get("event_start_timestamp")
    frame = event.get("event_start_frame")

    if t0_source == "event_frame" and frame is not None:
        if event_frame_rate <= 0:
            raise SystemExit("--event-frame-rate must be positive")
        return int(round(float(frame) / event_frame_rate * 1_000_000)) + event_time_offset_us

    if timestamp is not None:
        return int(timestamp) + event_time_offset_us

    if frame is not None:
        if event_frame_rate <= 0:
            raise SystemExit("--event-frame-rate must be positive")
        return int(round(float(frame) / event_frame_rate * 1_000_000)) + event_time_offset_us

    return default_t0_us + event_time_offset_us


def _load_coc_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read COC parquet files.") from exc

    if args.coc_parquet is None:
        raise SystemExit("--coc-parquet is required for COC metadata generation")
    if not args.coc_parquet.exists():
        raise SystemExit(f"COC parquet does not exist: {args.coc_parquet}")

    clip_index = _load_filtered_clip_index(args.output_dir, args.chunk_ids)
    clip_to_chunk = {str(clip_id): int(row["chunk"]) for clip_id, row in clip_index.iterrows()}

    coc_df = pd.read_parquet(args.coc_parquet)
    required_columns = {args.events_column}
    missing_columns = required_columns - set(coc_df.columns)
    if missing_columns:
        raise SystemExit(
            f"COC parquet {args.coc_parquet} is missing required columns: {sorted(missing_columns)}"
        )
    has_clip_id_column = "clip_id" in coc_df.columns
    if not has_clip_id_column and coc_df.index.name != "clip_id":
        raise SystemExit(
            f"COC parquet {args.coc_parquet} must provide clip_id either as a column or as the index name."
        )

    rows: list[dict[str, object]] = []
    skipped_empty_events = 0
    skipped_missing_coc = 0
    skipped_too_early = 0
    skipped_not_in_chunk = 0
    for source_row_index, source_row in coc_df.iterrows():
        clip_id = str(source_row["clip_id"] if has_clip_id_column else source_row_index)
        if clip_id not in clip_to_chunk:
            skipped_not_in_chunk += 1
            continue

        events = _parse_events(source_row[args.events_column])
        if not events:
            skipped_empty_events += 1
            continue
        for event_index, event in enumerate(events):
            coc = event.get("coc") or event.get("cot") or event.get("reasoning")
            if coc is None:
                skipped_missing_coc += 1
                continue

            t0_us = _event_t0_us(
                event,
                args.default_t0_us,
                args.event_frame_rate,
                args.event_time_offset_us,
                args.t0_source,
            )
            if args.min_t0_us is not None and t0_us < args.min_t0_us:
                skipped_too_early += 1
                continue

            row = {
                "clip_id": clip_id,
                "t0_us": t0_us,
                "coc": str(coc),
                "chunk": clip_to_chunk[clip_id],
                "event_index": event_index,
                "source_row_index": source_row_index,
                "event_start_frame": event.get("event_start_frame"),
                "event_start_timestamp": event.get("event_start_timestamp"),
            }
            for column in ("feature", "event_cluster", "split"):
                if column in coc_df.columns:
                    row[column] = source_row[column]
            rows.append(row)

    if not rows:
        raise SystemExit("no COC rows found after chunk/event filtering")
    generated_chunks = sorted({int(row["chunk"]) for row in rows})
    top_generated_chunks = Counter(int(row["chunk"]) for row in rows).most_common(5)
    print(
        "COC rows built:",
        len(rows),
        "| generated_chunks:",
        generated_chunks,
        "| generated_chunk_count:",
        len(generated_chunks),
        "| top_generated_chunks:",
        top_generated_chunks,
        "| skipped_not_in_chunk:",
        skipped_not_in_chunk,
        "| skipped_empty_events:",
        skipped_empty_events,
        "| skipped_missing_coc:",
        skipped_missing_coc,
        "| skipped_too_early:",
        skipped_too_early,
    )
    return rows


def _sample_rows(
    rows: list[dict[str, object]],
    num_samples: int | None,
    random_seed: int,
) -> list[dict[str, object]]:
    if num_samples is None:
        return rows
    if num_samples <= 0:
        raise SystemExit(f"--num-samples must be positive, got {num_samples}")
    if num_samples > len(rows):
        raise SystemExit(
            f"--num-samples={num_samples} exceeds available rows after filtering: {len(rows)}"
        )
    if num_samples == len(rows):
        return rows
    rng = random.Random(random_seed)
    return rng.sample(rows, num_samples)


def write_index_parquets(args: argparse.Namespace) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to write Alpamayo train/val parquet files.") from exc

    if args.coc_parquet is not None:
        rows = _load_coc_rows(args)
    elif args.clip_ids_file is not None:
        if not args.clip_ids_file.exists():
            raise SystemExit(f"clip id file does not exist: {args.clip_ids_file}")
        rows = _load_clip_rows_from_json(args.clip_ids_file, args.default_t0_us)
    else:
        rows = _load_clip_rows_from_local_index(args.output_dir, args.default_t0_us, args.chunk_ids)

    rows = _sample_rows(rows, args.num_samples, args.random_seed)
    for i, row in enumerate(rows):
        row["idx"] = i
        row["data_source"] = "alpamayo_physical_ai_av_coc" if args.coc_parquet else "alpamayo_physical_ai_av"
        row.setdefault("ground_truth", row.get("coc", ""))

    val_count = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 1 else 1
    train_rows = rows[:-val_count] if len(rows) > 1 else rows
    val_rows = rows[-val_count:]

    args.metadata_output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.metadata_output_dir / "train.parquet"
    val_path = args.metadata_output_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    print(f"Wrote train index: {train_path} ({len(train_rows)} rows)")
    print(f"Wrote val index: {val_path} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
