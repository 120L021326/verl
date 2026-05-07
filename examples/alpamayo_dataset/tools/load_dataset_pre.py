# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Download the Physical AI AV dataset from Hugging Face.

example: for downloading 4camera + egomotion for AR1 finetuning
python scripts/download_pai.py --chunk-ids 0-2 \
         --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
         --calibration camera_intrinsics sensor_extrinsics --labels egomotion
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MANDATORY_PATTERNS = [
    "features.csv",
    "clip_index.parquet",
    "metadata/**",
]
OPTIONAL_COMPONENTS = ("camera", "calibration", "labels", "lidar", "radar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Physical AI AV dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nvidia/PhysicalAI-Autonomous-Vehicles"),
        help="Local directory to store downloaded files.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not call Hugging Face snapshot_download; only use already-local data and build train/val parquet files.",
    )
    parser.add_argument(
        "--metadata-output-dir",
        type=Path,
        default=Path("clip_metadata_parquet"),
        help="Directory where train.parquet and val.parquet index files are written.",
    )
    parser.add_argument(
        "--clip-ids-file",
        type=Path,
        default=None,
        help="Optional JSON list of clip ids or objects with clip_id/t0_us. If omitted, rows are built from local clip_index.parquet.",
    )
    parser.add_argument(
        "--default-t0-us",
        type=int,
        default=5_100_000,
        help="Timestamp used when a clip id entry does not provide t0_us.",
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
        help="Optional final number of metadata rows to keep after chunk or clip-id filtering.",
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
        help="Chunk IDs to download. Supports: single '0', multi '0 1', or range '0-3' (exclusive end, downloads 0,1,2). Downloads all if not specified.",
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
    args = parser.parse_args()
    return args


def parse_component_subparts(args: argparse.Namespace) -> list[tuple[str, str]]:
    component_pairs: list[tuple[str, str]] = []
    for component in OPTIONAL_COMPONENTS:
        subparts = getattr(args, component) or []
        for subpart in subparts:
            cleaned = subpart.strip().strip("/")
            if not cleaned:
                continue
            component_pairs.append((component, cleaned))
    return component_pairs


def build_allow_patterns(
    component_pairs: list[tuple[str, str]],
    chunk_ids: list[str] | None,
) -> list[str]:
    patterns: list[str] = list(MANDATORY_PATTERNS)

    normalized_chunks = [f"chunk_{int(chunk):04d}" for chunk in (chunk_ids or [])]

    for component, subpart in component_pairs:
        if normalized_chunks:
            for chunk in normalized_chunks:
                patterns.append(f"{component}/{subpart}/{subpart}.{chunk}.*")
        else:
            patterns.append(f"{component}/{subpart}/{subpart}.*")

    # De-duplicate while preserving order.
    return list(dict.fromkeys(patterns))


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.chunk_ids is None:
        args.chunk_ids = []
    elif " " in args.chunk_ids:
        args.chunk_ids = args.chunk_ids.split(" ")
    elif "-" in args.chunk_ids:
        start = int(args.chunk_ids.split("-")[0])
        end = int(args.chunk_ids.split("-")[1])
        args.chunk_ids = list(range(start, end))
    else:
        args.chunk_ids = [int(args.chunk_ids)]

    print("downloading chunks: ", args.chunk_ids if args.chunk_ids else "all")

    try:
        component_pairs = parse_component_subparts(args)
        allow_patterns = build_allow_patterns(component_pairs, args.chunk_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
        print("Included mandatory patterns: " + ", ".join(MANDATORY_PATTERNS))
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


def _load_clip_rows_from_local_index(output_dir: Path, default_t0_us: int, chunk_ids: list[int] | None) -> list[dict[str, object]]:
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

    rows: list[dict[str, object]] = []
    for clip_id, row in clip_index.iterrows():
        t0_us = default_t0_us
        event_t0s = row.get("event_t0s", None)
        if event_t0s is not None and len(event_t0s) > 0:
            t0_us = int(event_t0s[0])
        rows.append({"clip_id": str(clip_id), "t0_us": t0_us})
    if not rows:
        raise SystemExit(f"no clips found in local index: {clip_index_path}")
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

    if args.clip_ids_file is not None:
        if not args.clip_ids_file.exists():
            raise SystemExit(f"clip id file does not exist: {args.clip_ids_file}")
        rows = _load_clip_rows_from_json(args.clip_ids_file, args.default_t0_us)
    else:
        rows = _load_clip_rows_from_local_index(args.output_dir, args.default_t0_us, args.chunk_ids)
    rows = _sample_rows(rows, args.num_samples, args.random_seed)
    for i, row in enumerate(rows):
        row["idx"] = i
        row["data_source"] = "alpamayo_physical_ai_av"

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
