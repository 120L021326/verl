#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python"
DATA_DIR="/share/datasets/Alpamayo_pai_av_big"
METADATA_OUTPUT_DIR="${EXAMPLE_DIR}/clip_metadata_parquet"
CHUNK_IDS=""
CLIP_IDS_FILE=""
DEFAULT_T0_US="5100000"
VAL_RATIO="0.1"
NUM_SAMPLES=""
RANDOM_SEED="11"
FORCE_REBUILD_METADATA="false"

print_usage() {
  cat <<'EOF'
Usage: scripts/build_metadata.sh [options]

Build Alpamayo dataset metadata parquet files from a local Physical AI AV dataset.

Options:
  --python PATH              Python executable to use. Default: python
  --data-dir PATH            Local Physical AI AV dataset root.
  --metadata-output-dir PATH Output directory for train.parquet and val.parquet.
  --chunk-ids IDS            Chunk filter, e.g. 3116 or 0-8. Empty means all chunks.
  --clip-ids-file PATH       Optional JSON file with clip ids or clip_id/t0_us objects.
  --default-t0-us VALUE      t0_us used when a clip id does not provide one.
  --val-ratio VALUE          Fraction of rows reserved for validation.
  --num-samples N            Final number of metadata rows to keep after filtering.
  --random-seed SEED         Seed used with --num-samples. Default: 11
  --force-rebuild            Regenerate even if metadata files already exist.
  -h, --help                 Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --metadata-output-dir) METADATA_OUTPUT_DIR="$2"; shift 2 ;;
    --chunk-ids) CHUNK_IDS="$2"; shift 2 ;;
    --clip-ids-file) CLIP_IDS_FILE="$2"; shift 2 ;;
    --default-t0-us) DEFAULT_T0_US="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --force-rebuild) FORCE_REBUILD_METADATA="true"; shift 1 ;;
    -h|--help) print_usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; print_usage >&2; exit 2 ;;
  esac
done

TRAIN_FILE="${METADATA_OUTPUT_DIR}/train.parquet"
VAL_FILE="${METADATA_OUTPUT_DIR}/val.parquet"

if [[ -f "${TRAIN_FILE}" && -f "${VAL_FILE}" ]]; then
  if [[ "${FORCE_REBUILD_METADATA}" != "true" ]]; then
    echo "Dataset metadata already prepared:"
    echo "  train: ${TRAIN_FILE}"
    echo "  val:   ${VAL_FILE}"
    echo "Use --force-rebuild to regenerate metadata with new chunk or clip filters."
    exit 0
  fi
  echo "Force rebuilding dataset metadata:"
  echo "  removing ${TRAIN_FILE}"
  echo "  removing ${VAL_FILE}"
  rm -f "${TRAIN_FILE}" "${VAL_FILE}"
fi

DATASET_ARGS=(
  "${EXAMPLE_DIR}/tools/load_dataset_pre.py"
  --skip-download
  --output-dir "${DATA_DIR}"
  --metadata-output-dir "${METADATA_OUTPUT_DIR}"
  --default-t0-us "${DEFAULT_T0_US}"
  --val-ratio "${VAL_RATIO}"
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov
  --labels egomotion
)

if [[ -n "${CHUNK_IDS}" ]]; then
  DATASET_ARGS+=(--chunk-ids "${CHUNK_IDS}")
fi

if [[ -n "${NUM_SAMPLES}" ]]; then
  DATASET_ARGS+=(--num-samples "${NUM_SAMPLES}" --random-seed "${RANDOM_SEED}")
fi

if [[ -n "${CLIP_IDS_FILE}" ]]; then
  DATASET_ARGS+=(--clip-ids-file "${CLIP_IDS_FILE}")
fi

"${PYTHON_BIN}" "${DATASET_ARGS[@]}"
