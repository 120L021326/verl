#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python"
DATA_DIR="/share/datasets/Alpamayo_pai_av_big"
COC_PARQUET=""
METADATA_OUTPUT_DIR="${EXAMPLE_DIR}/data"
CHUNK_IDS=""
CLIP_IDS_FILE=""
DEFAULT_T0_US="5100000"
EVENTS_COLUMN="events"
EVENT_FRAME_RATE="10.0"
T0_SOURCE="event_timestamp"
EVENT_TIME_OFFSET_US="0"
MIN_T0_US="1600001"
VAL_RATIO="0.1"
NUM_SAMPLES=""
RANDOM_SEED="11"
FORCE_REBUILD_METADATA="false"

print_usage() {
  cat <<'EOF'
Usage: scripts/build_metadata.sh [options]

Build Alpamayo Demo3 metadata parquet files from local Physical AI AV data.

Options:
  --python PATH              Python executable to use. Default: python
  --data-dir PATH            Local Physical AI AV dataset root.
  --coc-parquet PATH         OOD reasoning parquet with clip_id and events columns.
  --metadata-output-dir PATH Output directory for train.parquet and val.parquet.
  --chunk-ids IDS            Chunk filter, e.g. 3116 or 3116-3120. Empty means all chunks.
  --clip-ids-file PATH       Optional JSON file with clip ids or clip_id/t0_us objects.
  --default-t0-us VALUE      t0_us used when no event timestamp/frame is available.
  --events-column NAME       Column in --coc-parquet containing event lists. Default: events
  --event-frame-rate VALUE   FPS used when deriving t0_us from event_start_frame. Default: 10.0
  --t0-source VALUE          event_timestamp or event_frame. Default: event_timestamp
  --event-time-offset-us N   Offset added to event-derived t0_us. Negative samples before event start.
  --min-t0-us N              Drop rows below this t0_us. Default: 1600001
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
    --coc-parquet) COC_PARQUET="$2"; shift 2 ;;
    --metadata-output-dir) METADATA_OUTPUT_DIR="$2"; shift 2 ;;
    --chunk-ids) CHUNK_IDS="$2"; shift 2 ;;
    --clip-ids-file) CLIP_IDS_FILE="$2"; shift 2 ;;
    --default-t0-us) DEFAULT_T0_US="$2"; shift 2 ;;
    --events-column) EVENTS_COLUMN="$2"; shift 2 ;;
    --event-frame-rate) EVENT_FRAME_RATE="$2"; shift 2 ;;
    --t0-source) T0_SOURCE="$2"; shift 2 ;;
    --event-time-offset-us) EVENT_TIME_OFFSET_US="$2"; shift 2 ;;
    --min-t0-us) MIN_T0_US="$2"; shift 2 ;;
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
    echo "Use --force-rebuild to regenerate metadata with new chunk or COC filters."
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

if [[ -n "${COC_PARQUET}" ]]; then
  DATASET_ARGS+=(
    --coc-parquet "${COC_PARQUET}"
    --events-column "${EVENTS_COLUMN}"
    --event-frame-rate "${EVENT_FRAME_RATE}"
    --t0-source "${T0_SOURCE}"
    --event-time-offset-us "${EVENT_TIME_OFFSET_US}"
    --min-t0-us "${MIN_T0_US}"
  )
fi

if [[ -n "${NUM_SAMPLES}" ]]; then
  DATASET_ARGS+=(--num-samples "${NUM_SAMPLES}" --random-seed "${RANDOM_SEED}")
fi

if [[ -n "${CLIP_IDS_FILE}" ]]; then
  DATASET_ARGS+=(--clip-ids-file "${CLIP_IDS_FILE}")
fi

"${PYTHON_BIN}" "${DATASET_ARGS[@]}"
