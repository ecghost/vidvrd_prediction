#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./local_defaults.sh
python infer_stream.py --dataset_root "$DATASET_ROOT" --init_checkpoint "$CHECKPOINT" \
  --output_dir output/infer --window_size 4 --clip_len 30 --tgt_split all \
  --device "$DEVICE" "$@"
