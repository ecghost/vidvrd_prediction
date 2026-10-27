#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./local_defaults.sh
python infer_stream.py --dataset_root "$DATASET_ROOT" --init_checkpoint "$CHECKPOINT" \
  --output_dir output --window_size 4 --clip_len 30 --tgt_split all --clip_top_n 20 \
  --inference_batch_size 32 --max_active_videos 1 \
  --device "$DEVICE" "$@"
