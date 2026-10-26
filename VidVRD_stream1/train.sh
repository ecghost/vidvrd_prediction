#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./local_defaults.sh
if [[ -n "${TEXT_VECTORS:-}" ]]; then
  text_args=(--text_vectors "$TEXT_VECTORS")
else
  text_args=(--text_model_weights "$TEXT_MODEL_WEIGHTS")
fi
python train_stage1.py --dataset_root "$DATASET_ROOT" "${text_args[@]}" \
  --output_dir output/train --window_size 4 --clip_len 30 --batch_size 32 \
  --learning_rate 1e-4 --num_epochs 40 --obj_loss_weight 0.2 --int_loss_weight 0.1 \
  --src_split base --tgt_split all --seed 3407 --device "$DEVICE" "$@"
