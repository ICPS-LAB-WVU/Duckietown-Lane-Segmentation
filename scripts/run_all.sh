#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VARIANTS=(
  vanilla
  depthwise
  se_decoder
  aspp
  depthwise_se
  depthwise_aspp
  se_aspp
  full
)

DEFAULT_TRAIN_CONFIGS=(
  "configs/train/default.yaml"
  "configs/train/focal_dice.yaml"
  "configs/train/dice_ce.yaml"
  "configs/train/dsc.yaml"
)

if [[ "$#" -eq 0 ]] || [[ "${1:-}" == "all" ]]; then
  TRAIN_CONFIGS=("${DEFAULT_TRAIN_CONFIGS[@]}")
else
  TRAIN_CONFIGS=("$@")
fi

for train_config in "${TRAIN_CONFIGS[@]}"; do
  train_name="$(basename "${train_config}" .yaml)"
  for variant in "${VARIANTS[@]}"; do
    echo "Starting variant=${variant} loss_config=${train_name}"
    OUTPUT_DIR="${ROOT_DIR}/outputs/${variant}/${train_name}"
    mkdir -p "${OUTPUT_DIR}"
    python3 "${ROOT_DIR}/scripts/train.py" \
      --model-config "${ROOT_DIR}/configs/model/${variant}.yaml" \
      --train-config "${ROOT_DIR}/${train_config}" \
      --output-dir "${OUTPUT_DIR}" \
      2>&1 | tee "${OUTPUT_DIR}/run.log"
  done
done
