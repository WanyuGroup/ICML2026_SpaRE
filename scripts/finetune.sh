#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-path-or-hf-id}"
DATA="${DATA:-data/public/train.jsonl}"
OUTPUT="${OUTPUT:-artifacts/finetuned-lm}"

spare finetune \
  --model "$MODEL" \
  --data "$DATA" \
  --prompt-column "${PROMPT_COLUMN:-text}" \
  --target-column "${TARGET_COLUMN:-group_selfies}" \
  --output "$OUTPUT" \
  --epochs "${EPOCHS:-2}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --lr "${LR:-5e-5}" \
  --max-length "${MAX_LENGTH:-1024}"
