#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-artifacts/finetuned-lm}"
DATA="${DATA:-data/public/train.jsonl}"
OUTPUT="${OUTPUT:-artifacts/activation_shards}"

spare collect-activations \
  --model "$MODEL" \
  --data "$DATA" \
  --prompt-column text \
  --target-column group_selfies \
  --layers ${LAYERS:-10 22} \
  --output "$OUTPUT" \
  --selector "${SELECTOR:-all}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-length "${MAX_LENGTH:-1024}"

