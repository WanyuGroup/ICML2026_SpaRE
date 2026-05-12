#!/usr/bin/env bash
set -euo pipefail

LAYER="${LAYER:-10}"
ACTIVATIONS="${ACTIVATIONS:-artifacts/activation_shards/layer_${LAYER}}"
OUTPUT="${OUTPUT:-artifacts/saes/layer_${LAYER}.pt}"

spare train-sae \
  --activations "$ACTIVATIONS" \
  --output "$OUTPUT" \
  --expansion-factor "${EXPANSION_FACTOR:-40}" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --lr "${LR:-1e-4}" \
  --l1 "${L1:-1e-5}"

