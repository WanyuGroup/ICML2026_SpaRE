#!/usr/bin/env bash
set -euo pipefail

SAE="${SAE:-artifacts/saes/layer_10.pt}"
POSITIVE_ACTIVATIONS="${POSITIVE_ACTIVATIONS:-artifacts/global_positive/layer_10}"
NEGATIVE_ACTIVATIONS="${NEGATIVE_ACTIVATIONS:-artifacts/global_negative/layer_10}"
OUTPUT="${OUTPUT:-artifacts/concepts/global_property.pt}"

spare extract-global \
  --sae "$SAE" \
  --positive-activations "$POSITIVE_ACTIVATIONS" \
  --negative-activations "$NEGATIVE_ACTIVATIONS" \
  --name "${NAME:-global_property}" \
  --layer "${LAYER:-10}" \
  --threshold "${THRESHOLD:-0.5}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --output "$OUTPUT"
