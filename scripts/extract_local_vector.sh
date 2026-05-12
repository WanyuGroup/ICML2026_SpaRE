#!/usr/bin/env bash
set -euo pipefail

SAE="${SAE:-artifacts/saes/layer_22.pt}"
TOKENIZER="${TOKENIZER:-artifacts/finetuned-lm}"
TOKEN="${TOKEN:-[C]}"
ACTIVATIONS="${ACTIVATIONS:-artifacts/local_token/layer_22}"
OUTPUT="${OUTPUT:-artifacts/concepts/local_token.pt}"

spare extract-local \
  --sae "$SAE" \
  --tokenizer "$TOKENIZER" \
  --token "$TOKEN" \
  --activations "$ACTIVATIONS" \
  --name "${NAME:-local_token}" \
  --layer "${LAYER:-22}" \
  --threshold "${THRESHOLD:-0.5}" \
  --min-fraction "${MIN_FRACTION:-1.0}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --output "$OUTPUT"
