#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-artifacts/finetuned-lm}"
CONCEPT="${CONCEPT:-artifacts/concepts/global_property.pt}"
PROMPT_TEXT="${PROMPT_TEXT:-A small polar alcohol used as a solvent.}"

spare generate \
  --model "$MODEL" \
  --prompt "$PROMPT_TEXT" \
  --concept "$CONCEPT" \
  --strength "${STRENGTH:-0.8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top-p "${TOP_P:-0.95}" \
  --top-k "${TOP_K:-0}" \
  --seed "${SEED:-0}"
