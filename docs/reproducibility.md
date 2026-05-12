# Reproducibility Guide

This repository opens the software path for SpaRE. Training checkpoints,
SAE checkpoints, activation shards, and concept vectors are generated locally
with the commands below.

## Environment

Use Python `3.10` through `3.13`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you need the upstream Group SELFIES package for dataset preparation, install
the optional extra:

```bash
python -m pip install -e ".[groupselfies]"
```

## Data Layout

Recommended layout:

```text
data/
  public/
    train.jsonl
    local_carbon.jsonl
    soluble_positive.jsonl
    soluble_negative.jsonl
artifacts/
  finetuned-lm/
  activation_shards/
  saes/
  concepts/
outputs/
```

Training records should contain a natural-language prompt and a converted
molecule target:

```json
{"text": "A small polar alcohol used as a solvent.", "group_selfies": "[C][O]"}
```

If the input is SMILES, convert it to Group SELFIES before training. This
six-script release expects already prepared `text` and `group_selfies` fields.

Fine-tuning automatically adds all bracket tokens found in `group_selfies` to
the tokenizer before resizing the model embeddings. Later commands validate
that requested Group SELFIES tokens are already in the tokenizer and fail if
they are not.

## Phase 1: Fine-Tune The Molecular LM

```bash
spare finetune \
  --model path-or-hf-id \
  --data data/public/train.jsonl \
  --prompt-column text \
  --target-column group_selfies \
  --output artifacts/finetuned-lm \
  --epochs 2 \
  --batch-size 64 \
  --lr 5e-5 \
  --max-length 1024
```

The output directory is intentionally ignored by git.

## Phase 2: Collect Activations

Collect late-layer activations for local control and early/mid-layer
activations for global control.

```bash
spare collect-activations \
  --model artifacts/finetuned-lm \
  --data data/public/train.jsonl \
  --prompt-column text \
  --target-column group_selfies \
  --layers 10 22 \
  --output artifacts/activation_shards \
  --selector all \
  --batch-size 4 \
  --max-length 1024
```

For target-token local concept extraction, collect activations from exemplar
records that contain the target atom or functional group at the intended token
position. Keep those shards in a separate ignored artifact directory.

## Phase 3: Train Layer-Specific SAEs

```bash
spare train-sae \
  --activations artifacts/activation_shards/layer_10 \
  --output artifacts/saes/layer_10.pt \
  --expansion-factor 40 \
  --epochs 8 \
  --batch-size 1024 \
  --lr 1e-4 \
  --l1 1e-5
```

Repeat for each layer used for control. The paper reports layer `22` for local
control and layer `10` for global control.

## Phase 4: Extract Concepts

Local concept:

```bash
spare extract-local \
  --sae artifacts/saes/layer_22.pt \
  --tokenizer artifacts/finetuned-lm \
  --token "[C]" \
  --activations artifacts/local_carbon/layer_22 \
  --name carbon \
  --layer 22 \
  --threshold 0.5 \
  --min-fraction 1.0 \
  --output artifacts/concepts/carbon.pt
```

Global concept:

```bash
spare extract-global \
  --sae artifacts/saes/layer_10.pt \
  --positive-activations artifacts/soluble_positive/layer_10 \
  --negative-activations artifacts/soluble_negative/layer_10 \
  --name solubility \
  --layer 10 \
  --threshold 0.5 \
  --output artifacts/concepts/solubility.pt
```

Concept files are derived artifacts and are ignored by git.

## Phase 5: Edited Generation

Global property control:

```bash
spare generate \
  --model artifacts/finetuned-lm \
  --prompt "A small polar alcohol used as a solvent." \
  --concept artifacts/concepts/solubility.pt \
  --strength 0.8 \
  --max-new-tokens 128 \
  --seed 0
```

Local substructure control:

```bash
spare generate \
  --model artifacts/finetuned-lm \
  --prompt "A small polar alcohol used as a solvent." \
  --concept artifacts/concepts/carbon.pt \
  --strength 1.0 \
  --local-step 4 \
  --max-new-tokens 128 \
  --seed 0
```

## Release Check

Before pushing or packaging:

```bash
python -m spare_molgen.release_guard .
```

This gate intentionally fails if checkpoint-like files, activation shards,
concept vectors, private data directories, or large artifact-like files are in
the source tree.
