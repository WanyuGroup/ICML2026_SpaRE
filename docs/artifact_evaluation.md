# Artifact Evaluation Guide

This document is intended for reviewers and readers who want to inspect the
released code without access to private checkpoints or generated artifacts.

## What This Artifact Contains

- The six public SpaRE workflow scripts.
- Group SELFIES token handling, including tokenizer extension during
  fine-tuning.
- SAE training and concept extraction code.
- Inference-time activation editing hooks.
- Tests and release checks.

## What This Artifact Does Not Contain

- Base model checkpoints.
- Fine-tuned model checkpoints.
- SAE checkpoints.
- Activation shards.
- Concept vectors.
- Private datasets.
- Generated molecule libraries.

Those files are generated locally by the commands in
`docs/reproducibility.md` and are ignored by git.

## Quick Code Check

Use Python `3.10` through `3.13`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m compileall src tests
spare --help
pytest -q
python -m spare_molgen.release_guard .
```

Expected result:

- `spare --help` lists only the six public workflow commands.
- `pytest -q` passes the unit tests.
- `python -m spare_molgen.release_guard .` passes when no generated artifacts
  are inside the repository.

## Group SELFIES Tokenizer Check

The implementation treats every bracket token of the form `[...]` in the
`group_selfies` target column as a Group SELFIES token. During fine-tuning,
missing bracket tokens are added to the tokenizer and the model embedding
matrix is resized. Later commands fail if a requested token is not already in
the tokenizer.

The behavior is covered by:

- `tests/test_data.py`
- `tests/test_tokenizer_utils.py`

## Reproducing Paper-Style Runs

A full paper-style run requires a base molecular language model, Group
SELFIES-formatted data, and GPU compute. Follow:

- `docs/reproducibility.md`
- `scripts/finetune.sh`
- `scripts/collect_activations.sh`
- `scripts/train_sae.sh`
- `scripts/extract_local_vector.sh`
- `scripts/extract_global_vector.sh`
- `scripts/controlled_generate.sh`

The script outputs go under ignored artifact directories.
