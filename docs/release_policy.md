# Release Policy

This project is designed so base models are supplied locally or by model id,
while training checkpoints and private artifacts are generated locally by users
and kept outside the source tree.

## Allowed Files

- Source code under `src/`
- Public documentation under `docs/`
- Configuration templates under `configs/`
- Small public examples that contain no private molecules, identifiers, or
  benchmark splits with restricted redistribution terms
- Tests that use synthetic tensors or toy strings

## Disallowed Files

- Foundation-model checkpoints
- Fine-tuned checkpoints
- SAE checkpoints
- Concept vectors
- Activation shards
- Tokenizer exports derived from private training data
- Private datasets
- Generated molecule libraries from proprietary projects
- Logs containing prompts, molecule IDs, usernames, paths, API keys, or server
  names

## Practical Rule

Anything produced by a training or inference command should stay under
`artifacts/`, `outputs/`, `runs/`, `checkpoints/`, `activation_shards/`, or
`concepts/`. Those paths are ignored.

Run:

```bash
python -m spare_molgen.release_guard .
```

before packaging or pushing the repository.
