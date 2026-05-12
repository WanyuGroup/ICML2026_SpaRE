# Release Checklist

Use this checklist before publishing the code repository.

## Required

- `README.md` names the paper and includes the OpenReview URL.
- `docs/artifact_evaluation.md` gives reviewers a no-private-artifact check.
- `CITATION.bib` matches the paper BibTeX entry.
- `LICENSE` is present.
- `NOTICE`, `ACKNOWLEDGEMENTS.md`, and `docs/third_party.md` acknowledge
  Group SELFIES and its Apache-2.0 license.
- `.gitignore` and `MANIFEST.in` exclude generated artifacts.
- `python -m spare_molgen.release_guard .` passes from the release root.
- `spare --help` exposes exactly the six public workflow commands.
- `pytest -q` passes in a Python `3.10` to `3.13` environment.
- No review-only PDF is inside the release repository.
- No private paths, usernames, server names, API keys, or internal dataset names
  appear in source or docs.

## Generated Checkpoint And Artifact Policy

The released workflow should be generated with exactly these six scripts:

```bash
bash scripts/finetune.sh
bash scripts/collect_activations.sh
bash scripts/train_sae.sh
bash scripts/extract_local_vector.sh
bash scripts/extract_global_vector.sh
bash scripts/controlled_generate.sh
```

Keep these generated files under ignored local artifact directories.

- Base model weights.
- Fine-tuned model weights.
- SAE weights.
- Activation shards.
- Concept vectors.
- Private datasets.
- Proprietary molecule candidates.
- Experiment logs with prompts or molecule identifiers.
