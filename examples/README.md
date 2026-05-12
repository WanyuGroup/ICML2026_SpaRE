# Examples

This directory contains tiny public-format examples for checking parsers and
command wiring. They are not benchmark data and should not be used to reproduce
paper metrics.

`toy_molecules.jsonl` demonstrates the expected ChEBI-style text-to-molecule
format:

```json
{"text": "A small polar alcohol used as a solvent.", "group_selfies": "[C][O]"}
```

For real experiments, prepare public or licensed molecule datasets under
`data/public/` and keep generated artifacts under ignored directories such as
`artifacts/` and `outputs/`.
