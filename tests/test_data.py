from __future__ import annotations

import json

from spare_molgen.data import (
    collect_group_selfies_tokens,
    format_prompt,
    iter_text_to_molecule_examples,
    iter_texts,
    write_jsonl,
)


def test_iter_texts_jsonl(tmp_path):
    path = tmp_path / "molecules.jsonl"
    rows = [{"text": "[C]"}, {"text": " [O] "}]
    write_jsonl(rows, path)

    assert list(iter_texts(path)) == ["[C]", "[O]"]


def test_iter_texts_json(tmp_path):
    path = tmp_path / "molecules.json"
    path.write_text(json.dumps([{"text": "[N]"}]), encoding="utf-8")

    assert list(iter_texts(path)) == ["[N]"]


def test_iter_text_to_molecule_examples(tmp_path):
    path = tmp_path / "chebi_style.jsonl"
    rows = [{"text": "A small polar alcohol.", "group_selfies": "[C][O]"}]
    write_jsonl(rows, path)

    assert list(iter_text_to_molecule_examples(path)) == [
        "Text: A small polar alcohol.\nMolecule: [C][O]"
    ]


def test_format_prompt_uses_text_prompt():
    assert format_prompt("A small polar alcohol.") == "Text: A small polar alcohol.\nMolecule:"


def test_collect_group_selfies_tokens_accepts_any_bracket_token():
    texts = ["[C][=O][Branch1][*]", "prefix [Ring2] suffix"]

    assert collect_group_selfies_tokens(texts) == ["[*]", "[=O]", "[Branch1]", "[C]", "[Ring2]"]

