from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

GROUP_SELFIES_TOKEN_RE = re.compile(r"\[[^\[\]]+\]")


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"JSONL rows must be objects: {path}")
                    yield record
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("data", data.get("records", []))
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list or data/records list: {path}")
        for record in data:
            if not isinstance(record, dict):
                raise ValueError(f"JSON records must be objects: {path}")
            yield record
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffix == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield {"text": text}
    else:
        raise ValueError(f"Unsupported data file suffix: {path.suffix}")


def iter_texts(
    path: str | Path,
    text_column: str = "text",
    limit: int | None = None,
) -> Iterator[str]:
    count = 0
    for record in iter_records(path):
        if text_column not in record:
            raise KeyError(f"Column {text_column!r} not found in {path}")
        text = str(record[text_column]).strip()
        if text:
            yield text
            count += 1
            if limit is not None and count >= limit:
                return


def iter_group_selfies_tokens(text: str) -> Iterator[str]:
    yield from GROUP_SELFIES_TOKEN_RE.findall(text)


def collect_group_selfies_tokens(texts: Iterable[str]) -> list[str]:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(iter_group_selfies_tokens(text))
    return sorted(tokens)


DEFAULT_PROMPT_TEMPLATE = "Text: {prompt}\nMolecule:"
DEFAULT_TRAIN_TEMPLATE = "Text: {prompt}\nMolecule: {target}"


def format_prompt(prompt: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(prompt=prompt.strip())


def format_supervised_example(
    prompt: str,
    target: str,
    template: str = DEFAULT_TRAIN_TEMPLATE,
) -> str:
    return template.format(prompt=prompt.strip(), target=target.strip())


def iter_text_to_molecule_examples(
    path: str | Path,
    prompt_column: str = "text",
    target_column: str = "group_selfies",
    template: str = DEFAULT_TRAIN_TEMPLATE,
    limit: int | None = None,
) -> Iterator[str]:
    count = 0
    for record in iter_records(path):
        if prompt_column not in record:
            raise KeyError(f"Prompt column {prompt_column!r} not found in {path}")
        if target_column not in record:
            raise KeyError(f"Target column {target_column!r} not found in {path}")
        prompt = str(record[prompt_column]).strip()
        target = str(record[target_column]).strip()
        if prompt and target:
            yield format_supervised_example(prompt, target, template=template)
            count += 1
            if limit is not None and count >= limit:
                return


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
