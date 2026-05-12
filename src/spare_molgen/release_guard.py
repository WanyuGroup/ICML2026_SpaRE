from __future__ import annotations

import argparse
import re
from pathlib import Path


ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
ARTIFACT_DIR_NAMES = {
    "artifacts",
    "activation_shards",
    "checkpoints",
    "concepts",
    "outputs",
    "runs",
    "wandb",
}
IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}
DISALLOWED_PATH_PARTS = (("data", "private"),)
MAX_SOURCE_FILE_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SENSITIVE_TEXT_PATTERNS = {
    "local user path": re.compile(r"/" r"Users/[^\s\"'`]+"),
    "private key block": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "inline secret assignment": re.compile(
        r"(?i)\b(api[_-]?key|password|secret|access[_-]?token|auth[_-]?token|"
        r"bearer[_-]?token|github[_-]?token|hf[_-]?token)\b\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_./+=-]{8,}"
    ),
}


def _is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES or part.endswith(".egg-info") for part in path.parts)


def _contains_part_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if len(sequence) > len(parts):
        return False
    return any(parts[index : index + len(sequence)] == sequence for index in range(len(parts)))


def release_check(root: str | Path) -> list[str]:
    root = Path(root)
    findings: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if _is_ignored_path(rel):
            continue
        if path.is_dir():
            continue
        if any(part in ARTIFACT_DIR_NAMES for part in rel.parts):
            findings.append(f"artifact path should not be published: {rel}")
            continue
        if any(_contains_part_sequence(rel.parts, parts) for parts in DISALLOWED_PATH_PARTS):
            findings.append(f"private data path should not be published: {rel}")
            continue
        if path.suffix.lower() in ARTIFACT_SUFFIXES:
            findings.append(f"checkpoint/tensor-like file should not be published: {rel}")
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_SOURCE_FILE_BYTES:
            findings.append(f"large source-tree file needs review ({size} bytes): {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES and size <= MAX_SOURCE_FILE_BYTES:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for label, pattern in SENSITIVE_TEXT_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{label} found in source text: {rel}")
                    break
    return findings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m spare_molgen.release_guard")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)
    findings = release_check(args.root)
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        raise SystemExit(1)
    print("OK: no obvious checkpoint, tensor, artifact, or large-file release hazards found")


if __name__ == "__main__":
    main()
