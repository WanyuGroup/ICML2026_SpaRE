from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from pathlib import Path

import torch
from tqdm import tqdm

from spare_molgen.data import DEFAULT_TRAIN_TEMPLATE, iter_text_to_molecule_examples, iter_texts
from spare_molgen.tokenizer_utils import require_group_selfies_tokens_in_tokenizer


def list_shards(path: str | Path) -> list[Path]:
    root = Path(path)
    shards = sorted(root.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No activation shards found under {root}")
    return shards


def infer_activation_dim(path: str | Path) -> int:
    shard = torch.load(list_shards(path)[0], map_location="cpu")
    activations = shard["activations"]
    return int(activations.shape[-1])


def iter_activation_batches(
    path: str | Path,
    batch_size: int,
    shuffle_shards: bool = True,
    device: str | torch.device | None = None,
) -> Iterator[torch.Tensor]:
    shards = list_shards(path)
    if shuffle_shards:
        random.shuffle(shards)
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu")
        activations = shard["activations"].float()
        if shuffle_shards:
            perm = torch.randperm(activations.shape[0])
            activations = activations[perm]
        for start in range(0, activations.shape[0], batch_size):
            batch = activations[start : start + batch_size]
            if device is not None:
                batch = batch.to(device)
            yield batch


def count_activations(path: str | Path) -> int:
    total = 0
    for shard_path in list_shards(path):
        shard = torch.load(shard_path, map_location="cpu")
        total += int(shard["activations"].shape[0])
    return total


def _select_activations(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    selector: str,
) -> torch.Tensor:
    if selector == "all":
        return hidden[attention_mask.bool()].detach().cpu()
    if selector == "last":
        lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, lengths].detach().cpu()
    raise ValueError(f"Unsupported selector: {selector}")


def collect_activation_shards(
    model,
    tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    layers: list[int],
    text_column: str | None = None,
    prompt_column: str = "text",
    target_column: str = "group_selfies",
    template: str = DEFAULT_TRAIN_TEMPLATE,
    batch_size: int = 4,
    max_length: int = 1024,
    selector: str = "all",
    shard_size: int = 8192,
    limit: int | None = None,
    device: str | torch.device | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        (output_root / f"layer_{layer}").mkdir(parents=True, exist_ok=True)

    buffers: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    counts: dict[int, int] = {layer: 0 for layer in layers}
    shard_ids: dict[int, int] = {layer: 0 for layer in layers}
    if text_column:
        texts_iter = iter_texts(data_path, text_column=text_column, limit=limit)
    else:
        texts_iter = iter_text_to_molecule_examples(
            data_path,
            prompt_column=prompt_column,
            target_column=target_column,
            template=template,
            limit=limit,
        )
    total_batches = None if limit is None else math.ceil(limit / batch_size)

    def flush(layer: int, force: bool = False) -> None:
        if not buffers[layer]:
            return
        current = torch.cat(buffers[layer], dim=0)
        if current.shape[0] < shard_size and not force:
            buffers[layer] = [current]
            return
        while current.shape[0] >= shard_size or (force and current.shape[0] > 0):
            chunk = current[:shard_size] if current.shape[0] > shard_size else current
            current = current[chunk.shape[0] :]
            shard_path = output_root / f"layer_{layer}" / f"shard_{shard_ids[layer]:05d}.pt"
            torch.save({"layer": layer, "activations": chunk.contiguous()}, shard_path)
            shard_ids[layer] += 1
            counts[layer] += int(chunk.shape[0])
            if not force and current.shape[0] < shard_size:
                break
        buffers[layer] = [current] if current.shape[0] else []

    batch: list[str] = []
    with torch.no_grad():
        for text in tqdm(texts_iter, total=limit, desc="collect texts"):
            batch.append(text)
            if len(batch) < batch_size:
                continue
            _collect_batch(model, tokenizer, batch, max_length, layers, selector, device, buffers)
            for layer in layers:
                flush(layer)
            batch = []
        if batch:
            _collect_batch(model, tokenizer, batch, max_length, layers, selector, device, buffers)
        for layer in layers:
            flush(layer, force=True)

    metadata = {
        "data_path": str(data_path),
        "layers": layers,
        "text_column": text_column,
        "prompt_column": prompt_column,
        "target_column": target_column,
        "template": template,
        "batch_size": batch_size,
        "max_length": max_length,
        "selector": selector,
        "limit": limit,
        "total_batches": total_batches,
        "counts": counts,
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _collect_batch(
    model,
    tokenizer,
    batch: list[str],
    max_length: int,
    layers: list[int],
    selector: str,
    device: str | torch.device,
    buffers: dict[int, list[torch.Tensor]],
) -> None:
    require_group_selfies_tokens_in_tokenizer(
        tokenizer,
        batch,
        context="activation collection",
    )
    encoded = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    attention_mask = encoded["attention_mask"]
    for layer in layers:
        state_index = layer + 1
        if state_index >= len(hidden_states):
            raise IndexError(f"Layer {layer} not available in hidden_states")
        selected = _select_activations(hidden_states[state_index], attention_mask, selector)
        buffers[layer].append(selected)
