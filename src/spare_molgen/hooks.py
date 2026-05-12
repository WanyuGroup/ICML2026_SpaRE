from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from spare_molgen.layers import get_layer


Positions = Literal["all", "last"] | tuple[int, ...]


@dataclass
class RuntimeEdit:
    layer: int
    vector: torch.Tensor
    strength: float = 1.0
    positions: Positions = "last"


def _hidden_from_output(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


def _replace_hidden(output: object, hidden: torch.Tensor) -> object:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


def _resolve_positions(positions: Positions, seq_len: int) -> list[int]:
    if positions == "all":
        return list(range(seq_len))
    if positions == "last":
        return [seq_len - 1]
    resolved: list[int] = []
    for pos in positions:
        resolved_pos = pos if pos >= 0 else seq_len + pos
        if 0 <= resolved_pos < seq_len:
            resolved.append(resolved_pos)
    return resolved


class ActivationEditor:
    """Context manager that injects activation-space edits into transformer layers."""

    def __init__(
        self,
        model: nn.Module,
        edits: list[RuntimeEdit],
        layer_path: str | None = None,
    ) -> None:
        self.model = model
        self.edits = edits
        self.layer_path = layer_path
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationEditor":
        edits_by_layer: dict[int, list[RuntimeEdit]] = {}
        for edit in self.edits:
            edits_by_layer.setdefault(edit.layer, []).append(edit)

        for layer_index, layer_edits in edits_by_layer.items():
            layer = get_layer(self.model, layer_index, self.layer_path)
            self._handles.append(layer.register_forward_hook(self._make_hook(layer_edits)))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_edits: list[RuntimeEdit]):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
            hidden = _hidden_from_output(output)
            edited = hidden.clone()
            seq_len = edited.shape[1]
            for edit in layer_edits:
                positions = _resolve_positions(edit.positions, seq_len)
                if not positions:
                    continue
                vector = edit.vector.to(device=edited.device, dtype=edited.dtype)
                if vector.ndim != 1 or vector.shape[0] != edited.shape[-1]:
                    raise ValueError(
                        f"Edit vector shape {tuple(vector.shape)} does not match "
                        f"hidden dimension {edited.shape[-1]}"
                    )
                edited[:, positions, :] = edited[:, positions, :] + edit.strength * vector
            return _replace_hidden(output, edited)

        return hook

