from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch.nn as nn


COMMON_LAYER_PATHS = (
    "transformer.h",
    "gpt_neox.layers",
    "model.layers",
    "model.decoder.layers",
    "decoder.layers",
)


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def get_transformer_layers(model: nn.Module, layer_path: str | None = None) -> Sequence[nn.Module]:
    paths = (layer_path,) if layer_path else COMMON_LAYER_PATHS
    for path in paths:
        if not path:
            continue
        try:
            layers = get_nested_attr(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            return layers
    tried = ", ".join(path for path in paths if path)
    raise ValueError(
        "Could not locate transformer layers. "
        f"Tried: {tried}. Pass --layer-path for this architecture."
    )


def get_layer(model: nn.Module, index: int, layer_path: str | None = None) -> nn.Module:
    layers = get_transformer_layers(model, layer_path=layer_path)
    if index < 0 or index >= len(layers):
        raise IndexError(f"Layer index {index} is outside [0, {len(layers) - 1}]")
    return layers[index]

