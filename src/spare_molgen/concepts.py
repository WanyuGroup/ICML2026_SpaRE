from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from spare_molgen.activation_store import iter_activation_batches
from spare_molgen.sae import SparseAutoencoder


@dataclass
class ConceptVector:
    name: str
    kind: str
    layer: int
    vector: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["vector"] = self.vector.detach().cpu()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ConceptVector":
        payload = torch.load(path, map_location=map_location)
        return cls(
            name=payload["name"],
            kind=payload["kind"],
            layer=int(payload["layer"]),
            vector=payload["vector"],
            metadata=dict(payload.get("metadata", {})),
        )


def _collect_codes(
    sae: SparseAutoencoder,
    activations: str | Path,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(sae.parameters()).device
    codes: list[torch.Tensor] = []
    sae.eval()
    with torch.no_grad():
        for batch in iter_activation_batches(
            activations,
            batch_size=batch_size,
            shuffle_shards=False,
            device=device,
        ):
            codes.append(sae.encode(batch).detach().cpu())
    if not codes:
        raise ValueError(f"No activation batches found: {activations}")
    return torch.cat(codes, dim=0)


def build_local_concept(
    sae: SparseAutoencoder,
    activations: str | Path,
    name: str,
    layer: int,
    token: str | None = None,
    threshold: float = 0.5,
    min_fraction: float = 1.0,
    batch_size: int = 1024,
) -> ConceptVector:
    codes = _collect_codes(sae, activations, batch_size=batch_size)
    active_fraction = (codes >= threshold).float().mean(dim=0)
    feature_ids = torch.nonzero(active_fraction >= min_fraction, as_tuple=False).flatten()
    if feature_ids.numel() == 0:
        raise ValueError(
            "No local concept features met the threshold. "
            "Try lowering --threshold or --min-fraction."
        )
    vectors = sae.decoder_vectors(feature_ids.to(next(sae.parameters()).device)).detach().cpu()
    vector = vectors.mean(dim=0)
    return ConceptVector(
        name=name,
        kind="local",
        layer=layer,
        vector=vector,
        metadata={
            "threshold": threshold,
            "min_fraction": min_fraction,
            "feature_count": int(feature_ids.numel()),
            "feature_ids": feature_ids.tolist(),
            "token": token,
        },
    )


def build_global_concept(
    sae: SparseAutoencoder,
    positive_activations: str | Path,
    negative_activations: str | Path,
    name: str,
    layer: int,
    threshold: float = 0.5,
    batch_size: int = 1024,
) -> ConceptVector:
    device = next(sae.parameters()).device
    pos_codes = _collect_codes(sae, positive_activations, batch_size=batch_size, device=device)
    neg_codes = _collect_codes(sae, negative_activations, batch_size=batch_size, device=device)

    pos_all = (pos_codes >= threshold).all(dim=0)
    neg_all = (neg_codes < threshold).all(dim=0)
    selected = torch.nonzero(pos_all & neg_all, as_tuple=False).flatten()
    if selected.numel() > 0:
        direction_codes = torch.zeros(pos_codes.shape[1], dtype=torch.float32)
        direction_codes[selected] = pos_codes[:, selected].mean(dim=0) - neg_codes[:, selected].mean(dim=0)
        vector = F.linear(direction_codes.to(device), sae.decoder.weight, bias=None).detach().cpu()
        mode = "thresholded_decoder_difference"
    else:
        direction_codes = pos_codes.mean(dim=0) - neg_codes.mean(dim=0)
        vector = F.linear(direction_codes.to(device), sae.decoder.weight, bias=None).detach().cpu()
        mode = "mean_decoder_difference"

    return ConceptVector(
        name=name,
        kind="global",
        layer=layer,
        vector=vector,
        metadata={
            "threshold": threshold,
            "feature_count": int(selected.numel()),
            "feature_ids": selected.tolist(),
            "fallback_or_mode": mode,
        },
    )
