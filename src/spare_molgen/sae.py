from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from spare_molgen.activation_store import count_activations, infer_activation_dim, iter_activation_batches


@dataclass
class SAEConfig:
    input_dim: int
    expansion_factor: int = 40
    latent_dim: int | None = None
    l1_coefficient: float = 1e-5
    code_normalization: str = "l2"


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim or config.input_dim * config.expansion_factor
        self.encoder = nn.Linear(config.input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, config.input_dim)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(z))
        if self.config.code_normalization == "l2":
            return torch.nn.functional.normalize(h, p=2, dim=-1)
        if self.config.code_normalization == "max":
            denom = h.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            return h / denom
        if self.config.code_normalization == "none":
            return h
        raise ValueError(f"Unsupported code normalization: {self.config.code_normalization}")

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(z)
        return self.decode(h), h

    def loss(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_hat, h = self(z)
        recon = torch.mean((z - z_hat) ** 2)
        sparsity = torch.mean(torch.abs(h))
        loss = recon + self.config.l1_coefficient * sparsity
        return loss, {"loss": loss.detach(), "recon": recon.detach(), "l1": sparsity.detach()}

    def decoder_vectors(self, feature_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.weight[:, feature_ids].transpose(0, 1).contiguous()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SparseAutoencoder":
        payload = torch.load(path, map_location=map_location)
        model = cls(SAEConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model


def make_optimizer(
    parameters,
    optimizer: str,
    lr: float,
    weight_decay: float = 0.0,
):
    name = optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "muon":
        try:
            from muon import Muon
        except ImportError as exc:
            raise RuntimeError(
                "Muon optimizer requested but no importable 'muon' package was found. "
                "Install your chosen Muon implementation or use --optimizer adamw."
            ) from exc
        return Muon(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer}")


def train_sae_from_shards(
    activations: str | Path,
    output: str | Path,
    input_dim: int | None = None,
    expansion_factor: int = 40,
    latent_dim: int | None = None,
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 1e-4,
    l1: float = 1e-5,
    optimizer: str = "adamw",
    code_normalization: str = "l2",
    device: str | torch.device | None = None,
) -> SparseAutoencoder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = input_dim or infer_activation_dim(activations)
    model = SparseAutoencoder(
        SAEConfig(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            latent_dim=latent_dim,
            l1_coefficient=l1,
            code_normalization=code_normalization,
        )
    ).to(device)
    opt = make_optimizer(model.parameters(), optimizer=optimizer, lr=lr)
    total = count_activations(activations)
    steps = max(1, total // batch_size)

    model.train()
    for epoch in range(epochs):
        progress = tqdm(
            iter_activation_batches(activations, batch_size=batch_size, device=device),
            total=steps,
            desc=f"sae epoch {epoch + 1}/{epochs}",
        )
        for batch in progress:
            opt.zero_grad(set_to_none=True)
            loss, metrics = model.loss(batch)
            loss.backward()
            opt.step()
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.4g}",
                recon=f"{metrics['recon'].item():.4g}",
                l1=f"{metrics['l1'].item():.4g}",
            )
    model.save(output)
    return model

