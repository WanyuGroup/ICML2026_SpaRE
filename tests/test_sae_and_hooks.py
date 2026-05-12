from __future__ import annotations

# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spare_molgen.hooks import ActivationEditor, RuntimeEdit
from spare_molgen.sae import SAEConfig, SparseAutoencoder


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([IdentityBlock()])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyTransformer()

    def forward(self, x):
        return self.transformer.h[0](x)


def test_activation_editor_modifies_selected_position():
    model = TinyModel()
    x = torch.zeros(1, 3, 4)
    edit = RuntimeEdit(layer=0, vector=torch.ones(4), strength=2.0, positions="last")

    with ActivationEditor(model, [edit]):
        y = model(x)

    assert torch.equal(y[0, 0], torch.zeros(4))
    assert torch.equal(y[0, 1], torch.zeros(4))
    assert torch.equal(y[0, 2], torch.full((4,), 2.0))


def test_sae_forward_and_roundtrip(tmp_path):
    model = SparseAutoencoder(SAEConfig(input_dim=4, expansion_factor=2))
    z = torch.randn(3, 4)

    z_hat, h = model(z)

    assert z_hat.shape == z.shape
    assert h.shape == (3, 8)

    path = tmp_path / "sae.pt"
    model.save(path)
    loaded = SparseAutoencoder.load(path)
    assert loaded.config.input_dim == 4
    assert loaded.config.expansion_factor == 2
