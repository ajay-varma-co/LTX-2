import pytest

pytest.importorskip("torch")

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import LatentState


def make_state(batch=1, channels=2, frames=1, height=2, width=2, device="cpu"):
    latent = torch.zeros((batch, channels, frames, height, width), dtype=torch.float32, device=device)
    denoise_mask = torch.ones_like(latent)
    return LatentState(latent=latent, denoise_mask=denoise_mask)


def test_gaussian_noiser_shapes_and_sigma_zero():
    gen = torch.Generator(device="cpu").manual_seed(0)
    noiser = GaussianNoiser(generator=gen)

    state = make_state()

    # sigma=0 should return original latent
    out = noiser(state, noise_scale=0.0)
    assert out.latent.shape == state.latent.shape
    assert out.latent.dtype == state.latent.dtype
    assert out.latent.device.type == "cpu"
    assert torch.allclose(out.latent, state.latent)


def test_gaussian_noiser_positive_sigma_changes_values():
    gen = torch.Generator(device="cpu").manual_seed(123)
    noiser = GaussianNoiser(generator=gen)

    state = make_state()

    out = noiser(state, noise_scale=0.1)
    assert out.latent.shape == state.latent.shape
    assert torch.isfinite(out.latent).all()
    # With positive sigma and non-zero mask, expect non-equality
    assert not torch.allclose(out.latent, state.latent)
