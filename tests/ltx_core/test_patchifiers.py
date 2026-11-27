import pytest

pytest.importorskip("torch")

import torch

from ltx_core.components import patchifiers as pf
from ltx_core.types import VideoLatentShape, AudioLatentShape


def test_video_patchifier_roundtrip_small_tensor():
    patch_size = 2
    patchifier = pf.VideoLatentPatchifier(patch_size=patch_size)

    # Tiny tensor: B=1, C=2, F=1 (temporal), H=4, W=4 to be divisible by 2
    x = torch.arange(1 * 2 * 1 * 4 * 4, dtype=torch.float32).view(1, 2, 1, 4, 4)

    # Expected shapes
    out_shape = VideoLatentShape(batch=1, channels=2, frames=1, height=4, width=4)

    patches = patchifier.patchify(x)
    # patchify: (b c (f p1) (h p2) (w p3)) -> b (f h w) (c p1 p2 p3)
    # with p1=1, p2=p3=2 => seq = 1*2*2 = 4, dim = 2*1*2*2 = 8
    assert patches.shape == (1, 4, 8)

    x_recon = patchifier.unpatchify(patches, out_shape)
    assert x_recon.shape == x.shape
    assert x_recon.dtype == x.dtype
    assert x_recon.device == x.device
    assert torch.allclose(x_recon, x)


def test_video_patchifier_get_patch_grid_bounds_dims_and_values():
    patch_size = 2
    patchifier = pf.VideoLatentPatchifier(patch_size=patch_size)
    out_shape = VideoLatentShape(batch=1, channels=2, frames=1, height=4, width=4)

    coords = patchifier.get_patch_grid_bounds(out_shape, device=torch.device("cpu"))
    # Shape: (B, 3, num_patches, 2)
    assert coords.shape == (1, 3, 4, 2)

    # Check the first patch bounds along H and W for correctness
    # Expected: temporal bounds [0,1), height [0,2), width [0,2)
    first_patch = coords[0, :, 0]
    assert torch.equal(first_patch[0], torch.tensor([0, 1]))
    assert torch.equal(first_patch[1], torch.tensor([0, 2]))
    assert torch.equal(first_patch[2], torch.tensor([0, 2]))


def test_audio_patchifier_roundtrip_small_tensor():
    patchifier = pf.AudioPatchifier(patch_size=2, sample_rate=16000, hop_length=160, audio_latent_downsample_factor=4)

    # Tiny tensor: B=1, C=2, T=4, F=4
    x = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)

    out_shape = AudioLatentShape(batch=1, channels=2, frames=4, mel_bins=4)

    patches = patchifier.patchify(x)
    # patchify: b c t f -> b t (c f) => (1, 4, 8)
    assert patches.shape == (1, 4, 8)

    x_recon = patchifier.unpatchify(patches, out_shape)
    assert x_recon.shape == x.shape
    assert torch.allclose(x_recon, x)

    # get_patch_grid_bounds: (B, 1, T, 2) with times in seconds
    coords = patchifier.get_patch_grid_bounds(out_shape, device=torch.device("cpu"))
    assert coords.shape == (1, 1, 4, 2)
    # With hop_length=160, sr=16000, downsample=4, causal shift applied,
    # first start should be 0.0s and first end should be 0.01s
    starts = coords[0, 0, :, 0]
    ends = coords[0, 0, :, 1]
    assert torch.isclose(starts[0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(ends[0], torch.tensor(0.01), atol=1e-6)
