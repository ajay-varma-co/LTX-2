import pytest

pytest.importorskip("torch")

import torch

from ltx_core.components.guiders import CFGGuider, CFGStarRescalingGuider, STGGuider, LtxAPGGuider


def make_pred(shape=(1, 2, 2, 2)):
    return torch.randn(shape, dtype=torch.float32)


def test_cfg_guider_behavior():
    cond = make_pred()
    uncond = make_pred()

    # scale=1 => delta zero
    guider = CFGGuider(scale=1.0)
    delta = guider.delta(cond, uncond)
    assert torch.allclose(delta, torch.zeros_like(cond))
    assert not guider.enabled()

    # scale>1: moves towards cond
    guider = CFGGuider(scale=3.0)
    delta = guider.delta(cond, uncond)
    combined = uncond + delta
    # Directional check: cosine similarity with (cond - uncond) should be positive
    direction = (cond - uncond).flatten()
    step = (combined - uncond).flatten()
    cos = torch.dot(direction, step) / (direction.norm() * step.norm() + 1e-8)
    assert cos > 0.99
    assert guider.enabled()


def test_cfg_star_rescaling_shapes_and_finiteness():
    cond = make_pred()
    uncond = make_pred()

    guider = CFGStarRescalingGuider(scale=2.0)
    delta = guider.delta(cond, uncond)
    assert delta.shape == cond.shape
    assert torch.isfinite(delta).all()
    assert guider.enabled()


def test_stg_guider_zero_and_positive_scale():
    pos = make_pred()
    perturbed = make_pred()

    guider = STGGuider(scale=0.0)
    d0 = guider.delta(pos, perturbed)
    assert torch.allclose(d0, torch.zeros_like(pos))
    assert not guider.enabled()

    guider = STGGuider(scale=1.5)
    d1 = guider.delta(pos, perturbed)
    assert d1.shape == pos.shape
    assert torch.isfinite(d1).all()
    assert guider.enabled()


essentially_zero = 1e-5


def test_ltx_apg_guider_behavior_and_shapes():
    cond = make_pred()
    uncond = make_pred()

    guider = LtxAPGGuider(scale=2.0, eta=0.5, norm_threshold=0.0)
    delta = guider.delta(cond, uncond)
    assert delta.shape == cond.shape
    assert torch.isfinite(delta).all()
    assert guider.enabled()

    # When scale=1.0 and zero threshold, delta can be small but not forced zero
    guider_unit = LtxAPGGuider(scale=1.0, eta=1.0, norm_threshold=0.0)
    d_unit = guider_unit.delta(cond, uncond)
    assert d_unit.shape == cond.shape
    assert torch.isfinite(d_unit).all()
