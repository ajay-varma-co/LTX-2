import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch")

import torch

from ltx_core.components import schedulers as sched


def assert_tensor_cpu_finite(x: torch.Tensor):
    assert x.device.type == "cpu"
    assert torch.isfinite(x).all(), "Tensor contains NaN or Inf"


def test_ltx2_scheduler_basic_properties():
    steps = 12
    scheduler = sched.LTX2Scheduler()
    sigmas = scheduler.execute(steps)

    # Length must be steps + 1 as implemented
    assert isinstance(sigmas, torch.Tensor)
    assert sigmas.ndim == 1
    assert len(sigmas) == steps + 1
    assert_tensor_cpu_finite(sigmas)

    # Expected bounds: [0, 1]
    assert (sigmas >= 0).all()
    assert (sigmas <= 1).all()

    # Monotone non-increasing from 1 to ~0
    diffs = torch.diff(sigmas)
    assert (diffs <= 1e-6).all(), f"Sigmas not non-increasing: {sigmas}"

    # Endpoints
    assert torch.isclose(sigmas[0], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(sigmas[-1], torch.tensor(0.0), atol=1e-3)


@pytest.mark.parametrize(
    "steps,threshold_noise,linear_steps",
    [
        (8, 0.025, None),
        (9, 0.05, 4),
        (2, 0.025, None),
    ],
)
def test_linear_quadratic_scheduler_monotone_and_bounds(steps, threshold_noise, linear_steps):
    scheduler = sched.LinearQuadraticScheduler()
    sigmas = scheduler.execute(steps, threshold_noise=threshold_noise, linear_steps=linear_steps)

    # Length should be steps + 1 per implementation
    assert sigmas.shape[0] == steps + 1
    assert_tensor_cpu_finite(sigmas)

    # Bounds and monotonicity
    assert (sigmas >= 0).all()
    assert (sigmas <= 1).all()

    diffs = torch.diff(sigmas)
    # Non-increasing
    assert (diffs <= 1e-6).all(), f"Sigmas not non-increasing: {sigmas}"

    # Endpoints
    assert torch.isclose(sigmas[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(sigmas[-1], torch.tensor(0.0), atol=1e-5)


def test_beta_scheduler_properties():
    # Beta scheduler depends on scipy; skip this test if scipy is unavailable
    pytest.importorskip("scipy")

    scheduler = sched.BetaScheduler()
    steps = 10
    sigmas = scheduler.execute(steps, alpha=0.6, beta=0.6)

    # Warning in docstring: len can be <= steps + 1 due to de-dup timesteps
    assert isinstance(sigmas, torch.Tensor)
    assert 2 <= sigmas.numel() <= steps + 1
    assert_tensor_cpu_finite(sigmas)

    # Bounds
    assert (sigmas >= 0).all()
    assert (sigmas <= 1).all()

    # Should be non-increasing across indices
    diffs = torch.diff(sigmas)
    assert (diffs <= 1e-6).all(), f"Sigmas not non-increasing: {sigmas}"

    # First value should be close to 1, last equals 0 by construction
    assert sigmas[0] <= 1.0 + 1e-6 and sigmas[0] >= 0.0
    assert torch.isclose(sigmas[-1], torch.tensor(0.0), atol=1e-6)

