"""Growth + anti-stagnation: function-preserving capacity, and telling
learnable-underfit apart from unlearnable noise (the noisy-TV guard)."""
import torch

from atlas.training.growth import (
    GrowableCorrector, CompetenceTracker, GrowthController,
)


def test_grow_is_function_preserving_and_adds_params():
    c = GrowableCorrector(latent_dim=8, action_dim=4, hidden=16)
    z = torch.randn(3, 8)
    a = torch.randn(3, 4)

    before = c(z, a)
    assert torch.allclose(before, torch.zeros_like(before))  # starts as identity (0)
    p0 = c.num_params()

    c.grow()
    after = c(z, a)
    # zero-init block => output unchanged at the instant of growth
    assert torch.allclose(before, after, atol=1e-6)
    assert c.num_params() > p0                                # but capacity increased

    c.grow()
    assert c.num_params() > p0
    assert torch.allclose(c(z, a), before, atol=1e-6)


def _feed(tracker, family, values):
    for v in values:
        tracker.update(family, v)


def test_tracker_discriminates_learnable_vs_noise_vs_mastered():
    t = CompetenceTracker(window=40, master_thresh=0.02, high_thresh=0.08)

    # learnable-underfit: high, LOW-variance, not improving -> needs capacity
    _feed(t, "det", [0.30 + 0.001 * (i % 2) for i in range(40)])
    assert t.needs_capacity("det")
    assert not t.is_noisy("det")

    # noisy-TV: high mean, HIGH-variance, not improving -> noisy, do NOT grow
    _feed(t, "noise", [0.02 if i % 2 else 0.60 for i in range(40)])
    assert t.is_noisy("noise")
    assert not t.needs_capacity("noise")

    # improving: error clearly decreasing -> learning, leave alone
    _feed(t, "learn", [0.5 - 0.01 * i for i in range(40)])
    assert t.is_improving("learn")
    assert not t.needs_capacity("learn")

    # mastered: low error
    _feed(t, "done", [0.005 for _ in range(40)])
    assert t.is_mastered("done")


def test_controller_grows_on_learnable_not_on_noise():
    c = GrowableCorrector(8, 4, hidden=16)
    t = CompetenceTracker(window=40, high_thresh=0.08)
    ctrl = GrowthController(c, t, log=lambda *a: None, cooldown=0)

    # only noise present -> must NOT grow, and curiosity decayed
    for i in range(40):
        t.update("noise", 0.02 if i % 2 else 0.60)
    assert ctrl.maybe_grow(step=100) is False
    assert ctrl.curiosity_weight("noise") < 1.0

    # add a learnable-underfit family -> must grow now
    for i in range(40):
        t.update("det", 0.30 + 0.001 * (i % 2))
    assert ctrl.maybe_grow(step=200) is True
    assert len(c.blocks) == 1
    assert ctrl.curiosity_weight("det") == 1.0
