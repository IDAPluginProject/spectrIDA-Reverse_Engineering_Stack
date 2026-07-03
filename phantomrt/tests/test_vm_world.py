"""VMWorldEnv against a deterministic FakeVM: real-outcome featurization,
novelty/coverage, and auto-recovery rollback."""
import numpy as np

from atlas.environments.vm_world import VMWorldEnv, STATE_DIM
from tests.fake_vm import FakeVM


def test_step_returns_featurized_real_outcome():
    env = VMWorldEnv(FakeVM(), log=lambda *a: None)
    obs = env.reset()
    assert obs.shape == (STATE_DIM,)

    obs2, reward, done, info = env.step("echo hello")
    assert obs2.shape == (STATE_DIM,)
    assert reward > 0
    assert not done
    assert info["command"] == "echo hello"
    assert info["result"].command == "echo hello"      # the REAL result
    assert "coverage" in info


def test_featurization_is_deterministic():
    env = VMWorldEnv(FakeVM(), log=lambda *a: None)
    r = env.vm.run("ls -la")
    a = env.featurize("ls -la", r)
    b = env.featurize("ls -la", r)
    assert np.allclose(a, b)


def test_novelty_reward_decays_with_repetition():
    env = VMWorldEnv(FakeVM(), log=lambda *a: None)
    env.reset()
    _, r1, _, _ = env.step("echo same")
    _, r2, _, _ = env.step("echo same")     # same behavior signature
    assert r2 < r1                           # repeated behavior worth less
    assert len(env.seen) >= 1


def test_coverage_grows_with_distinct_behaviors():
    env = VMWorldEnv(FakeVM(), log=lambda *a: None)
    env.reset()
    for cmd in ["echo a", "ls", "pwd", "cat x", "grep y", "sort z"]:
        env.step(cmd)
    assert len(env.seen) >= 2                 # distinct behaviors discovered


def test_auto_recovery_rolls_back_on_brick():
    vm = FakeVM()
    vm.brick_on_danger = True
    env = VMWorldEnv(vm, log=lambda *a: None)
    env.reset()
    # a destructive command bricks the fake VM -> env must roll back
    env.step("rm -rf /tmp/x")
    assert vm.rolled_back == 1
    assert env.recoveries == 1
    assert vm.health_ok()                     # restored
