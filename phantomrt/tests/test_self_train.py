"""The core claims of the real learner:
  1. the loop is closed (the command the model selects is the command that runs);
  2. surprise = real prediction error, and training reduces it;
  3. it generalizes to held-out commands it never trained on (understand > memorize).
"""
import numpy as np
import torch

from atlas.environments.vm_world import VMWorldEnv
from atlas.agents.command_space import CommandProposer, embed_command
from atlas.training.self_train import SelfTrainer
from tests.fake_vm import FakeVM


def _trainer(epsilon=0.0, seed=0):
    vm = FakeVM()
    env = VMWorldEnv(vm, log=lambda *a: None)
    proposer = CommandProposer(vm)
    proposer.binaries = ["echo", "ls", "cat", "grep", "sort", "wc", "head"]
    proposer._discovered = True
    t = SelfTrainer(env, proposer, device="cpu", latent_dim=16, hidden=32,
                    batch_size=8, epsilon=epsilon, log=lambda *a: None, seed=seed)
    return t, env, vm


def test_loop_is_closed_selected_command_is_executed():
    t, env, vm = _trainer(epsilon=0.0)
    obs = env.reset()
    cands = ["echo a", "ls -la", "pwd", "cat /etc/hostname"]
    cmd = t._select(obs, cands)
    assert cmd in cands                                   # picks a real candidate
    _, _, _, info = env.step(cmd)
    assert info["command"] == cmd
    assert vm.calls[-1] == cmd                            # ...and that's what ran


def test_training_reduces_real_prediction_error():
    torch.manual_seed(0); np.random.seed(0)
    t, env, vm = _trainer()
    cmds = ["echo hi", "ls -la", "cat f", "grep x", "sort z", "wc -l", "head y", "pwd"]
    data = []
    obs = env.reset()
    for c in cmds:
        res = vm.run(c)
        nxt = env.featurize(c, res)
        a = embed_command(c)
        data.append((obs, a, nxt))
        t.replay.add(obs, a, nxt, 1.0)

    def mean_err():
        return float(np.mean([t._prediction_error(o, a, n) for o, a, n in data]))

    before = mean_err()
    for _ in range(200):
        t._train_batch()
    after = mean_err()
    assert after < before * 0.6                           # genuinely learned it


def test_generalizes_to_held_out_commands():
    torch.manual_seed(0); np.random.seed(0)
    t, env, vm = _trainer()
    train_cmds = ["echo 1", "echo 2", "ls a", "ls b", "cat x", "cat y",
                  "grep p", "grep q", "sort m", "wc -l"]
    held_cmds = ["echo 3", "ls c", "cat z", "grep r"]     # never trained on

    def err(cmds):
        es = []
        for c in cmds:
            res = vm.run(c); nxt = env.featurize(c, res)
            es.append(t._prediction_error(env._last, embed_command(c), nxt))
        return float(np.mean(es))

    held_before = err(held_cmds)
    for c in train_cmds:                                   # fill replay (train only)
        res = vm.run(c)
        t.replay.add(env._last, embed_command(c), env.featurize(c, res), 1.0)
    for _ in range(250):
        t._train_batch()
    held_after = err(held_cmds)

    # error drops on commands NEVER trained on => learned structure, not lookup
    assert held_after < held_before


def test_short_end_to_end_run_closes_loop_and_logs():
    t, env, vm = _trainer(epsilon=0.1)
    hist = t.run(max_steps=120, report_every=60)
    assert len(vm.calls) >= 120                            # it actually ran commands
    assert hist["coverage"][-1] >= 1
    assert len(hist["heldout_err"]) >= 1                   # generalization tracked
