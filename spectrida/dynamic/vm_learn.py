"""The self-directed VM learner — Atlas's original curiosity engine.

Turns the learning agent loose in an isolated throwaway WSL VM: it proposes and
runs real commands, predicts their outcomes, learns from the real results
(surprise = prediction error), and grows its own capacity when it hits a
learnable-but-underfit wall. This is the research/self-improvement side of Atlas,
exposed here so it can be driven from the same MCP surface.

Unlike the emulate/hunt/live tools this does NOT annotate the function graph — it
explores a whole VM, not a single binary — so it returns learning statistics
(prediction error trend, coverage, growth events, honest per-family competence).
Heavy: provisions a WSL2 distro and (ideally) uses a GPU. Long-running.
"""
from __future__ import annotations


def learn_vm(steps: int = 500, device: str = "cpu", report_every: int = 100,
             on_progress=None) -> dict:
    """Run the self-directed learner in an isolated VM for ``steps`` iterations."""
    from atlas.vm.wsl_vm import WslVM
    from atlas.environments.vm_world import VMWorldEnv
    from atlas.agents.command_space import CommandProposer
    from atlas.training.self_train import SelfTrainer

    log = (lambda m: on_progress(m)) if on_progress else (lambda m: None)

    vm = WslVM(log=log)
    vm.provision()
    if not vm.health_ok():
        return {"error": "isolated VM unhealthy after provisioning"}
    if not vm.has_snapshot("base"):
        vm.snapshot("base")   # rollback point for auto-recovery

    env = VMWorldEnv(vm, log=log)
    proposer = CommandProposer(vm)
    trainer = SelfTrainer(env, proposer, device=device, log=log)
    trainer.run(max_steps=steps, report_every=report_every)

    tr = trainer.tracker
    hist = trainer.history
    mastered = [f for f in tr.err if tr.is_mastered(f)]
    noisy = [f for f in tr.err if tr.is_noisy(f)]
    return {
        "status": "complete",
        "steps": steps,
        "families_seen": len(tr.err),
        "mastered": mastered,
        "noisy_skipped": noisy,           # unlearnable — correctly not chased
        "coverage": len(env.seen),        # distinct behaviors discovered
        "growth_events": len(trainer.growth.events),
        "recoveries": env.recoveries,
        "params": sum(p.numel() for p in trainer.wm.parameters()),
        # honest learning signal — held-out generalization vs train error
        "train_err_trend": hist.get("train_err", [])[-5:],
        "heldout_err_trend": hist.get("heldout_err", [])[-5:],
    }
