"""End of the hard edge: Atlas's curiosity loop fuzzes an EMULATED function
(no OS, arm64) and its crash finding flows into the graph annotator."""
import pytest

ks = pytest.importorskip("keystone")

from atlas.analysis.unicorn_harness import EmulatedFuzzEnv, EMU_STATE_DIM
from atlas.analysis.binary_fuzz import make_token, InputProposer, embed_input, input_family
from atlas.analysis.graph_annotator import GraphAnnotator
from atlas.training.self_train import SelfTrainer

ARM64_DEREF = bytes(ks.Ks(ks.KS_ARCH_ARM64, ks.KS_MODE_LITTLE_ENDIAN)
                    .asm("ldr x1, [x0]; ldr x2, [x1]; ret")[0])


def test_env_step_detects_emulated_crash():
    env = EmulatedFuzzEnv(ARM64_DEREF, "arm64", log=lambda *a: None)
    env.reset()
    _, _, _, info = env.step(make_token((0xDEAD_BEEF_0000).to_bytes(8, "little")))
    assert info["crashed"]
    assert env.get_observation_dim() == EMU_STATE_DIM
    assert env.summary()["unique_crashes"] >= 1


def test_atlas_loop_finds_emulated_crash():
    env = EmulatedFuzzEnv(ARM64_DEREF, "arm64", log=lambda *a: None)
    trainer = SelfTrainer(env, InputProposer(), device="cpu", latent_dim=16, hidden=32,
                          batch_size=8, epsilon=0.4, log=lambda *a: None,
                          embed_fn=embed_input, family_fn=input_family)
    trainer.run(max_steps=60, report_every=60)
    assert env.summary()["unique_crashes"] >= 1     # the loop found a crashing input


class _Drv:
    def __init__(self): self.calls = []
    def session(self): return self
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def run(self, q, **p): self.calls.append((q, p)); return _R()
    def close(self): pass
class _R:
    def single(self): return {"n": 1}


def test_emulated_findings_annotate_graph():
    env = EmulatedFuzzEnv(ARM64_DEREF, "arm64", log=lambda *a: None,
                          binary="main.nso", addr=0x71000e3518)
    env.reset()
    env.step(make_token((0xDEAD_BEEF_0000).to_bytes(8, "little")))    # cause a crash
    ann = GraphAnnotator(driver=_Drv())
    n = ann.annotate_fuzz_run("main.nso", env, addr=0x71000e3518)
    assert n == 1
    props = ann._driver.calls[0][1]["props"]
    assert props["atlas_crashes"] is True and props["atlas_reachable"] is True
