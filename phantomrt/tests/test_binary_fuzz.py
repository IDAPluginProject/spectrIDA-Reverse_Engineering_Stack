"""Binary-fuzz env: token/embedding/proposer logic (hermetic) + a real
compile-and-crash integration test guarded on the isolated VM being present."""
import numpy as np
import pytest

from atlas.analysis.binary_fuzz import (
    make_token, parse_token, embed_input, input_family, FuzzResult,
    InputProposer, BinaryFuzzEnv, FUZZ_ACTION_DIM,
)


def test_token_roundtrip():
    for payload in [b"", b"A" * 100, b"C%n%n", bytes(range(256))]:
        mode, back = parse_token(make_token(payload, "stdin"))
        assert mode == "stdin" and back == payload


def test_embed_is_fixed_dim_and_deterministic():
    t = make_token(b"AAAA%n", "stdin")
    a, b = embed_input(t), embed_input(t)
    assert a.shape == (FUZZ_ACTION_DIM,)
    assert np.allclose(a, b)
    # a long input and a short input embed differently (structure, not identity)
    assert not np.allclose(embed_input(make_token(b"A" * 200)), a)


def test_input_family_classification():
    assert input_family(make_token(b"")) == "empty"
    assert input_family(make_token(b"C%n%x")) == "format"
    assert input_family(make_token(b"A" * 80)) == "long"
    assert input_family(make_token(b"ZZZZ")) == "repeat"
    assert input_family(make_token(bytes([0, 1, 2, 200, 255]))) == "binary"


def test_proposer_emits_valid_tokens_and_grows_corpus():
    p = InputProposer()
    toks = p.propose(16)
    assert len(toks) >= 1
    for t in toks:                         # every proposal is a decodable token
        parse_token(t)
    n0 = len(p.corpus)
    p.observe(make_token(b"newseed"), FuzzResult(0, "", frozenset(), new_coverage=2))
    assert len(p.corpus) == n0 + 1         # coverage gain -> corpus grows
    p.observe(make_token(b"dull"), FuzzResult(0, "", frozenset(), new_coverage=0))
    assert len(p.corpus) == n0 + 1         # no gain -> no growth


def test_parse_execution_output():
    sample = "junk===RC:139\n===COV\n0x111\n0x222\n===OUT\nsome output"
    rc, cov, out = BinaryFuzzEnv._parse(sample)
    assert rc == 139
    assert cov == {"0x111", "0x222"}
    assert out == "some output"


def test_crash_kind_detection():
    assert FuzzResult(139, "", frozenset(), 0).crash_kind == "segv"
    assert FuzzResult(134, "", frozenset(), 0).crash_kind == "abort"
    assert FuzzResult(0, "", frozenset(), 0).crashed is False


# ── real integration: compile the target in the VM and actually crash it ─────
def _vm_available():
    try:
        from atlas.vm import WslVM
        return WslVM().exists()
    except Exception:
        return False


@pytest.mark.skipif(not _vm_available(), reason="isolated atlas-vm not provisioned")
def test_real_compile_and_crash():
    from atlas.vm import WslVM
    env = BinaryFuzzEnv(WslVM(log=lambda *a: None), log=lambda *a: None)

    # benign input: routes through a handler, no crash, some coverage
    _, _, _, ok = env.step(make_token(b"Bhello"))
    assert not ok["crashed"]
    assert ok["coverage"] >= 1

    # long 'A' input overflows the 16-byte buffer -> real crash (segv/abort)
    crashed = False
    for _ in range(3):
        _, _, _, info = env.step(make_token(b"A" * 200))
        if info["crashed"]:
            crashed = True
            break
    assert crashed
    assert len(env.crash_inputs) >= 1
