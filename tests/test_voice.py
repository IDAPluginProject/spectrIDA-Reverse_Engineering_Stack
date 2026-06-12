import random

from spectrida import voice


def test_at_least_a_million():
    assert voice.combinations() >= 1_000_000


def test_every_context_produces_a_line():
    for ctx in ("analyzing", "naming_done", "empty", "error", "welcome", "idle", "goodbye"):
        line = voice.quip(ctx)
        assert isinstance(line, str) and line


def test_variety():
    lines = {voice.quip("analyzing", rng=random.Random(i)) for i in range(30)}
    assert len(lines) > 5  # combinatorial, not a fixed line
