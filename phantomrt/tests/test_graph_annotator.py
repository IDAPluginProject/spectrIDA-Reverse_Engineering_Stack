"""Atlas -> graph annotation edge. Query-construction is unit-tested with a fake
driver (no DB needed); a guarded round-trip runs against a live Neo4j if present."""
import json

import pytest

from atlas.analysis.graph_annotator import GraphAnnotator, _coerce


# ── fake neo4j driver that records queries and returns canned records ────────
class _Result:
    def __init__(self, rec): self._rec = rec
    def single(self): return self._rec


class _Session:
    def __init__(self, drv): self.drv = drv
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def run(self, query, **params):
        self.drv.calls.append((query, params))
        return _Result(self.drv.responses.pop(0))


class _Driver:
    def __init__(self, responses): self.responses = list(responses); self.calls = []
    def session(self): return _Session(self)
    def close(self): pass


def test_coerce_types():
    assert _coerce("s") == "s" and _coerce(3) == 3 and _coerce(True) is True
    assert _coerce([1, 2, 3]) == [1, 2, 3]                 # list of prims kept
    assert json.loads(_coerce({"a": 1})) == {"a": 1}       # dict -> json string


def test_annotate_prefixes_and_sets_props():
    drv = _Driver([{"n": 1}])
    ann = GraphAnnotator(driver=drv)
    n = ann.annotate("bin.nso", {"reachable": True, "atlas_x": 5}, addr=1234)
    assert n == 1
    query, params = drv.calls[0]
    assert "SET f += $props" in query and "Function {binary:$binary, addr:$addr}" in query
    assert params["binary"] == "bin.nso" and params["addr"] == 1234
    props = params["props"]
    assert props["atlas_reachable"] is True     # prefixed
    assert props["atlas_x"] == 5                 # not double-prefixed
    assert "atlas_updated" in props              # timestamp stamped


def test_match_requires_addr_or_name():
    ann = GraphAnnotator(driver=_Driver([]))
    with pytest.raises(ValueError):
        ann.annotate("b", {"reachable": True})


def test_read_returns_only_atlas_props():
    drv = _Driver([{"p": {"name": "foo", "addr": 9, "atlas_crashes": True}}])
    ann = GraphAnnotator(driver=drv)
    assert ann.read("b", addr=9) == {"atlas_crashes": True}


def test_clear_removes_atlas_keys_only():
    drv = _Driver([{"ks": ["atlas_a", "atlas_b"]}, {"n": 1}])
    ann = GraphAnnotator(driver=drv)
    assert ann.clear("b", addr=1) == 1
    remove_query = drv.calls[1][0]
    assert "REMOVE f.`atlas_a`, f.`atlas_b`" in remove_query


class _FakeFuzzEnv:
    def summary(self):
        return {"functions_covered": 42, "unique_crashes": 2,
                "crash_inputs": {"segv:3": "4141", "abort:3": "4242"}}


def test_annotate_fuzz_run_builds_facts():
    drv = _Driver([{"n": 1}])
    ann = GraphAnnotator(driver=drv)
    ann.annotate_fuzz_run("bin.nso", _FakeFuzzEnv(), addr=7)
    props = drv.calls[0][1]["props"]
    assert props["atlas_crashes"] is True
    assert set(props["atlas_crash_kinds"]) == {"segv", "abort"}
    assert props["atlas_functions_covered"] == 42


# ── guarded live round-trip against the real spectrIDA graph ─────────────────
def _live():
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "ChapterThree123"))
        with d.session() as s:
            s.run("RETURN 1").single()
        return d
    except Exception:
        return None


@pytest.mark.skipif(_live() is None, reason="live Neo4j not reachable")
def test_live_roundtrip_and_cleanup():
    d = _live()
    with d.session() as s:
        rec = s.run("MATCH (f:Function) RETURN f.binary AS b, f.addr AS a LIMIT 1").single()
    if not rec:
        pytest.skip("graph has no Function nodes")
    ann = GraphAnnotator(driver=d, log=lambda *a: None)
    assert ann.annotate(rec["b"], {"reachable": True, "crashes": False}, addr=rec["a"]) == 1
    got = ann.read(rec["b"], addr=rec["a"])
    assert got["atlas_reachable"] is True
    ann.clear(rec["b"], addr=rec["a"])
    assert ann.read(rec["b"], addr=rec["a"]) == {}      # cleaned up
    d.close()
