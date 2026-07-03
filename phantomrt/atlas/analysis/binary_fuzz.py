"""
BinaryFuzzEnv — point Atlas at ONE binary and let it hunt crashes.

This is the target-binary variant of the VM world: the action is an INPUT fed to
the target, the outcome is (crash? which signal? which functions ran?), and the
reward is coverage-guided — inputs that reach NEW functions or NEW crashes are
worth more. The world model learns input -> behavior and curiosity steers toward
unexplored code + crashes.

Coverage is function-level, via gcc `-finstrument-functions` plus a small shim
that records each function entered and dumps the set on exit OR on a fatal signal
(so crashing runs still report the path that led to the crash). Function-level
coverage is deliberate: it is exactly the granularity a spectrIDA-style function
graph speaks, so this plugs into that later.

Works on:
  * a provided C source (compiled here, instrumented), or
  * the built-in vulnerable demo target (default).
Prebuilt binaries can't be function-instrumented this way — for those you'd fall
back to black-box (crash-only) coverage; that's a documented follow-on.
"""

from __future__ import annotations

import base64
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

FUZZ_ACTION_DIM = 24
FUZZ_STATE_DIM = 20

# ── coverage shim (compiled WITHOUT instrumentation; dumps on exit or signal) ─
_COV_SHIM = r"""
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
static void* seen[8192]; static int nseen = 0;
__attribute__((no_instrument_function)) static void dump(void){
    const char* p = getenv("COV_FILE"); if(!p) p = "/tmp/atlas_fuzz/cov";
    FILE* f = fopen(p, "w"); if(!f) return;
    for(int i=0;i<nseen;i++) fprintf(f, "%p\n", seen[i]);
    fclose(f);
}
__attribute__((no_instrument_function)) static void onsig(int s){ dump(); _exit(128+s); }
__attribute__((no_instrument_function)) __attribute__((constructor))
static void init(void){
    atexit(dump);
    signal(SIGSEGV,onsig); signal(SIGABRT,onsig);
    signal(SIGBUS,onsig);  signal(SIGFPE,onsig);
}
__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void* fn, void* site){
    for(int i=0;i<nseen;i++) if(seen[i]==fn) return;
    if(nseen<8192) seen[nseen++]=fn;
}
__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void* fn, void* site){ (void)fn; (void)site; }
"""

# ── built-in vulnerable demo target (compiles on modern glibc; has structure) ─
DEFAULT_TARGET = r"""
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static void handle_a(const char* s){ char b[16]; strcpy(b, s); printf("A:%s\n", b); }
static void handle_b(const char* s){ printf("B:len=%zu\n", strlen(s)); }
static void handle_c(const char* s){ if(s[0]) printf(s); putchar('\n'); }
static void handle_d(const char* s){ int n = atoi(s+1); char* p = malloc(n>0?n:1);
    if(p){ memset(p, 'x', n); free(p); } printf("D:%d\n", n); }
static int route(const char* s){
    if(!s[0]) return 0;
    switch(s[0]){
        case 'A': handle_a(s+1); return 1;   /* stack overflow on long input */
        case 'B': handle_b(s+1); return 2;
        case 'C': handle_c(s+1); return 3;   /* format string */
        case 'D': handle_d(s);   return 4;   /* size-driven alloc */
        default:  return -1;
    }
}
int main(void){ char in[256];
    if(!fgets(in, sizeof(in), stdin)) return 0;
    in[strcspn(in, "\n")] = 0;
    return route(in) >= 0 ? 0 : 2;
}
"""


# ── action token: "mode|base64(payload)" ─────────────────────────────────────
def make_token(payload: bytes, mode: str = "stdin") -> str:
    return f"{mode}|{base64.b64encode(payload).decode()}"


def parse_token(token: str) -> tuple[str, bytes]:
    mode, b64 = token.split("|", 1)
    return mode, base64.b64decode(b64)


_FMT = (b"%s", b"%n", b"%x", b"%p")


def embed_input(token: str) -> np.ndarray:
    """Structured, compositional embedding of an input (so the model generalizes
    across inputs instead of memorizing exact byte strings)."""
    mode, p = parse_token(token)
    v = np.zeros(FUZZ_ACTION_DIM, dtype=np.float32)
    n = len(p)
    v[0] = 1.0 if mode == "stdin" else 0.0
    v[1] = 1.0 if mode == "argv" else 0.0
    v[2] = min(n / 256.0, 1.0)
    if n:
        arr = np.frombuffer(p, dtype=np.uint8)
        v[3] = np.mean((arr >= 32) & (arr < 127))       # printable
        v[4] = np.mean((arr >= 48) & (arr < 58))        # digits
        v[5] = np.mean(((arr >= 65) & (arr < 91)) | ((arr >= 97) & (arr < 123)))  # letters
        v[6] = np.mean(arr == 0)                         # nulls
        v[7] = np.mean(arr > 127)                        # high bytes
        v[8] = np.mean((arr == 32) | (arr == 9) | (arr == 10))  # whitespace
        # longest run of a single byte (repetition signature)
        run = best = 1
        for i in range(1, n):
            run = run + 1 if arr[i] == arr[i-1] else 1
            best = max(best, run)
        v[9] = min(best / 64.0, 1.0)
        counts = np.bincount(arr, minlength=256) / n
        v[10] = float(-np.sum(counts[counts > 0] * np.log2(counts[counts > 0]))) / 8.0
    v[11] = min(sum(p.count(f) for f in _FMT) / 4.0, 1.0)   # format specifiers
    v[12] = 1.0 if p[:1] in (b"A", b"B", b"C", b"D") else 0.0  # routes to a handler
    v[13] = 1.0 if n == 0 else 0.0
    v[14] = 1.0 if n in (15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256) else 0.0  # boundary
    v[15] = 1.0 if b"\n" in p else 0.0
    v[16] = 1.0 if (n and len(set(p)) == 1) else 0.0        # single repeated byte
    return v


def input_family(token: str) -> str:
    """Coarse input *type* — the unit of per-family competence/curiosity."""
    _, p = parse_token(token)
    if not p:
        return "empty"
    if any(f in p for f in _FMT):
        return "format"
    if len(p) >= 64:
        return "long"
    if len(set(p)) == 1:
        return "repeat"
    printable = sum(32 <= b < 127 for b in p) / len(p)
    if printable < 0.7:
        return "binary"
    return "text"


@dataclass
class FuzzResult:
    exit_code: int
    stdout: str
    cov_ids: frozenset
    new_coverage: int
    timed_out: bool = False
    duration: float = 0.0

    @property
    def crashed(self) -> bool:
        return self.exit_code >= 128 or self.exit_code in (134, 136, 139)

    @property
    def crash_kind(self) -> str:
        return {139: "segv", 134: "abort", 136: "fpe", 135: "bus"}.get(
            self.exit_code, "signal" if self.crashed else "")


# ── input proposer (coverage-guided corpus, AFL-style but unrestricted) ──────
class InputProposer:
    def __init__(self, mode: str = "stdin", rng=None):
        import random
        self.mode = mode
        self.rng = rng or random.Random(0)
        # corpus of inputs that earned new coverage — the seeds for mutation
        self.corpus: list[bytes] = [b"", b"A", b"BX", b"C%s", b"D8"]

    def observe(self, token: str, result: FuzzResult) -> None:
        if result.new_coverage > 0:
            _, p = parse_token(token)
            if p not in self.corpus and len(self.corpus) < 2000:
                self.corpus.append(p)

    def propose(self, n: int = 16) -> list[str]:
        out = set()
        strat = [self._seeded, self._long, self._format, self._boundary,
                 self._binary, self._mutate, self._route_prefix, self._empty]
        guard = 0
        while len(out) < n and guard < n * 6:
            guard += 1
            try:
                out.add(make_token(self.rng.choice(strat)(), self.mode))
            except Exception:
                pass
        return list(out)[:n]

    def _seeded(self):
        return self.rng.choice(self.corpus)

    def _long(self):
        c = bytes([self.rng.randint(65, 90)])
        return c * self.rng.randint(20, 300)

    def _format(self):
        return self.rng.choice([b"C", b""]) + b"".join(
            self.rng.choice(_FMT) for _ in range(self.rng.randint(1, 8)))

    def _boundary(self):
        pre = self.rng.choice([b"A", b"B", b"C", b"D", b""])
        return pre + b"A" * self.rng.choice([14, 15, 16, 17, 31, 32, 33, 63, 64, 65])

    def _binary(self):
        return bytes(self.rng.randint(0, 255) for _ in range(self.rng.randint(1, 64)))

    def _mutate(self):
        base = bytearray(self.rng.choice(self.corpus) or b"A")
        for _ in range(self.rng.randint(1, 4)):
            if not base:
                base.append(self.rng.randint(0, 255)); continue
            op = self.rng.randint(0, 2)
            i = self.rng.randrange(len(base))
            if op == 0:
                base[i] = self.rng.randint(0, 255)          # flip
            elif op == 1:
                base.insert(i, self.rng.randint(0, 255))     # insert
            else:
                del base[i]                                   # delete
        return bytes(base)

    def _route_prefix(self):
        return self.rng.choice([b"A", b"B", b"C", b"D"]) + bytes(
            self.rng.randint(32, 126) for _ in range(self.rng.randint(0, 40)))

    def _empty(self):
        return b""


# ── the environment ──────────────────────────────────────────────────────────
class BinaryFuzzEnv:
    """Atlas's crash-hunting environment for a single target binary."""

    WORKDIR = "/tmp/atlas_fuzz"

    def __init__(self, vm, source: str | None = None, mode: str = "stdin",
                 timeout: int = 4, log=print):
        self.vm = vm
        self.mode = mode
        self.timeout = timeout
        self.log = log
        self.source = source if source is not None else DEFAULT_TARGET

        self.covered_global: set = set()          # all functions ever reached
        self.crash_inputs: dict[str, bytes] = {}   # crash_kind+path -> input
        self.seen: Counter = Counter()             # behavior signatures (coverage metric)
        self._last = np.zeros(FUZZ_STATE_DIM, dtype=np.float32)
        self.recoveries = 0                        # binaries crash w/o bricking the VM
        self.steps = 0
        self._compile()

    # ── setup ────────────────────────────────────────────────────────────────
    def _compile(self):
        d = self.WORKDIR
        put = (
            f"mkdir -p {d} && "
            f"cat > {d}/target.c <<'ATLAS_EOF'\n{self.source}\nATLAS_EOF\n"
            f"cat > {d}/cov.c <<'ATLAS_EOF'\n{_COV_SHIM}\nATLAS_EOF\n"
            f"gcc -c -no-pie {d}/cov.c -o {d}/cov.o 2>{d}/cc.log && "
            f"gcc -no-pie -fno-pie -finstrument-functions {d}/target.c {d}/cov.o "
            f"-o {d}/target 2>>{d}/cc.log; echo RC:$?"
        )
        r = self.vm.run(put, timeout=60)
        if "RC:0" not in r.stdout:
            log = self.vm.run(f"cat {d}/cc.log").stdout
            raise RuntimeError(f"target compile failed:\n{log}")
        self.log(f"[fuzz] compiled instrumented target in VM ({d}/target)")

    # ── BaseEnvironment-ish API (so SelfTrainer can drive it) ────────────────
    def get_action_dim(self):
        return FUZZ_ACTION_DIM

    def get_observation_dim(self):
        return FUZZ_STATE_DIM

    def reset(self):
        self._last = np.zeros(FUZZ_STATE_DIM, dtype=np.float32)
        return self._last.copy()

    def render(self):
        return None

    # ── run the target with an input, parse coverage + crash ─────────────────
    def _execute(self, token: str, record: bool = True) -> FuzzResult:
        mode, payload = parse_token(token)
        b64 = base64.b64encode(payload).decode()
        d = self.WORKDIR
        run = f"{d}/target" if mode == "stdin" else f'{d}/target "$(cat {d}/in)"'
        script = (
            f"printf %s '{b64}' | base64 -d > {d}/in 2>/dev/null; "
            f"COV_FILE={d}/cov timeout {self.timeout} {run} < {d}/in > {d}/out 2>&1; "
            f"rc=$?; echo \"===RC:$rc\"; echo ===COV; sort -u {d}/cov 2>/dev/null; "
            f"echo ===OUT; head -c 300 {d}/out"
        )
        r = self.vm.run(script, timeout=self.timeout + 5)
        rc, cov_ids, out = self._parse(r.stdout)
        new = 0
        if record:
            fresh = cov_ids - self.covered_global
            new = len(fresh)
            self.covered_global |= cov_ids
        else:
            new = len(cov_ids - self.covered_global)
        return FuzzResult(rc, out, frozenset(cov_ids), new, timed_out=(rc == 124),
                          duration=r.duration)

    @staticmethod
    def _parse(s: str):
        rc, cov, out = 0, set(), ""
        try:
            head, rest = s.split("===RC:", 1)
            rc_str, rest = rest.split("\n", 1)
            rc = int(rc_str.strip())
            cov_block, out = rest.split("===OUT", 1)
            cov_block = cov_block.split("===COV", 1)[-1]
            cov = {ln.strip() for ln in cov_block.splitlines() if ln.strip()}
            out = out.strip()
        except Exception:
            pass
        return rc, cov, out

    def step(self, token: str):
        self.steps += 1
        res = self._execute(token, record=True)
        obs = self.featurize(token, res)
        sig = self._signature(res)
        self.seen[sig] += 1
        reward = float(res.new_coverage) + (3.0 if res.crashed else 0.0)

        if res.crashed:
            _, payload = parse_token(token)
            key = f"{res.crash_kind}:{len(res.cov_ids)}"
            if key not in self.crash_inputs:
                self.crash_inputs[key] = payload
                self.log(f"[fuzz] CRASH ({res.crash_kind}) on {payload[:40]!r} "
                         f"(len={len(payload)}) — {len(self.crash_inputs)} unique so far")

        self._last = obs
        info = {"command": token, "result": res, "family": input_family(token),
                "recovered": False, "coverage": len(self.covered_global),
                "crashed": res.crashed}
        return obs, reward, False, info

    def run_probe(self, token: str):
        return self.featurize(token, self._execute(token, record=False))

    # ── featurization ────────────────────────────────────────────────────────
    def featurize(self, token: str, res: FuzzResult) -> np.ndarray:
        v = np.zeros(FUZZ_STATE_DIM, dtype=np.float32)
        v[0] = max(-1.0, min(1.0, res.exit_code / 128.0))
        v[1] = 1.0 if res.exit_code == 0 else 0.0
        v[2] = 1.0 if res.crashed else 0.0
        v[3] = 1.0 if res.timed_out else 0.0
        v[4] = 1.0 if res.crash_kind == "segv" else 0.0
        v[5] = 1.0 if res.crash_kind == "abort" else 0.0
        v[6] = min(len(res.cov_ids) / 8.0, 1.0)                 # functions reached
        v[7] = min(res.new_coverage / 4.0, 1.0)                 # NEW functions
        v[8] = min(len(res.stdout) / 200.0, 1.0)
        v[9] = 1.0 if res.stdout.strip() else 0.0
        v[10] = min(len(self.covered_global) / 8.0, 1.0)        # global progress
        v[11] = min(res.duration / self.timeout, 1.0)
        v[12] = 1.0 if "%" in res.stdout else 0.0
        v[13] = 1.0                                             # bias
        return v

    def _signature(self, res: FuzzResult):
        return (res.crash_kind or f"rc{res.exit_code}", len(res.cov_ids))

    # ── reporting ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {"functions_covered": len(self.covered_global),
                "unique_crashes": len(self.crash_inputs),
                "crash_inputs": {k: v.hex() for k, v in self.crash_inputs.items()}}

