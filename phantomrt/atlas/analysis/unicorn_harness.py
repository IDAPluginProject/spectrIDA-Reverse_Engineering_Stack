"""
The hard edge: exercise ONE function by CPU emulation — no OS, any architecture.

spectrIDA's targets are ARM64 (Switch NSO) and Android .so — they can't run in
the WSL VM. So instead of *running the program*, we emulate the function's raw
machine code with Unicorn: map its bytes, set up a stack + an input buffer, point
the first argument at the input, run for a bounded number of instructions, and
catch the faults. A function that dereferences an input-derived bad pointer, walks
off a buffer, etc. shows up as an unmapped-memory fault — a real crash signal,
found without a Switch in sight.

Arch-agnostic: pass ``arch`` ("arm64", "x86_64", "arm", "x86") — which is exactly
what spectrIDA's FormatHandler hands us via ``PreparedImage.arch``. So the same
harness fuzzes a Windows PE function (x86-64) and an Odyssey NSO function (arm64).

Honest limits: this emulates a function *in a vacuum*. Pure-computation functions
(parsers, crypto, validation, string/math) work great. Functions that call into
the OS / other libs will fetch-fault at the call unless those are stubbed — that's
standard harness work (or graduate to Qiling for full OS emulation).
"""

from __future__ import annotations

from dataclasses import dataclass

from unicorn import (
    Uc, UcError,
    UC_ARCH_ARM64, UC_ARCH_ARM, UC_ARCH_X86,
    UC_MODE_ARM, UC_MODE_32, UC_MODE_64, UC_MODE_LITTLE_ENDIAN,
    UC_HOOK_BLOCK, UC_HOOK_MEM_UNMAPPED, UC_HOOK_INSN_INVALID, UC_HOOK_INTR,
    UC_PROT_ALL,
    UC_MEM_READ_UNMAPPED, UC_MEM_WRITE_UNMAPPED, UC_MEM_FETCH_UNMAPPED,
)

_PAGE = 0x1000
# a single "return" instruction per arch — used to stub out-of-chain calls
_RET = {"arm64": b"\xc0\x03\x5f\xd6", "x86_64": b"\xc3", "arm": b"\x1e\xff\x2f\xe1"}


def _align_down(x): return x & ~(_PAGE - 1)
def _align_up(x): return (x + _PAGE - 1) & ~(_PAGE - 1)

# memory layout (well-separated so a wild pointer lands in unmapped space = fault)
CODE_BASE = 0x0100_0000
CODE_SIZE = 0x0010_0000
STACK_BASE = 0x0200_0000
STACK_SIZE = 0x0010_0000
INPUT_BASE = 0x0300_0000
INPUT_SIZE = 0x0010_0000
RET_MAGIC = 0x0AAA_0000       # unmapped sentinel: function "returns" here → clean stop

_FAULT = {
    UC_MEM_READ_UNMAPPED: "read_unmapped",
    UC_MEM_WRITE_UNMAPPED: "write_unmapped",
    UC_MEM_FETCH_UNMAPPED: "fetch_unmapped",
}


@dataclass
class EmuResult:
    crashed: bool
    crash_kind: str          # read_unmapped / write_unmapped / fetch_unmapped / invalid_insn / ""
    fault_addr: int
    cov_ids: frozenset       # basic-block addresses hit (as hex strings)
    blocks: int
    returned: bool           # reached the return sentinel cleanly
    timed_out: bool          # hit instruction budget without returning
    new_coverage: int = 0    # blocks not seen before — the env fills this in
    stubbed_calls: int = 0   # out-of-chain calls stubbed (return 0) instead of faulting
    syscalls: int = 0        # svc/syscall instructions skipped

    @property
    def status(self) -> str:
        if self.crashed:
            # a fault on a near-null address is almost always an uninitialized
            # global/`this` pointer (missing live state), NOT a real bug — say so
            # honestly instead of crying crash.
            return "needs_state" if self.fault_addr < 0x10000 else "crash"
        if self.returned:
            return "clean"
        return "inconclusive"    # ran out of budget / couldn't complete faithfully

    @property
    def note(self) -> str:
        return {
            "needs_state": f"null-ish deref @+{hex(self.fault_addr)} — uninitialized "
                           f"global/this; needs live engine state (LLM reasons statically)",
            "crash": f"{self.crash_kind} @ {hex(self.fault_addr)} on a wild address — "
                     f"candidate bug (verify: is that field input-controlled?)",
            "clean": f"returned cleanly; {self.stubbed_calls} calls stubbed",
            "inconclusive": "hit instruction budget without returning",
        }[self.status]

    # duck-typed shims so the fuzz/annotate reuse works unchanged
    @property
    def exit_code(self) -> int:
        return 139 if self.crashed else (124 if self.timed_out else 0)

    @property
    def stdout(self) -> str:
        return ""


class EmulationHarness:
    """Emulates a single function's machine code with fuzzed inputs."""

    def __init__(self, arch: str = "arm64", max_insns: int = 20000):
        self.arch = arch
        self.max_insns = max_insns
        self._cfg = self._arch_config(arch)
        self.arch_key = self._cfg["key"]        # normalized: arm64 / x86_64 / arm

    @staticmethod
    def _arch_config(arch: str) -> dict:
        a = arch.lower().replace("-", "").replace("_", "")
        if a in ("arm64", "aarch64"):
            from unicorn import arm64_const as C
            return {"key": "arm64", "uc": (UC_ARCH_ARM64, UC_MODE_ARM), "sp": C.UC_ARM64_REG_SP,
                    "pc": C.UC_ARM64_REG_PC, "lr": C.UC_ARM64_REG_LR,
                    "args": [C.UC_ARM64_REG_X0, C.UC_ARM64_REG_X1, C.UC_ARM64_REG_X2,
                             C.UC_ARM64_REG_X3], "push_ret": False}
        if a in ("x8664", "x64", "amd64"):
            from unicorn import x86_const as C
            return {"key": "x86_64", "uc": (UC_ARCH_X86, UC_MODE_64), "sp": C.UC_X86_REG_RSP,
                    "pc": C.UC_X86_REG_RIP, "lr": None,
                    "args": [C.UC_X86_REG_RDI, C.UC_X86_REG_RSI, C.UC_X86_REG_RDX,
                             C.UC_X86_REG_RCX], "push_ret": True}
        if a in ("arm", "arm32", "thumb"):
            from unicorn import arm_const as C
            return {"key": "arm", "uc": (UC_ARCH_ARM, UC_MODE_ARM), "sp": C.UC_ARM_REG_SP,
                    "pc": C.UC_ARM_REG_PC, "lr": C.UC_ARM_REG_LR,
                    "args": [C.UC_ARM_REG_R0, C.UC_ARM_REG_R1, C.UC_ARM_REG_R2,
                             C.UC_ARM_REG_R3], "push_ret": False}
        raise ValueError(f"unsupported arch: {arch}")

    def run(self, code: bytes | None = None, input_bytes: bytes = b"", *,
            regions=None, entry: int | None = None, stub_calls: bool = False,
            extra_args=()) -> EmuResult:
        """Emulate a function with fuzzed input.

        Single-function mode (default): pass ``code`` — it's mapped at CODE_BASE.
        Chain mode: pass ``regions`` = [(va, bytes), ...] (whole sections, so
        internal calls land on real code) and ``entry`` (the function's VA).
        ``stub_calls=True`` turns out-of-chain calls into RET-stubs (return 0) and
        skips syscalls, so only real data faults count as crashes.
        """
        cfg = self._cfg
        uc = Uc(*cfg["uc"])
        uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_ALL)
        uc.mem_map(INPUT_BASE, INPUT_SIZE, UC_PROT_ALL)

        if regions is None:                          # single-function mode
            uc.mem_map(CODE_BASE, CODE_SIZE, UC_PROT_ALL)
            uc.mem_write(CODE_BASE, code or b"")
            start = CODE_BASE
        else:                                        # chain mode: map sections
            # merge page ranges first — adjacent sections can round into the same
            # page and mapping them separately raises UC_ERR_MAP.
            ranges = sorted((_align_down(va), _align_up(va + len(d))) for va, d in regions)
            merged: list[list[int]] = []
            for st, en in ranges:
                if merged and st <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], en)
                else:
                    merged.append([st, en])
            for st, en in merged:
                uc.mem_map(st, en - st, UC_PROT_ALL)
            for va, data in regions:
                uc.mem_write(va, data)
            start = entry
        if input_bytes:
            uc.mem_write(INPUT_BASE, input_bytes[:INPUT_SIZE])

        sp = STACK_BASE + STACK_SIZE // 2
        argvals = [INPUT_BASE, len(input_bytes), *extra_args]   # arg0=input ptr, arg1=len
        for reg, val in zip(cfg["args"], argvals):
            uc.reg_write(reg, val & 0xFFFFFFFFFFFFFFFF)
        if cfg["push_ret"]:
            sp -= 8
            uc.mem_write(sp, RET_MAGIC.to_bytes(8, "little"))
        else:
            uc.reg_write(cfg["lr"], RET_MAGIC)
        uc.reg_write(cfg["sp"], sp)

        cov: set[int] = set()
        fault = {"kind": "", "addr": 0}
        stats = {"stubbed": 0, "syscalls": 0}
        ret_bytes = _RET.get(self.arch_key, b"\xc3")

        uc.hook_add(UC_HOOK_BLOCK, lambda u, a, s, d: cov.add(a))

        def on_bad_mem(u, access, address, size, value, data):
            # a call/branch to unmapped code = an out-of-chain call → stub it
            if stub_calls and access == UC_MEM_FETCH_UNMAPPED:
                page = _align_down(address)
                try:
                    u.mem_map(page, _PAGE, UC_PROT_ALL)
                    u.mem_write(page, ret_bytes * (_PAGE // len(ret_bytes)))
                except UcError:
                    pass
                u.reg_write(cfg["args"][0], 0)      # stubbed call returns 0
                stats["stubbed"] += 1
                return True                          # resume → executes RET → returns
            fault["kind"] = _FAULT.get(access, "mem_unmapped")
            fault["addr"] = address
            return False                             # real data fault → crash
        uc.hook_add(UC_HOOK_MEM_UNMAPPED, on_bad_mem)

        def on_bad_insn(u, data):
            fault["kind"] = "invalid_insn"
            fault["addr"] = u.reg_read(cfg["pc"])
            return False
        uc.hook_add(UC_HOOK_INSN_INVALID, on_bad_insn)

        if stub_calls:                               # skip syscalls (svc/int)
            def on_intr(u, intno, data):
                stats["syscalls"] += 1
                u.reg_write(cfg["args"][0], 0)
            uc.hook_add(UC_HOOK_INTR, on_intr)

        crashed = False
        try:
            uc.emu_start(start, RET_MAGIC, timeout=0, count=self.max_insns)
        except UcError:
            crashed = True
            if not fault["kind"]:
                fault["kind"] = "cpu_fault"

        pc = uc.reg_read(cfg["pc"])
        returned = (not crashed) and (pc == RET_MAGIC)
        return EmuResult(
            crashed=crashed, crash_kind=fault["kind"], fault_addr=fault["addr"],
            cov_ids=frozenset(hex(a) for a in cov), blocks=len(cov),
            returned=returned, timed_out=(not crashed and not returned),
            stubbed_calls=stats["stubbed"], syscalls=stats["syscalls"],
        )


# ── Atlas environment: fuzz an emulated function with the curiosity loop ──────
EMU_STATE_DIM = 14


class EmulatedFuzzEnv:
    """Presents a single emulated function as an Atlas env, so SelfTrainer's
    curiosity loop learns which INPUTS crash it. Same interface as
    BinaryFuzzEnv, so the trainer/proposer/annotator all reuse unchanged."""

    def __init__(self, code: bytes, arch: str = "arm64", max_insns: int = 20000,
                 log=print, binary: str = "emulated", addr: int = 0):
        import numpy as np
        from collections import Counter
        self._np = np
        self.harness = EmulationHarness(arch, max_insns=max_insns)
        self.code = code
        self.arch = arch
        self.log = log
        self.binary = binary
        self.addr = addr
        self.covered_global: set = set()
        self.crash_inputs: dict[str, bytes] = {}
        self.seen = Counter()
        self._last = np.zeros(EMU_STATE_DIM, dtype=np.float32)
        self.recoveries = 0            # emulation is sandboxed; nothing to recover
        self.steps = 0

    def get_action_dim(self):
        from .binary_fuzz import FUZZ_ACTION_DIM
        return FUZZ_ACTION_DIM

    def get_observation_dim(self):
        return EMU_STATE_DIM

    def reset(self):
        self._last = self._np.zeros(EMU_STATE_DIM, dtype=self._np.float32)
        return self._last.copy()

    def render(self):
        return None

    def _run(self, token: str, record: bool):
        from .binary_fuzz import parse_token
        _, payload = parse_token(token)
        res = self.harness.run(self.code, payload)
        fresh = res.cov_ids - self.covered_global
        res.new_coverage = len(fresh)
        if record:
            self.covered_global |= res.cov_ids
        return payload, res

    def step(self, token: str):
        from .binary_fuzz import input_family
        self.steps += 1
        payload, res = self._run(token, record=True)
        obs = self.featurize(token, res)
        sig = (res.crash_kind or ("timeout" if res.timed_out else "ok"),
               min(res.blocks, 16))
        self.seen[sig] += 1
        reward = float(res.new_coverage) + (3.0 if res.crashed else 0.0)
        if res.crashed:
            key = f"{res.crash_kind}@{hex(res.fault_addr)}"
            if key not in self.crash_inputs:
                self.crash_inputs[key] = payload
                self.log(f"[emu] CRASH {res.crash_kind} @ {hex(res.fault_addr)} "
                         f"on {payload[:24]!r} — {len(self.crash_inputs)} unique")
        self._last = obs
        info = {"command": token, "result": res, "family": input_family(token),
                "recovered": False, "coverage": len(self.covered_global),
                "crashed": res.crashed}
        return obs, reward, False, info

    def run_probe(self, token: str):
        _, res = self._run(token, record=False)
        return self.featurize(token, res)

    def featurize(self, token: str, res: EmuResult):
        np = self._np
        v = np.zeros(EMU_STATE_DIM, dtype=np.float32)
        v[0] = 1.0 if res.crashed else 0.0
        v[1] = 1.0 if res.returned else 0.0
        v[2] = 1.0 if res.timed_out else 0.0
        v[3] = 1.0 if res.crash_kind == "read_unmapped" else 0.0
        v[4] = 1.0 if res.crash_kind == "write_unmapped" else 0.0
        v[5] = 1.0 if res.crash_kind == "fetch_unmapped" else 0.0
        v[6] = 1.0 if res.crash_kind == "invalid_insn" else 0.0
        v[7] = min(res.blocks / 16.0, 1.0)
        v[8] = min(res.new_coverage / 4.0, 1.0)
        v[9] = min(len(self.covered_global) / 32.0, 1.0)
        v[10] = 1.0 if res.fault_addr else 0.0
        v[11] = 1.0 if res.crash_kind.startswith("write") else 0.0
        v[13] = 1.0
        return v

    def summary(self) -> dict:
        return {"functions_covered": len(self.covered_global),
                "unique_crashes": len(self.crash_inputs),
                "crash_inputs": {k: v.hex() for k, v in self.crash_inputs.items()}}

