"""A deterministic in-process stand-in for WslVM (no real WSL needed in tests).

Outcomes are a deterministic function of the command, so the command->outcome
mapping is *learnable* — exactly what the world model should be able to fit and
generalize. Health can be toggled to exercise auto-recovery.
"""
import zlib

from atlas.vm.wsl_vm import RunResult


class FakeVM:
    def __init__(self, healthy: bool = True):
        self.calls: list[str] = []
        self._healthy = healthy
        self.rolled_back = 0
        self.brick_on_danger = False

    def run(self, command: str, timeout: int = 8, cwd: str = "~") -> RunResult:
        self.calls.append(command)
        seed = zlib.crc32(command.encode()) & 0xFFFFFFFF
        # deterministic, structured outcome
        exit_code = 0 if seed % 4 else 1
        out = "x" * (seed % 200)
        err = "" if exit_code == 0 else "error: bad"
        if self.brick_on_danger and ("rm " in command or "dd " in command):
            self._healthy = False
        return RunResult(command, out, err, exit_code, 0.01)

    def health_ok(self) -> bool:
        return self._healthy

    def rollback(self, tag: str = "base") -> None:
        self.rolled_back += 1
        self._healthy = True

    def snapshot(self, tag: str = "base"):
        return None

    def has_snapshot(self, tag: str = "base") -> bool:
        return True
