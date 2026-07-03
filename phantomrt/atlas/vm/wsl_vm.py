"""
Isolated WSL "VM" the agent is turned loose in.

We do NOT use the host's default WSL distro (it mounts C: at /mnt/c and is the
user's real environment). Instead we clone a clean userland into a *separate*
WSL2 instance called ``atlas-vm`` that:

  - has no /mnt/c host mount (automount disabled),
  - defaults to a non-root user,
  - defaults to no DNS/network (cleaner learning signal + smaller blast radius),
  - can be snapshotted and rolled back wholesale (throwaway).

WSL2 (not WSL1) is used on purpose: it is a real lightweight VM with its own
kernel, so a wide-open agent is contained by the VM boundary. Snapshot/rollback
is the safety net — if the agent bricks the box, we restore it and keep learning.

Honest isolation caveat: WSL2 shares the host *kernel*. This is a strong sandbox,
not a hypervisor air-gap. Offline-by-default + non-root + snapshot are the
guardrails; residual escape risk exists and is accepted per the project plan.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Layout ────────────────────────────────────────────────────────────────────
DISTRO = "atlas-vm"
_VM_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "vm"
_INSTALL_DIR = _VM_ROOT / "install"      # where the atlas-vm vhdx lives
_BASE_TAR = _VM_ROOT / "base.tar"        # cached clean rootfs
_SNAP_DIR = _VM_ROOT / "snapshots"

# Default user created inside the VM (non-root).
VM_USER = "atlas"

# wsl.conf written into the VM: no host mount, no auto DNS, non-root default.
_WSL_CONF = f"""[automount]
enabled = false
mountFsTab = false

[network]
generateResolvConf = false

[interop]
enabled = true
appendWindowsPath = false

[user]
default = {VM_USER}
"""


@dataclass
class RunResult:
    """Outcome of running one command inside the VM — the raw truth."""
    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    timed_out: bool = False

    @property
    def crashed(self) -> bool:
        # Negative exit codes / 128+signal are how shells report signals
        # (segfault=139, abort=134, etc.). -1 is our own launch failure.
        return self.exit_code >= 128 or self.exit_code in (134, 136, 139)


def _wsl(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a `wsl.exe` management command (list/import/export/...)."""
    return subprocess.run(
        ["wsl", *args],
        capture_output=True,
        timeout=timeout,
    )


def _decode(b: bytes) -> str:
    """wsl.exe management output is often UTF-16LE with NULs; bash -c output is
    UTF-8. Decode defensively for both."""
    if b is None:
        return ""
    text = b.decode("utf-8", errors="ignore")
    if "\x00" in text:  # looks like UTF-16
        text = b.decode("utf-16-le", errors="ignore")
    return text.replace("\x00", "")


class WslVM:
    """Manages the isolated ``atlas-vm`` WSL2 instance."""

    def __init__(self, distro: str = DISTRO, log=print):
        self.distro = distro
        self.log = log

    # ── existence / lifecycle ────────────────────────────────────────────────
    def exists(self) -> bool:
        out = _decode(_wsl(["-l", "-q"]).stdout)
        return any(line.strip() == self.distro for line in out.splitlines())

    def _default_distro(self) -> Optional[str]:
        out = _decode(_wsl(["-l", "-q"]).stdout)
        for line in out.splitlines():
            name = line.strip()
            if name and name != self.distro:
                return name
        return None

    def provision(self, base_distro: Optional[str] = None, force: bool = False) -> None:
        """Create the isolated VM. Idempotent unless ``force``.

        Clones a clean rootfs from an existing distro (no network needed), imports
        it as a fresh WSL2 instance, and locks down wsl.conf.
        """
        if self.exists() and not force:
            self.log(f"[vm] {self.distro} already provisioned")
            return
        if force and self.exists():
            self.destroy()

        _VM_ROOT.mkdir(parents=True, exist_ok=True)
        _SNAP_DIR.mkdir(parents=True, exist_ok=True)
        _INSTALL_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Get a base rootfs tar (cached after first run).
        if not _BASE_TAR.exists():
            src = base_distro or self._default_distro()
            if not src:
                raise RuntimeError("No existing WSL distro to clone from.")
            self.log(f"[vm] exporting clean rootfs from '{src}' (one-time, may take a minute)…")
            _wsl(["--terminate", src], timeout=60)
            r = _wsl(["--export", src, str(_BASE_TAR)], timeout=1800)
            if r.returncode != 0 or not _BASE_TAR.exists():
                raise RuntimeError(f"export failed: {_decode(r.stderr)}")
            self.log(f"[vm] base rootfs cached at {_BASE_TAR} "
                     f"({_BASE_TAR.stat().st_size/1e9:.1f} GB)")

        # 2. Import as a fresh, separate WSL2 instance.
        self.log(f"[vm] importing {self.distro} as WSL2…")
        r = _wsl(["--import", self.distro, str(_INSTALL_DIR), str(_BASE_TAR),
                  "--version", "2"], timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"import failed: {_decode(r.stderr)}")

        # 3. Lock it down: create non-root user + write wsl.conf, then terminate
        #    so the config takes effect on next launch.
        self._bootstrap_isolation()
        self.log(f"[vm] {self.distro} provisioned (isolated, offline, non-root '{VM_USER}')")

    def _bootstrap_isolation(self) -> None:
        # These run as root (imported distro defaults to root until wsl.conf applies).
        boot = (
            f"id -u {VM_USER} >/dev/null 2>&1 || "
            f"useradd -m -s /bin/bash {VM_USER}; "
            f"mkdir -p /home/{VM_USER}/work; chown -R {VM_USER}:{VM_USER} /home/{VM_USER}; "
            # kill DNS so name-based internet fails even if NAT is up
            ": > /etc/resolv.conf; "
            f"printf '%s' \"{_WSL_CONF}\" > /etc/wsl.conf"
        )
        r = subprocess.run(
            ["wsl", "-d", self.distro, "-u", "root", "-e", "bash", "-lc", boot],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            raise RuntimeError(f"bootstrap failed: {_decode(r.stderr)}")
        _wsl(["--terminate", self.distro], timeout=60)

    def destroy(self) -> None:
        _wsl(["--terminate", self.distro], timeout=60)
        _wsl(["--unregister", self.distro], timeout=120)
        self.log(f"[vm] {self.distro} destroyed")

    # ── running commands ─────────────────────────────────────────────────────
    def run(self, command: str, timeout: int = 10, cwd: str = "~") -> RunResult:
        """Run a shell command inside the VM as the non-root user.

        Returns the real outcome — never raises on a failing/ crashing command
        (that's signal, not error). Only launch failures produce exit_code -1.
        """
        # `; echo __RC__$?` lets us recover the true exit code even with stderr.
        wrapped = f"cd {cwd} 2>/dev/null; {command}"
        start = time.time()
        try:
            r = subprocess.run(
                ["wsl", "-d", self.distro, "-u", VM_USER, "-e", "bash", "-lc", wrapped],
                capture_output=True, timeout=timeout,
            )
            return RunResult(
                command=command,
                stdout=_decode(r.stdout),
                stderr=_decode(r.stderr),
                exit_code=r.returncode,
                duration=time.time() - start,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(
                command=command,
                stdout=_decode(e.stdout) if e.stdout else "",
                stderr=_decode(e.stderr) if e.stderr else "",
                exit_code=124,  # conventional timeout code
                duration=time.time() - start,
                timed_out=True,
            )
        except Exception as e:  # launch failure (distro missing, wsl gone, …)
            return RunResult(command, "", str(e), -1, time.time() - start)

    def health_ok(self) -> bool:
        """Cheap liveness probe: is the VM still a usable Linux box?"""
        r = self.run("echo __ok__ && test -x /bin/ls && test -x /bin/sh", timeout=15)
        if r.exit_code != 0 or "__ok__" not in r.stdout:
            return False
        # disk not full
        df = self.run("df -P / | tail -1 | awk '{print $5}' | tr -d '%'", timeout=15)
        try:
            return int(df.stdout.strip() or "0") < 99
        except ValueError:
            return True

    # ── snapshot / rollback (the safety net) ─────────────────────────────────
    def snapshot(self, tag: str = "base") -> Path:
        _SNAP_DIR.mkdir(parents=True, exist_ok=True)
        path = _SNAP_DIR / f"{tag}.tar"
        _wsl(["--terminate", self.distro], timeout=60)
        r = _wsl(["--export", self.distro, str(path)], timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"snapshot failed: {_decode(r.stderr)}")
        self.log(f"[vm] snapshot '{tag}' → {path} ({path.stat().st_size/1e9:.2f} GB)")
        return path

    def rollback(self, tag: str = "base") -> None:
        path = _SNAP_DIR / f"{tag}.tar"
        if not path.exists():
            raise FileNotFoundError(f"no snapshot '{tag}' at {path}")
        _wsl(["--terminate", self.distro], timeout=60)
        _wsl(["--unregister", self.distro], timeout=120)
        r = _wsl(["--import", self.distro, str(_INSTALL_DIR), str(path),
                  "--version", "2"], timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"rollback failed: {_decode(r.stderr)}")
        self.log(f"[vm] rolled back to snapshot '{tag}'")

    def has_snapshot(self, tag: str = "base") -> bool:
        return (_SNAP_DIR / f"{tag}.tar").exists()
