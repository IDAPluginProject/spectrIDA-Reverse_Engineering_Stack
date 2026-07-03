"""
The action space: proposing shell commands and embedding them.

Two jobs:

1. ``embed_command`` — turn a command string into a fixed, *structured* vector.
   The embedding is deliberately compositional: it encodes the command's family,
   argument structure, operators, and intent — NOT a memorized id. This is what
   lets the world model generalize (predict `grep -c` from having learned `grep`
   and `wc -c` separately) instead of memorizing exact strings.

2. ``CommandProposer`` — generate candidate commands for the agent to choose
   among. Unrestricted: bare binaries, discovered files, mutations/compositions
   of past commands, file creation, compiling & running code, and yes destructive
   ops too. The VM boundary + snapshot rollback are the containment, not a filter.
   A self-expanding library of *working* commands becomes new building blocks —
   the action space itself grows as the agent discovers what the machine offers.
"""

from __future__ import annotations

import random
import re
import shlex
import numpy as np


ACTION_DIM = 32

# ── binary → (family, intent) knowledge, used only to *structure* the embedding.
# Unknown binaries fall back to "misc"; the agent still runs and learns them.
_FAMILY = {
    "text":    ["grep", "sed", "awk", "cut", "sort", "uniq", "tr", "wc", "head",
                "tail", "cat", "tac", "rev", "fold", "paste", "join", "column", "nl"],
    "file":    ["ls", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "stat", "find",
                "ln", "readlink", "basename", "dirname", "du", "df", "file", "tree"],
    "archive": ["tar", "gzip", "gunzip", "zip", "unzip", "xz", "bzip2", "cpio"],
    "compile": ["gcc", "g++", "cc", "make", "ld", "as", "ar", "objdump", "nm", "strip"],
    "interp":  ["python3", "python", "perl", "bash", "sh", "node", "ruby", "lua", "awk"],
    "proc":    ["ps", "top", "kill", "pkill", "nice", "nohup", "jobs", "sleep",
                "timeout", "watch", "pgrep", "pidof"],
    "perm":    ["chmod", "chown", "chgrp", "umask", "id", "whoami", "groups", "sudo"],
    "system":  ["uname", "hostname", "uptime", "date", "env", "printenv", "free",
                "lscpu", "mount", "dmesg", "sysctl", "ulimit"],
    "net":     ["ip", "ss", "ping", "curl", "wget", "netstat", "nc", "host", "dig"],
    "shell":   ["echo", "printf", "test", "true", "false", "seq", "yes", "xargs",
                "tee", "read", "expr", "let", "type", "which", "command"],
}
_BIN2FAM = {b: fam for fam, bins in _FAMILY.items() for b in bins}
_FAMILIES = list(_FAMILY.keys()) + ["misc"]

# rough intent per family (reads / writes / creates / deletes / executes / info)
_INTENT = {
    "text":   (1, 0, 0, 0, 0, 1), "file": (1, 1, 1, 1, 0, 1),
    "archive":(1, 1, 1, 0, 0, 0), "compile":(1, 1, 1, 0, 1, 0),
    "interp": (1, 1, 1, 0, 1, 0), "proc": (1, 0, 0, 1, 1, 1),
    "perm":   (1, 1, 0, 0, 0, 1), "system":(1, 0, 0, 0, 0, 1),
    "net":    (1, 1, 0, 0, 1, 1), "shell": (0, 1, 1, 0, 0, 1),
    "misc":   (1, 0, 0, 0, 1, 0),
}
# commands that tend to break the box — tracked as a feature, NOT blocked.
_DANGER = ("rm ", "rm -", "dd ", "mkfs", ":(){", "shutdown", "reboot", "> /dev",
           "chmod -R", "chown -R", "mv /", "> /etc", "kill -9 -1")


def primary_binary(command: str) -> str:
    """First real program token in a command (skips `sudo`, env assignments)."""
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    for t in toks:
        if "=" in t and not t.startswith("-"):   # VAR=val prefix
            continue
        if t in ("sudo", "nohup", "time", "nice", "timeout", "env"):
            continue
        return t.split("/")[-1]
    return toks[0].split("/")[-1] if toks else ""


def embed_command(command: str) -> np.ndarray:
    """Structured, compositional embedding of a command → [ACTION_DIM] in ~[0,1]."""
    v = np.zeros(ACTION_DIM, dtype=np.float32)
    cmd = command.strip()
    binn = primary_binary(cmd)
    fam = _BIN2FAM.get(binn, "misc")

    # [0:11] family one-hot
    v[_FAMILIES.index(fam)] = 1.0
    # [11:17] intent bits for the family
    v[11:17] = np.array(_INTENT[fam], dtype=np.float32)

    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    args = toks[1:] if toks else []

    # [17:32] structural features (all bounded)
    v[17] = min(len(toks) / 8.0, 1.0)                                  # n tokens
    v[18] = min(len(cmd) / 80.0, 1.0)                                  # length
    v[19] = min(sum(a.startswith("-") for a in args) / 4.0, 1.0)       # flags
    v[20] = 1.0 if "|" in cmd else 0.0                                 # pipe
    v[21] = 1.0 if (">" in cmd or ">>" in cmd) else 0.0                # redirect out
    v[22] = 1.0 if "<" in cmd else 0.0                                 # redirect in
    v[23] = 1.0 if any(c in cmd for c in "*?[") else 0.0               # glob
    v[24] = min(sum("/" in a for a in args) / 3.0, 1.0)               # path args
    v[25] = min(sum(bool(re.fullmatch(r"-?\d+", a)) for a in args) / 3.0, 1.0)  # numeric
    v[26] = 1.0 if ("$(" in cmd or "`" in cmd) else 0.0               # subshell
    v[27] = 1.0 if ("&&" in cmd or ";" in cmd or "||" in cmd) else 0.0 # chained
    v[28] = 1.0 if cmd.rstrip().endswith("&") else 0.0                # background
    v[29] = 1.0 if any(d in cmd for d in _DANGER) else 0.0            # destructive
    v[30] = 1.0 if binn in _BIN2FAM else 0.0                          # known binary
    v[31] = 1.0 if ("--help" in cmd or " -h" in cmd or binn == "man") else 0.0  # self-doc
    return v


def command_family(command: str) -> str:
    """The behavior family a command belongs to (for per-family competence)."""
    return _BIN2FAM.get(primary_binary(command), "misc")


class CommandProposer:
    """Generates candidate commands. Unrestricted and self-expanding."""

    def __init__(self, vm, rng: random.Random | None = None):
        self.vm = vm
        self.rng = rng or random.Random(0)
        self.binaries: list[str] = []
        self.files: set[str] = {"~", "/tmp", "/etc/hostname", "/proc/cpuinfo"}
        # library of commands observed to *work* — grows into new building blocks
        self.library: list[str] = [
            "echo hello", "ls -la", "pwd", "whoami", "uname -a",
            "cat /etc/os-release", "ls /usr/bin | head", "seq 1 5",
        ]
        self._discovered = False

    # ── discovery ────────────────────────────────────────────────────────────
    def discover(self) -> None:
        """One-time enumeration of what the machine offers (PATH binaries)."""
        r = self.vm.run("ls /usr/bin /bin 2>/dev/null | sort -u", timeout=20)
        bins = [b.strip() for b in r.stdout.splitlines() if b.strip() and "/" not in b]
        self.binaries = bins or list(_BIN2FAM.keys())
        self._discovered = True

    def observe(self, command: str, result) -> None:
        """Learn from a run: keep working commands, harvest discovered paths."""
        if result.exit_code == 0 and command not in self.library:
            if len(self.library) < 2000:
                self.library.append(command)
        # harvest path-looking tokens from output the agent just saw
        for tok in re.findall(r"/[\w./-]+", result.stdout[:4000]):
            if len(self.files) < 5000:
                self.files.add(tok)

    # ── proposing ────────────────────────────────────────────────────────────
    def propose(self, n: int = 16) -> list[str]:
        if not self._discovered:
            self.discover()
        cands: set[str] = set()
        strategies = [
            self._bare_binary, self._binary_help, self._on_file, self._mutate,
            self._compose_pipe, self._explore_fs, self._create_and_run, self._write_code,
        ]
        guard = 0
        while len(cands) < n and guard < n * 6:
            guard += 1
            try:
                c = self.rng.choice(strategies)()
            except Exception:
                c = None
            if c:
                cands.add(c.strip())
        return list(cands)[:n]

    def _rand_bin(self) -> str:
        return self.rng.choice(self.binaries) if self.binaries else "echo"

    def _rand_file(self) -> str:
        return self.rng.choice(sorted(self.files))

    def _bare_binary(self) -> str:
        return self._rand_bin()

    def _binary_help(self) -> str:
        return f"{self._rand_bin()} --help 2>&1 | head -5"

    def _on_file(self) -> str:
        b = self.rng.choice(["cat", "ls -la", "stat", "wc -l", "head", "file", "du -h"])
        return f"{b} {self._rand_file()}"

    def _mutate(self) -> str:
        base = self.rng.choice(self.library)
        pipe = self.rng.choice(["| head -3", "| wc -l", "| sort", "| grep -c .", "| tr a-z A-Z"])
        return f"{base} 2>&1 {pipe}"

    def _compose_pipe(self) -> str:
        a, b = self.rng.choice(self.library), self.rng.choice(
            ["wc -c", "grep .", "sort", "uniq -c", "head -4", "tac", "rev"])
        return f"{a} 2>/dev/null | {b}"

    def _explore_fs(self) -> str:
        return self.rng.choice([
            f"ls -la {self._rand_file()}", "find / -maxdepth 2 -type d 2>/dev/null | head",
            "ls /proc | head", f"cat {self._rand_file()} 2>&1 | head -3",
            "df -h", "free -m", "ps aux | head", "env | head",
        ])

    def _create_and_run(self) -> str:
        n = self.rng.randint(1, 999)
        return self.rng.choice([
            f"echo data{n} > /tmp/f{n}.txt && cat /tmp/f{n}.txt",
            f"mkdir -p /tmp/d{n} && ls -la /tmp/d{n}",
            f"seq 1 {n % 20 + 1} | sort -r",
            f"printf 'a\\nb\\nc\\n' | grep b",
        ])

    def _write_code(self) -> str:
        """Write and execute real code — learn compiling/running mechanics."""
        n = self.rng.randint(1, 999)
        return self.rng.choice([
            f"python3 -c 'print(sum(range({n % 50 + 1})))'",
            f"echo 'int main(){{return {n % 5};}}' > /tmp/p{n}.c "
            f"&& gcc /tmp/p{n}.c -o /tmp/p{n} 2>&1 && /tmp/p{n}; echo rc=$?",
            f"python3 -c 'import os; print(os.listdir(\"/tmp\")[:5])'",
            f"echo 'print(2**{n % 16})' | python3",
        ])
