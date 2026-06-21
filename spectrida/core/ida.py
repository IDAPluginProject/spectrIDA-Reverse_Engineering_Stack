"""idalib-backed IDA operations via a persistent worker subprocess.

The worker opens the .i64 once and answers commands over stdin/stdout, so the
TUI stays snappy (no reopening a 700 MB database on every click). idalib prints
noise to stdout, so every real response is prefixed with ``@@RESP`` and the
client skips everything else.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from spectrida.config import idalib_dir

# Worker: open db, then loop reading {"cmd","args"} lines, reply "@@RESP <json>".
_WORKER = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
import idapro

def emit(obj):
    sys.stdout.write("@@RESP " + json.dumps(obj) + "\n"); sys.stdout.flush()

rc = idapro.open_database(sys.argv[2], False)
if rc != 0:
    emit({"ok": False, "result": f"open_database failed rc={rc}"})
    sys.exit(1)
import idautils, idc, idaapi, ida_funcs

def _norm(a):
    return int(a, 16) if isinstance(a, str) and a.startswith("0x") else int(a)

emit({"ok": True, "result": "ready"})
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line); cmd = req.get("cmd"); a = req.get("args", {})
        if cmd == "quit":
            break
        elif cmd == "list":
            lim = int(a.get("limit", 200000)); out = []
            for ea in idautils.Functions():
                if len(out) >= lim: break
                fn = idaapi.get_func(ea); sz = fn.size() if fn else 0
                out.append({"name": idc.get_func_name(ea), "start": ea, "end": ea + sz, "size": sz})
            emit({"ok": True, "result": out})
        elif cmd == "disasm":
            addr = _norm(a["address"]); fn = idaapi.get_func(addr); out = []
            if fn:
                for ea in idautils.FuncItems(fn.start_ea):
                    out.append({"address": hex(ea), "text": idc.generate_disasm_line(ea, 0)})
            emit({"ok": True, "result": out})
        elif cmd == "decompile":
            try:
                cf = idaapi.decompile(_norm(a["address"])); emit({"ok": True, "result": str(cf) if cf else ""})
            except Exception as e:
                emit({"ok": True, "result": "// decompile error: %s" % e})
        elif cmd == "rename":
            ok = idc.set_name(_norm(a["address"]), a["name"], idc.SN_NOWARN | idc.SN_NOCHECK)
            emit({"ok": True, "result": bool(ok)})
        elif cmd == "save":
            idc.save_database(""); emit({"ok": True, "result": True})
        elif cmd == "xrefs_to":   # callers of this function
            addr = _norm(a["address"]); seen = {};
            for xr in idautils.XrefsTo(addr):
                fn = idaapi.get_func(xr.frm)
                if fn and fn.start_ea not in seen:
                    seen[fn.start_ea] = {"address": hex(fn.start_ea), "name": idc.get_func_name(fn.start_ea)}
            emit({"ok": True, "result": list(seen.values())})
        elif cmd == "xrefs_from":  # callees referenced inside this function
            addr = _norm(a["address"]); fn = idaapi.get_func(addr); seen = {}
            if fn:
                for ea in idautils.FuncItems(fn.start_ea):
                    for xr in idautils.XrefsFrom(ea, 0):
                        tf = idaapi.get_func(xr.to)
                        if tf and tf.start_ea != fn.start_ea and tf.start_ea not in seen:
                            seen[tf.start_ea] = {"address": hex(tf.start_ea), "name": idc.get_func_name(tf.start_ea)}
            emit({"ok": True, "result": list(seen.values())})
        elif cmd == "info":
            addr = _norm(a["address"]); fn = idaapi.get_func(addr)
            if fn:
                emit({"ok": True, "result": {"name": idc.get_func_name(addr), "start": fn.start_ea,
                                              "end": fn.end_ea, "size": fn.end_ea - fn.start_ea}})
            else:
                emit({"ok": True, "result": None})
        elif cmd == "demangle":
            # IDA's own demangler auto-detects the binary's actual ABI (Itanium
            # for GCC/Clang-built ELF/NSO, MSVC-style for Windows PE) — more
            # robust than an external demangler that only knows one scheme.
            names = a.get("names", []); mask = idc.get_inf_attr(idc.INF_SHORT_DN)
            out = {}
            for n in names:
                d = idc.demangle_name(n, mask)
                if d:
                    out[n] = d
            emit({"ok": True, "result": out})
        else:
            emit({"ok": False, "error": "unknown cmd %s" % cmd})
    except Exception as e:
        emit({"ok": False, "error": str(e)})
idapro.close_database(True)
"""


def _idalib_env() -> dict[str, str]:
    env = os.environ.copy()
    ida = idalib_dir()
    if ida:
        p = str(Path(ida).resolve())
        env["PATH"] = p + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = p + os.pathsep + env.get("PYTHONPATH", "")
    return env


class IDAHandle:
    def __init__(self, proc: asyncio.subprocess.Process, i64: str) -> None:
        self._proc = proc
        self.i64 = i64
        self._lock = asyncio.Lock()

    async def _readresp(self) -> dict:
        # skip idapro's stdout noise; only @@RESP lines are ours
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise RuntimeError("idalib worker exited unexpectedly")
            text = line.decode(errors="replace").strip()
            if text.startswith("@@RESP "):
                return json.loads(text[len("@@RESP "):])

    async def call(self, cmd: str, **args):
        async with self._lock:
            self._proc.stdin.write((json.dumps({"cmd": cmd, "args": args}) + "\n").encode())
            await self._proc.stdin.drain()
            resp = await self._readresp()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "idalib error"))
        return resp["result"]

    async def close(self) -> None:
        try:
            self._proc.stdin.write(b'{"cmd":"quit"}\n')
            await self._proc.stdin.drain()
            await asyncio.wait_for(self._proc.wait(), timeout=10)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass


_STREAM_LIMIT = 128 * 1024 * 1024  # 128 MB — list of 150k funcs is ~12 MB as JSON


async def open_ida(i64_path: str) -> IDAHandle:
    ida = idalib_dir()
    if not ida:
        raise RuntimeError("idalib not configured - run: spectrida onboard")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _WORKER, str(Path(ida).resolve()), i64_path,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, env=_idalib_env(),
        limit=_STREAM_LIMIT,
    )
    handle = IDAHandle(proc, i64_path)
    ready = await handle._readresp()   # waits for the "ready" @@RESP
    if not ready.get("ok"):
        raise RuntimeError("idalib worker failed to open the database")
    return handle


# ── thin async API used by the TUI ──────────────────────────────────────────

async def list_functions(ida: IDAHandle, limit: int = 200000) -> list[dict]:
    return await ida.call("list", limit=limit)

async def disasm(ida: IDAHandle, address: str | int) -> list[dict]:
    try:
        return await ida.call("disasm", address=_hex(address))
    except Exception:
        return []

async def decompile(ida: IDAHandle, address: str | int) -> str:
    try:
        return await ida.call("decompile", address=_hex(address))
    except Exception:
        return ""

async def rename(ida: IDAHandle, address: str | int, new_name: str) -> bool:
    try:
        ok = await ida.call("rename", address=_hex(address), name=new_name)
        if ok:
            await ida.call("save")
        return bool(ok)
    except Exception:
        return False

async def xrefs_to(ida: IDAHandle, address: str | int) -> list[dict]:
    try:
        return await ida.call("xrefs_to", address=_hex(address))
    except Exception:
        return []

async def xrefs_from(ida: IDAHandle, address: str | int) -> list[dict]:
    try:
        return await ida.call("xrefs_from", address=_hex(address))
    except Exception:
        return []

async def info(ida: IDAHandle, address: str | int) -> dict | None:
    """Live {name, start, end, size} for a function — used when the graph
    cache only has a placeholder node (no size/pseudocode recorded yet)."""
    try:
        return await ida.call("info", address=_hex(address))
    except Exception:
        return None

async def demangle(ida: IDAHandle, names: list[str]) -> dict[str, str]:
    """Demangle a batch of names via IDA's own demangler. Returns
    {original: demangled} — entries that weren't mangled or failed are omitted."""
    try:
        return await ida.call("demangle", names=names)
    except Exception:
        return {}


def _hex(address: str | int) -> str:
    return hex(address) if isinstance(address, int) else str(address)
