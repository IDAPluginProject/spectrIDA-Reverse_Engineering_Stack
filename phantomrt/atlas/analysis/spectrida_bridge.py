"""
The connector: spectrIDA's FormatHandler -> Atlas's emulation harness.

Reuses spectrIDA's own format plugins (`spectrida.analysis.formats`) to load ANY
supported binary (PE/Windows, NSO/Switch, ELF, .so), decompress if needed, and
hand back the real section bytes + arch. Atlas then maps the whole image (so
internal calls resolve = "chain emulation") and fuzzes a chosen function.

This is the honest edge: real bytes, real arch, from the same plugin spectrIDA
already trusts — no re-implementing loaders, no guessing addresses.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .unicorn_harness import EmulationHarness, EmuResult

DEFAULT_SPECTRIDA = r"C:\Users\Administrator\Desktop\scrape\mini-mythos\spectrIDA"

_PE_MACHINE = {0x8664: "x86_64", 0xAA64: "arm64", 0x14C: "x86", 0x1C0: "arm"}
_ELF_MACHINE = {0x3E: "x86_64", 0xB7: "arm64", 0x28: "arm", 0x03: "x86"}


def _detect_arch(path: str) -> str:
    """Read the true CPU from the PE/ELF header (the handler may hand back None)."""
    import struct
    with open(path, "rb") as f:
        head = f.read(0x40)
        if head[:2] == b"MZ":                          # PE
            e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
            f.seek(e_lfanew + 4)                        # skip "PE\0\0" signature
            machine = struct.unpack("<H", f.read(2))[0]
            return _PE_MACHINE.get(machine, "x86_64")
        if head[:4] == b"\x7fELF":                      # ELF
            return _ELF_MACHINE.get(struct.unpack_from("<H", head, 18)[0], "x86_64")
    return "x86_64"


class EmulatedBinary:
    """A binary loaded via spectrIDA's FormatHandler, ready to emulate functions."""

    def __init__(self, path: str, spectrida_path: str = DEFAULT_SPECTRIDA):
        if spectrida_path and spectrida_path not in sys.path:
            sys.path.insert(0, spectrida_path)
        from spectrida.analysis.formats.registry import detect

        self.path = path
        self.handler = detect(path)
        self.image = self.handler.prepare(path, tempfile.mkdtemp())
        # The handler's arch hint can be None (PE/ELF: it lets IDA decide). Never
        # guess — read the real machine type from the header, else emulation runs
        # the wrong CPU (e.g. x86-64 droid.exe mis-run as arm64 = garbage).
        self.arch = self.image.arch or _detect_arch(path)
        self._regions = None

    @property
    def format(self) -> str:
        return self.handler.name

    def regions(self):
        """Whole-image sections as (va, bytes), cached — real bytes so internal
        calls land on real code; .bss/short sections zero-padded to full size."""
        if self._regions is None:
            regs = []
            for s in self.image.sections:
                va = self.image.image_base + s.va
                data = self.handler.read_bytes(self.image, va, va + s.vsize)
                if len(data) < s.vsize:
                    data = data + b"\x00" * (s.vsize - len(data))
                regs.append((va, data))
            self._regions = regs
        return self._regions

    def emulate(self, entry_addr: int, input_bytes: bytes = b"",
                max_insns: int = 20000) -> EmuResult:
        """Emulate the function at ``entry_addr`` with the whole image mapped and
        out-of-chain calls / syscalls stubbed. arg0 points at the fuzzable input
        buffer (so a C++ ``this`` reads fuzzed object memory)."""
        h = EmulationHarness(self.arch, max_insns=max_insns)
        return h.run(regions=self.regions(), entry=entry_addr,
                     input_bytes=input_bytes, stub_calls=True)
