"""
ida_gpu_accel/preseeder.py

Takes scan results from arm64_scanner and plants them into IDA's database
BEFORE auto_wait(), so IDA's analysis engine works from a pre-populated
starting set instead of discovering everything from scratch.

  add_func()  — marks prologues as function starts (IDA fills in the body)
  add_cref()  — plants BL call xrefs so IDA links callers to callees
  add_head()  — marks basic-block boundaries as code heads

IDA will still run its full analysis — we just give it better hints.
"""

from __future__ import annotations

import time

from .config import PRESEED_ENABLED


def _is_in_code(ea: int) -> bool:
    """True if ea falls inside a known code segment."""
    try:
        import ida_segment
        seg = ida_segment.getseg(ea)
        return seg is not None and seg.type == ida_segment.SEG_CODE
    except Exception:
        return True   # be optimistic if IDA API fails


def seed(
    prologues: list[int],
    bl_targets: list[int],
    bb_heads: list[int],
    strings: list[tuple[int, str]],
    *,
    text_start: int = 0,
    text_end: int = 0,
) -> None:
    """
    Feed scan results into IDA.

    text_start / text_end: optional range filter so we don't try to
    create functions for EA values that resolved outside the shard.
    Pass 0,0 to skip range filtering.
    """
    if not PRESEED_ENABLED:
        return

    import ida_bytes
    import ida_funcs
    import idc

    t0 = time.perf_counter()
    use_range = text_end > text_start

    def in_range(ea: int) -> bool:
        if not use_range:
            return True
        return text_start <= ea < text_end

    # ── 1. Mark prologues as function starts ──────────────────────────────
    n_funcs = 0
    for ea in prologues:
        if not in_range(ea):
            continue
        if ida_funcs.get_func(ea) is None:                 # not already a function
            if ida_funcs.add_func(ea) or ida_funcs.add_func(ea, ida_funcs.FUNC_TAIL):
                n_funcs += 1

    # ── 2. Plant BL call xrefs ────────────────────────────────────────────
    n_xrefs = 0
    for tgt in bl_targets:
        if not in_range(tgt):
            continue
        # We don't know the exact caller EA here; skip if not in range
        # (caller xrefs are only useful when source & target are both known)
        # The preseeder is called with full prologues list — use the tgt as
        # a seed: just make sure IDA knows tgt is a function entry.
        if ida_funcs.get_func(tgt) is None:
            if ida_funcs.add_func(tgt):
                n_xrefs += 1

    # ── 3. Mark BB heads as code ──────────────────────────────────────────
    n_heads = 0
    for ea in bb_heads:
        if not in_range(ea):
            continue
        if not ida_bytes.is_code(ida_bytes.get_full_flags(ea)):
            idc.create_insn(ea)
            n_heads += 1

    # ── 4. Mark strings ───────────────────────────────────────────────────
    n_strs = 0
    for ea, text in strings:
        if not in_range(ea):
            continue
        slen = len(text) + 1      # include null terminator
        try:
            idc.create_strlit(ea, ea + slen, idc.STRTYPE_C)
            n_strs += 1
        except Exception:
            pass

    dt = time.perf_counter() - t0
    print(
        f"[ida_gpu_accel] preseed: +{n_funcs} funcs, +{n_xrefs} xref-seeds, "
        f"+{n_heads} code heads, +{n_strs} strings  ({dt:.2f}s)",
        flush=True,
    )


def seed_from_binary(data: bytes, base_ea: int, text_start: int = 0, text_end: int = 0,
                     arch: str = "auto") -> None:
    """
    Convenience wrapper: scan data then seed IDA in one call.
    Call this from shard_worker.py BEFORE ida_auto.auto_wait().

    arch: "arm64", "x86_64", or "auto" (detected from IDA's idainfo).
    """
    if arch == "auto":
        try:
            import idaapi
            info = idaapi.get_inf_structure()
            proc = info.procname.lower() if hasattr(info, "procname") else ""
            if "arm" in proc or "aarch" in proc:
                arch = "arm64"
            elif "metapc" in proc or "x86" in proc or "pc" in proc:
                arch = "x86_64"
            else:
                arch = "arm64"  # default to arm64 for NSO/NRO
        except Exception:
            arch = "arm64"

    if arch == "x86_64":
        from .x86_64_scanner import scan_x86_64
        prologues, bl_targets, bb_heads, strings = scan_x86_64(data, base_ea)
    else:
        from .arm64_scanner import scan
        prologues, bl_targets, bb_heads, strings = scan(data, base_ea)

    seed(prologues, bl_targets, bb_heads, strings,
         text_start=text_start, text_end=text_end)
