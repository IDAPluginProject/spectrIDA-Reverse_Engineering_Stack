"""IDB-as-RAG — harvest referenced knowledge from the IDA database.

Before naming a function, extract what IDA already knows at every address
that function references — globals, comments, types, strings — and inject
it into the naming prompt. Unlike N-hop context (which shows *which functions*
neighbor this one), this surfaces *what's already known* about the things
this function touches.

Two signals that compose:
  - N-hop: "this function calls memcpy and is called by player_update"
  - IDB-RAG: "address 0x18000A120 has comment '// SOCPacket handler'"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Skip these noise names — they carry zero signal
_NOISE_PREFIXES = (
    "sub_", "dword_", "off_", "loc_", "unk_", "byte_", "word_",
    "qword_", "asc_", "flt_", "dbl_", "def_", "seg_",
)

_MAX_REFS_PER_TYPE = 15
_MAX_KNOWLEDGE_ENTRIES = 20
_MAX_CONTEXT_CHARS = 2000


@dataclass
class ReferencedAddress:
    addr: str
    ref_type: str  # "code" | "data" | "string"
    name: str = ""


@dataclass
class KnowledgeEntry:
    addr: str
    name: str = ""
    comment: str = ""
    type_str: str = ""
    string_val: str = ""
    ref_type: str = ""  # from the reference that brought us here

    @property
    def has_content(self) -> bool:
        return bool(self.name or self.comment or self.type_str or self.string_val)


def _is_noise(name: str) -> bool:
    """Check if a name is an IDA placeholder."""
    if not name:
        return True
    return name.lower().startswith(_NOISE_PREFIXES)


async def harvest_references(db, func_addr: int) -> dict[str, list[ReferencedAddress]]:
    """Get all referenced addresses from a function body.

    Returns {"code": [...], "data": [...], "string": [...]} with
    capped, deduplicated references.
    """
    raw = await db.refs(func_addr)

    result: dict[str, list[ReferencedAddress]] = {}
    for ref_type in ("code", "data", "string"):
        refs = raw.get(ref_type, [])
        seen = set()
        entries = []
        for r in refs:
            addr = r.get("addr", "")
            if addr in seen:
                continue
            seen.add(addr)
            entries.append(ReferencedAddress(
                addr=addr,
                ref_type=ref_type,
                name=r.get("name", "") or r.get("value", ""),
            ))
            if len(entries) >= _MAX_REFS_PER_TYPE:
                break
        result[ref_type] = entries

    return result


async def gather_knowledge(
    db,
    references: dict[str, list[ReferencedAddress]],
) -> list[KnowledgeEntry]:
    """Look up what IDA knows at a set of referenced addresses.

    Returns filtered entries — only addresses with meaningful knowledge
    (named, commented, typed, or string-valued).
    """
    # Collect all unique addresses
    all_addrs = []
    addr_ref_type = {}
    for ref_type, refs in references.items():
        for ref in refs:
            all_addrs.append(ref.addr)
            addr_ref_type[ref.addr] = ref_type

    if not all_addrs:
        return []

    # Batch lookup
    raw = await db.knowledge(all_addrs[:50])  # cap total

    entries = []
    for item in raw:
        entry = KnowledgeEntry(
            addr=item.get("addr", ""),
            name=item.get("name", ""),
            comment=item.get("comment", ""),
            type_str=item.get("type", ""),
            string_val=item.get("string", ""),
            ref_type=addr_ref_type.get(item.get("addr", ""), ""),
        )
        if entry.has_content:
            entries.append(entry)

    # Prioritize: strings and named globals first, then typed data, then comments
    def priority(e: KnowledgeEntry) -> int:
        if e.string_val:
            return 0  # highest — actual string content
        if e.name and not _is_noise(e.name):
            return 1  # named globals/functions
        if e.type_str:
            return 2  # typed but unnamed
        if e.comment:
            return 3  # has comment
        return 4

    entries.sort(key=priority)
    return entries[:_MAX_KNOWLEDGE_ENTRIES]


def format_knowledge_block(
    entries: list[KnowledgeEntry],
    already_in_context: set[str] | None = None,
) -> str:
    """Format knowledge entries into a compact context block.

    Merges with existing N-hop context — deduplicates names already
    shown as caller/callee names.
    """
    if not entries:
        return ""

    if already_in_context is None:
        already_in_context = set()

    lines: list[str] = []
    skipped = 0

    for e in entries:
        # Skip if already in N-hop context
        if e.name and e.name in already_in_context:
            skipped += 1
            continue

        parts: list[str] = []

        if e.string_val:
            escaped = e.string_val.replace('"', '\\"')[:60]
            parts.append(f'"{escaped}"')
            if e.name:
                parts.append(e.name)
        elif e.name:
            parts.append(e.name)
            if e.comment:
                parts.append(f"// {e.comment[:40]}")
        elif e.comment:
            parts.append(f"comment: {e.comment[:60]}")
        elif e.type_str:
            parts.append(f"({e.type_str})")

        if parts:
            line = " ".join(parts)
            if len(line) > 80:
                line = line[:77] + "..."
            lines.append(line)

        if len(lines) >= 12:
            break

    if skipped > 0:
        lines.append(f"(+{skipped} already in callgraph context)")

    if not lines:
        return ""

    block = "Known references:\n" + "\n".join(f"  {l}" for l in lines)

    if len(block) > _MAX_CONTEXT_CHARS:
        block = block[:_MAX_CONTEXT_CHARS - 20] + "\n  ...(truncated)"

    return block
