"""Dynamic analysis — the runtime corner of the RE triangle.

spectrIDA's static side finds and names functions; this side *runs* them (CPU
emulation or live instrumentation) and reports what actually happens — reachable?
crashes? needs live state? — writing those facts back onto the shared graph so the
agent driving the MCP server can reason over structure AND behavior together.

Requires the optional ``atlas`` dependency (heavy: torch/unicorn/frida). The base
spectrIDA install stays light; if the extra isn't present these tools return a
friendly install hint instead of an import error.
"""
from __future__ import annotations

import logging

# Neo4j emits INFO-level "notification" spam (schema-already-exists, property-not-
# found) on many queries; quiet it so dynamic-tool output stays readable.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

try:
    import atlas.analysis  # noqa: F401  (probe the optional dependency)
    AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as e:  # ImportError, or a broken partial install
    AVAILABLE = False
    _IMPORT_ERROR = e

INSTALL_HINT = (
    "Dynamic analysis needs Atlas (emulation / live instrumentation / fuzzing).\n"
    'Install it:  pip install "spectrida[atlas]"'
)


def require() -> None:
    """Raise a clear, actionable error if the atlas extra isn't installed.
    Call at the top of every dynamic MCP tool so a missing extra is a friendly
    message, not a confusing ImportError deep in a call stack."""
    if not AVAILABLE:
        raise RuntimeError(f"{INSTALL_HINT}\n(import error: {_IMPORT_ERROR})")


def status() -> dict:
    """Cheap availability probe for doctor()/diagnostics."""
    return {"available": AVAILABLE,
            "error": None if AVAILABLE else str(_IMPORT_ERROR)}
