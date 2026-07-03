"""Write dynamic-analysis findings onto graph Function nodes.

Thin adapter over atlas's GraphAnnotator, reusing spectrIDA's existing Neo4j
driver (no second connection) and a product-native ``dyn_`` property prefix so
runtime facts sit alongside the static ones and surface through get_function().
"""
from __future__ import annotations


def annotator(graph):
    """Build a GraphAnnotator bound to spectrIDA's live graph driver.

    ``graph`` is the FunctionGraph (has ``.driver``). Properties are namespaced
    ``dyn_`` (e.g. dyn_status, dyn_reachable, dyn_crash_input) — SET-only on
    existing Function nodes, never MERGE (dynamic analysis annotates the static
    map, it does not invent functions)."""
    from atlas.analysis.graph_annotator import GraphAnnotator
    return GraphAnnotator(driver=graph.driver, prefix="dyn_")
