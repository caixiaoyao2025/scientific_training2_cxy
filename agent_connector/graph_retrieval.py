"""Graph-based tool retrieval using graph-tool-call.

Replaces the keyword-overlap retrieval in tool_retrieval.py with
graph-tool-call's hybrid retrieval (keyword + graph + optional embedding).
"""
from __future__ import annotations

import re
import yaml
from graph_tool_call import ToolGraph


def build_graph_from_registry(registry_path: str) -> ToolGraph:
    """Load MCP registry YAML and build a ToolGraph for retrieval."""
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tools = data.get("tools", [])
    mcp_tools = []
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        params_schema = t.get("params_schema") or []
        properties = {}
        required = []
        for p in params_schema:
            pname = (p.get("name") or "").lstrip("-").replace("-", "_")
            if not pname:
                continue
            ptype = p.get("type") or "string"
            jstype = "string"
            if ptype in ("int", "integer"):
                jstype = "integer"
            elif ptype in ("float", "number"):
                jstype = "number"
            elif ptype in ("bool", "boolean"):
                jstype = "boolean"
            prop: dict = {"type": jstype}
            desc = p.get("description") or ""
            if desc:
                prop["description"] = desc
            if p.get("choices"):
                prop["enum"] = p["choices"]
            properties[pname] = prop
            if p.get("required") is True:
                required.append(pname)
        arg_style = t.get("arg_style") or "cli"
        if arg_style == "subcommand" and t.get("subcommand_details"):
            for sub, detail in t["subcommand_details"].items():
                fname = f"{t['name']}_{sub.replace('-', '_')}"
                sub_desc = detail.get("description") or f"{t['name']} {sub}"
                sub_params = detail.get("params_schema") or []
                sub_props = {}
                sub_required = []
                for p in sub_params:
                    pname = (p.get("name") or "").lstrip("-").replace("-", "_")
                    if not pname:
                        continue
                    ptype = p.get("type") or "string"
                    jstype = "string"
                    if ptype in ("int", "integer"):
                        jstype = "integer"
                    elif ptype in ("float", "number"):
                        jstype = "number"
                    elif ptype in ("bool", "boolean"):
                        jstype = "boolean"
                    prop = {"type": jstype}
                    desc = p.get("description") or ""
                    if desc:
                        prop["description"] = desc
                    if p.get("choices"):
                        prop["enum"] = p["choices"]
                    sub_props[pname] = prop
                    if p.get("required") is True:
                        sub_required.append(pname)
                mcp_tools.append({
                    "name": fname,
                    "description": sub_desc,
                    "inputSchema": {
                        "type": "object",
                        "properties": sub_props,
                        "required": sub_required,
                    },
                })
        else:
            mcp_tools.append({
                "name": t["name"],
                "description": t.get("description") or t["name"],
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
    graph = ToolGraph()
    graph.ingest_mcp_tools(mcp_tools, detect_dependencies=True)
    return graph


# -- Abstention thresholds --------------------------------------------------
# wRRF score ranges (from graph-tool-call internals):
#   "high"   confidence  score >= 0.02
#   "medium" confidence  score >= 0.01
#   "low"    confidence  score <  0.01
#
# We use two gates:
#   1. ABS_THRESHOLD (0.015) — hard floor, never return results below this
#   2. When score < 0.02 ("high"), require that the query's core concept
#      appears in the top result's description via _query_domain_match()
NO_MATCH_ABS_THRESHOLD = 0.013


# Domain-check: extract core concept from query and verify the top result
# description is topically aligned.  Returns True when the match is plausible.
_DOMAIN_PATTERNS = [
    (re.compile(r"\brna\b", re.I),          {"rna", "transcript", "secondary structure", "folding"}),
    (re.compile(r"\bprotein\b", re.I),      {"protein", "amino acid", "conformation", "structure"}),
    (re.compile(r"\bmetagenom", re.I),      {"metagenom", "species", "classification", "ont", "reads"}),
    (re.compile(r"\bfasta\b|\bfastq\b", re.I), {"fasta", "fastq", "sequence", "file", "format"}),
    (re.compile(r"\bbinseq\b", re.I),       {"binseq", "bin", "encode", "decode", "file"}),
]


def _query_domain_match(query: str, result_desc: str) -> bool:
    """Check if the query's domain concept is represented in the result description."""
    q_lower = query.lower()
    d_lower = result_desc.lower()
    for pattern, keywords in _DOMAIN_PATTERNS:
        if pattern.search(q_lower):
            return any(kw in d_lower for kw in keywords)
    return True  # no domain signal extracted → don't filter


def retrieve_tools(graph: ToolGraph, query: str, top_k: int = 5):
    """Retrieve top-k tools for a query.

    Returns list of (name, score, confidence).  Returns [] when no tool
    passes the abstention check:
      1. Absolute floor: score < 0.015  →  no match
      2. When score < "high" (0.02), verify the query's domain concept
         appears in the top result's description.  If the #1 result
         doesn't match, scan the full top-k for any domain-matching
         results and return those instead.
    """
    results = graph.retrieve_with_scores(query, top_k=top_k)
    if not results:
        return []
    best = results[0]
    if best.score < NO_MATCH_ABS_THRESHOLD:
        return []
    # High confidence: return as-is
    if best.score >= 0.02:
        return [(r.tool.name, r.score, r.confidence) for r in results]
    # Medium confidence: check domain alignment of #1
    if _query_domain_match(query, best.tool.description):
        return [(r.tool.name, r.score, r.confidence) for r in results]
    # #1 doesn't match domain — scan top-k for any domain-matching result
    filtered = [
        (r.tool.name, r.score, r.confidence)
        for r in results
        if _query_domain_match(query, r.tool.description)
    ]
    return filtered
