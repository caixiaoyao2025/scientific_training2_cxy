"""Graph-based tool retrieval using graph-tool-call.

Replaces the keyword-overlap retrieval in tool_retrieval.py with
graph-tool-call's hybrid retrieval (keyword + graph + optional embedding).
"""
from __future__ import annotations

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
        # Convert ToolSpec params_schema to MCP inputSchema
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
            elif ptype == "bool" or ptype == "boolean":
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
        # Add subcommand leaf functions for subcommand tools
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
                    elif ptype == "bool" or ptype == "boolean":
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


def retrieve_tools(graph: ToolGraph, query: str, top_k: int = 5):
    """Retrieve top-k tools for a query. Returns list of (name, score, confidence)."""
    results = graph.retrieve_with_scores(query, top_k=top_k)
    return [(r.tool.name, r.score, r.confidence) for r in results]
