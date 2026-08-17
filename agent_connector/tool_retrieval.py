"""Lightweight tool retrieval: match user query against tool registry.

No embeddings needed — keyword/token overlap scoring. Designed to sit
between the user task and the LLM function-calling layer:

    query -> retrieve_tools() -> top-k candidate ToolSpecs -> LLM

If no candidates match, returns an empty list so the caller can reply
with a clear "no matching tool" message instead of blindly passing all
tools to the LLM.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def _tokenize(text: str) -> set[str]:
    """Lowercase token set: split on non-alphanumeric, drop short tokens."""
    text = unicodedata.normalize("NFKC", text or "")
    tokens = set(re.findall(r"[a-z0-9]{2,}", text.lower()))
    # remove common stop words that add noise to matching
    tokens -= {"the", "a", "an", "is", "are", "was", "were", "be", "been",
               "being", "have", "has", "had", "do", "does", "did", "will",
               "would", "could", "should", "may", "might", "can", "shall",
               "to", "of", "in", "for", "on", "with", "at", "by", "from",
               "as", "into", "through", "during", "before", "after", "and",
               "or", "but", "not", "no", "nor", "so", "yet", "both",
               "either", "neither", "each", "every", "all", "any", "few",
               "more", "most", "other", "some", "such", "than", "too",
               "very", "just", "about", "above", "below", "up", "down",
               "out", "off", "over", "under", "again", "further", "then",
               "once", "here", "there", "when", "where", "why", "how",
               "what", "which", "who", "whom", "this", "that", "these",
               "those", "help", "file", "files", "use", "using", "used"}
    return tokens


def _tool_keywords(tool: dict) -> set[str]:
    """Extract base searchable keywords from a tool registry entry.

    Only includes tool-level name, description, and tags. Subcommand-specific
    keywords come from _subcommand_keywords() to avoid polluting the base set
    with every subcommand's tokens (which would make all entries identical)."""
    kw: set[str] = set()
    name = (tool.get("name") or "").lower().replace("-", " ").replace("_", " ")
    kw.update(_tokenize(name))
    kw.update(_tokenize(tool.get("description") or ""))
    for tag in (tool.get("tags") or []):
        kw.update(_tokenize(str(tag)))
    return kw


def _subcommand_keywords(tool: dict, sub: str,
                         detail: dict) -> set[str]:
    """Keywords specific to one subcommand (on top of parent tool keywords)."""
    kw: set[str] = set()
    kw.update(_tokenize(sub))
    kw.update(_tokenize(detail.get("description") or ""))
    for p in (detail.get("params") or []):
        kw.update(_tokenize(p.get("description") or ""))
        kw.update(_tokenize(p.get("name") or ""))
    kw.update(_tokenize(detail.get("usage") or ""))
    return kw


def build_tool_index(tools: list[dict]) -> list[dict]:
    """Precompute keyword sets for each tool AND each subcommand.

    Returns list of {tool, subcommand, fn_name, keywords, desc_keywords}
    dicts. desc_keywords are tokens from the subcommand's OWN description
    only (not params), scored separately to break ties between subcommands
    that share param keywords."""
    entries = []
    for t in tools:
        base_kw = _tool_keywords(t)
        if t.get("arg_style") == "subcommand" and t.get("subcommand_details"):
            for sub, detail in (t.get("subcommand_details") or {}).items():
                fn_name = f"{t['name']}_{sub.replace('-', '_')}"
                sub_kw = _subcommand_keywords(t, sub, detail)
                desc_kw = _tokenize(detail.get("description") or "")
                entries.append({"tool": t, "subcommand": sub,
                                "fn_name": fn_name,
                                "keywords": base_kw | sub_kw,
                                "desc_keywords": desc_kw})
        else:
            entries.append({"tool": t, "subcommand": None,
                            "fn_name": t.get("name", ""),
                            "keywords": base_kw,
                            "desc_keywords": set()})
    return entries


def retrieve_tools(query: str, index: list[dict],
                   top_k: int = 5, min_score: int = 2) -> list[dict]:
    """Find the most relevant tools/subcommands for a user query.

    Scoring tiers:
      - base keyword overlap: 1 point each
      - description keyword overlap (unique to this sub): 5 points each
      - subcommand name match in query: 10 bonus points
    This ensures subcommand names and descriptions dominate over shared
    parameter terms. Returns top_k candidates with score >= min_score."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scored = []
    for entry in index:
        overlap = q_tokens & entry["keywords"]
        desc_overlap = q_tokens & entry.get("desc_keywords", set())
        score = len(overlap) + len(desc_overlap) * 5
        sub = entry.get("subcommand") or ""
        if sub and sub in q_tokens:
            score += 10
        # Tie-break: if query implies direction (encode=toBINSEQ, decode=fromBINSEQ),
        # boost the entry whose description matches the target direction.
        # e.g. "convert BINSEQ to FASTA" -> target is FASTA -> boost decode
        if "convert" in q_tokens or "restore" in q_tokens:
            desc_text = " ".join(entry.get("desc_keywords", set()))
            if "restore" in desc_text or "back" in desc_text:
                score += 1
        if score >= min_score:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"tool": e["tool"], "subcommand": e["subcommand"],
             "fn_name": e["fn_name"], "score": s}
            for s, e in scored[:top_k]]
