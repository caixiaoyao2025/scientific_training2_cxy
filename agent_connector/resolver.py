"""Optional LLM pass that confirms/refines the rule-based agent_schema.

Uses any OpenAI-compatible chat endpoint (OpenAI, SiliconFlow, vLLM, ...).
If no API key is available the rule-based result is returned unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

RESOLVER_SYSTEM_PROMPT = """You are an expert in Python agent frameworks.
You are given AST-derived candidate interfaces for how a framework
registers and executes tools. Based on the evidence, produce a JSON object
with exactly these keys:
{
  "agent_class": string|null,
  "registration_method": string|null,
  "registration_argument": string|null,
  "registration_via_decorator": boolean,
  "execution_method": string|null,
  "execution_class": string|null,
  "tool_class": string|null,
  "confidence": number 0..1,
  "reasoning": string
}
Only pick candidates that are clearly real tool interfaces. Be conservative."""


def resolve_schema_with_llm(
    schema: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str = "https://api.siliconflow.cn/v1",
    model: str = "Qwen/Qwen2.5-72B-Instruct-128K",
    include_evidence: bool = False,
) -> dict[str, Any]:
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        schema["resolver"] = {"status": "skipped_no_api_key"}
        return schema

    try:
        from openai import OpenAI
    except ImportError as exc:
        schema["resolver"] = {"status": "skipped_no_openai", "reason": str(exc)}
        return schema

    evidence = {
        "registrations": schema.get("registrations"),
        "executions": schema.get("executions"),
        "tool_classes": schema.get("tool_classes"),
    }

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RESOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, indent=2)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        schema["resolver"] = {"status": "error", "reason": str(exc)}
        return schema

    content = response.choices[0].message.content
    try:
        resolved = json.loads(content)
    except json.JSONDecodeError:
        schema["resolver"] = {"status": "error", "reason": "LLM returned non-JSON", "raw": content[:500]}
        return schema

    for key in ("agent_class", "registration_method", "registration_argument", "execution_method",
                "execution_class", "tool_class"):
        if resolved.get(key):
            schema[key] = resolved[key]
    if isinstance(resolved.get("confidence"), (int, float)):
        schema["confidence"] = resolved["confidence"]
    schema["resolver"] = {"status": "ok", "reasoning": resolved.get("reasoning", "")}
    return schema
