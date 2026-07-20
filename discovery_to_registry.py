import json
import re
import yaml

def load_tool_library(filename="tool_library_clean.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def guess_install_method(tool):
    lang = tool.get("github_metadata", {}).get("language", "").lower()
    github_url = tool.get("source", {}).get("github", "")
    name = tool.get("name", "").lower()

    if lang == "python":
        return "pip_url", github_url
    elif lang in ("go", "rust", "c", "c++"):
        return "binary_url", github_url
    else:
        return "pip_url", github_url

def guess_command(tool):
    name = tool.get("name", "").replace(" ", "_").replace("-", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return f"{name.lower()} {{{{input_file}}}}"

def tool_to_registry_entry(tool):
    name = tool.get("name", "unknown")
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name).strip('_').lower()
    description = tool.get("description", "No description available.")
    github_url = tool.get("source", {}).get("github", "")
    stars = tool.get("github_metadata", {}).get("stars", 0)
    language = tool.get("github_metadata", {}).get("language", "")
    tags = tool.get("tags", [])
    quality_score = tool.get("quality_score", 0)
    paper_doi = tool.get("source", {}).get("paper_doi", "")
    paper_title = tool.get("source", {}).get("paper_title", "")

    install_method, install_url = guess_install_method(tool)
    command_template = guess_command(tool)

    entry = {
        "name": clean_name,
        "type": "cli",
        "command": command_template,
        "description": f"[Auto-discovered] {description} (⭐{stars}, {language})",
        "output_control": {
            "intercept_large_output": True,
            "max_preview_lines": 50,
        },
        "inputs": {
            "input_file": {
                "type": "string",
                "description": "Input file path inside /data."
            }
        },
        "_discovery_metadata": {
            "github": github_url,
            "stars": stars,
            "language": language,
            "tags": tags,
            "quality_score": quality_score,
            "paper_doi": paper_doi,
            "paper_title": paper_title,
            "install_method": install_method,
            "install_url": install_url,
        }
    }

    return entry

def convert_to_registry(tools, output_file="discovered_registry.yaml"):
    registry = {"tools": []}

    for tool in tools:
        entry = tool_to_registry_entry(tool)
        registry["tools"].append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(registry, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Converted {len(tools)} tools to {output_file}")

    high_quality = [t for t in tools if t.get("quality_score", 0) >= 40]
    print(f"High quality tools (score>=40): {len(high_quality)}")
    for t in high_quality[:5]:
        print(f"  - {t['name']}: {t.get('description', '')[:60]}...")

if __name__ == "__main__":
    tools = load_tool_library()
    print(f"Loaded {len(tools)} tools from tool_library_clean.json")
    convert_to_registry(tools)
