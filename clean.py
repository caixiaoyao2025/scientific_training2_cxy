import json
import re

# Load tool library
with open("tool_library.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

# 1. Filter irrelevant tools
irrelevant_patterns = [
    r'normalize\.css',
    r'jquery',
    r'bootstrap',
    r'react',
    r'angular',
    r'vue\.js',
    r'd3\.js',
    r'three\.js'
]

filtered_tools = []
for tool in tools:
    name = tool.get("name", "").lower()
    desc = tool.get("description", "").lower()
    combined = name + " " + desc
    
    # Skip irrelevant tools
    is_irrelevant = any(re.search(p, combined) for p in irrelevant_patterns)
    if is_irrelevant:
        print(f"⏭️ Skipping irrelevant tool: {tool.get('name')}")
        continue
    
    # Clean trailing quotes from links
    github = tool.get("source", {}).get("github", "")
    if github.endswith('"') or github.endswith("'"):
        tool["source"]["github"] = github.rstrip('"\'')
    
    # Infer more accurate type from description
    desc = tool.get("description", "").lower()
    if "database" in desc or "ontology" in desc or "dictionary" in desc:
        tool["type"] = "database"
    elif "pipeline" in desc or "workflow" in desc or "benchmark" in desc:
        tool["type"] = "pipeline"
    elif "server" in desc or "web" in desc or "api" in desc:
        tool["type"] = "web_server"
    elif "library" in desc or "package" in desc:
        tool["type"] = "library"
    else:
        tool["type"] = "software"  # Keep default
    
    # Add useful tags: extract domain keywords from description
    domain_keywords = ["protein", "dna", "rna", "genome", "sequence", "structure", 
                       "binding", "folding", "design", "evolution", "antibody"]
    for kw in domain_keywords:
        if kw in desc:
            if kw not in tool["tags"]:
                tool["tags"].append(kw)
    
    filtered_tools.append(tool)

# Re-sort by quality score descending
filtered_tools.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

# Save cleaned tool library
with open("tool_library_clean.json", "w", encoding="utf-8") as f:
    json.dump(filtered_tools, f, ensure_ascii=False, indent=2)

print(f"\n✅ Cleaning done! Kept {len(filtered_tools)} tools")
print(f"📁 Saved to tool_library_clean.json")

# Print summary (optimized output)
print("\n📊 Cleaned tool list:")
print("-" * 60)
print(f"{'Tool Name':<25} {'Type':<12} {'⭐ Stars':<10} {'Quality'}")
print("-" * 60)
for tool in filtered_tools:
    name = tool.get('name', '')[:24]
    tool_type = tool.get('type', '')
    stars = tool.get('github_metadata', {}).get('stars', 0)
    score = tool.get('quality_score', 0)
    print(f"{name:<25} {tool_type:<12} {stars:<10} {score}")
print("-" * 60)