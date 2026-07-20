import json
import re

# 加载工具库
with open("tool_library.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

# 1. 过滤无关工具
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
    
    # 跳过无关工具
    is_irrelevant = any(re.search(p, combined) for p in irrelevant_patterns)
    if is_irrelevant:
        print(f"⏭️ 跳过无关工具: {tool.get('name')}")
        continue
    
    # 清理链接末尾的引号
    github = tool.get("source", {}).get("github", "")
    if github.endswith('"') or github.endswith("'"):
        tool["source"]["github"] = github.rstrip('"\'')
    
    # 根据描述推断更准确的类型
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
        tool["type"] = "software"  # 保持默认
    
    # 添加一个有用的标签：从描述中提取领域关键词
    domain_keywords = ["protein", "dna", "rna", "genome", "sequence", "structure", 
                       "binding", "folding", "design", "evolution", "antibody"]
    for kw in domain_keywords:
        if kw in desc:
            if kw not in tool["tags"]:
                tool["tags"].append(kw)
    
    filtered_tools.append(tool)

# 重新排序：按质量分降序
filtered_tools.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

# 保存清理后的工具库
with open("tool_library_clean.json", "w", encoding="utf-8") as f:
    json.dump(filtered_tools, f, ensure_ascii=False, indent=2)

print(f"\n✅ 清理完成！保留 {len(filtered_tools)} 个工具")
print(f"📁 保存到 tool_library_clean.json")

# 打印摘要（优化输出）
print("\n📊 清理后的工具列表:")
print("-" * 60)
print(f"{'工具名':<25} {'类型':<12} {'⭐ Stars':<10} {'质量分'}")
print("-" * 60)
for tool in filtered_tools:
    name = tool.get('name', '')[:24]
    tool_type = tool.get('type', '')
    stars = tool.get('github_metadata', {}).get('stars', 0)
    score = tool.get('quality_score', 0)
    print(f"{name:<25} {tool_type:<12} {stars:<10} {score}")
print("-" * 60)