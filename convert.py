import json
import re
import requests
from datetime import datetime

# ============================================================
# 1. Load raw data
# ============================================================
def load_raw_results(filename="github_from_html.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {filename} not found, please run agent.py first")
        return []

# ============================================================
# 2. Guess tool type from repository name
# ============================================================
def guess_tool_type(repo_name, description=""):
    """根据仓库名和描述猜测工具类型"""
    text = (repo_name + " " + description).lower()
    
    if "server" in text or "web" in text or "api" in text:
        return "web_server"
    elif "database" in text or "db" in text:
        return "database"
    elif "pipeline" in text or "workflow" in text:
        return "pipeline"
    elif "library" in text or "lib" in text or "sdk" in text:
        return "library"
    else:
        return "software"

# ============================================================
# 3. Fetch GitHub repository info (stars, etc.)
# ============================================================
def get_github_repo_info(repo_url):
    """从GitHub API获取仓库信息"""
    if not repo_url or "github.com" not in repo_url:
        return {"stars": 0, "description": "", "language": ""}
    
    try:
        # Parse owner/repo
        parts = repo_url.rstrip('/').split('/')
        if len(parts) < 5:
            return {"stars": 0, "description": "", "language": ""}
        owner_repo = parts[3] + "/" + parts[4]
        
        api_url = f"https://api.github.com/repos/{owner_repo}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "stars": data.get("stargazers_count", 0),
                "description": data.get("description", ""),
                "language": data.get("language", ""),
                "updated_at": data.get("updated_at", ""),
                "forks": data.get("forks_count", 0)
            }
    except:
        pass
    
    return {"stars": 0, "description": "", "language": ""}

# ============================================================
# 4. Generate tags
# ============================================================
def generate_tags(tool_name, paper_title, tool_type):
    """根据工具名、论文标题生成标签"""
    tags = set()
    
    # Extract from tool name
    words = re.sub(r'[^a-zA-Z ]', ' ', tool_name).split()
    for w in words:
        if len(w) > 2:
            tags.add(w.lower())
    
    # Extract keywords from paper title
    title_words = re.sub(r'[^a-zA-Z ]', ' ', paper_title).split()
    bio_keywords = ["protein", "design", "fold", "structure", "genome", "sequence", 
                    "bio", "enzyme", "docking", "simulation", "molecular"]
    for w in title_words:
        if w.lower() in bio_keywords:
            tags.add(w.lower())
    
    # Add type tag
    tags.add(tool_type)
    
    return list(tags)[:10]  # Max 10

# ============================================================
# 5. Calculate quality score
# ============================================================
def calculate_quality_score(github_info, paper_count=1):
    """计算工具质量评分（0-100）"""
    score = 0
    
    # GitHub stars: 0 stars = 0 points, 50+ stars = max
    stars = github_info.get("stars", 0)
    score += min(stars / 50 * 30, 30)
    
    # Bonus for having description
    if github_info.get("description"):
        score += 10
    
    # Bonus for having language tag
    if github_info.get("language"):
        score += 10
    
    # Bonus for recent update (within 3 months)
    updated = github_info.get("updated_at", "")
    if updated:
        try:
            from dateutil.parser import parse
            last_update = parse(updated)
            days_ago = (datetime.now() - last_update).days
            if days_ago < 30:
                score += 20
            elif days_ago < 90:
                score += 10
        except:
            pass
    
    # Bonus for being mentioned in multiple papers
    score += min(paper_count * 5, 20)
    
    return min(score, 100)

# ============================================================
# 6. Main conversion function
# ============================================================
def convert_to_standard(raw_results):
    """把原始数据转换成标准化工具列表"""
    standardized_tools = []
    
    # First deduplicate by GitHub link, merge info from multiple papers
    tool_map = {}
    
    for item in raw_results:
        github_links = item.get("github_links", [])
        if not github_links:
            continue
        
        # Take only the first GitHub link (usually just one)
        github_url = github_links[0]
        
        if github_url not in tool_map:
            tool_map[github_url] = {
                "github_url": github_url,
                "paper_titles": [],
                "paper_dois": [],
                "github_info": get_github_repo_info(github_url)
            }
        
        tool_map[github_url]["paper_titles"].append(item.get("title", ""))
        if item.get("doi"):
            tool_map[github_url]["paper_dois"].append(item.get("doi"))
    
    # Convert to standardized format
    for github_url, data in tool_map.items():
        github_info = data["github_info"]
        tool_name = github_url.split("/")[-1] or "unknown_tool"
        
        # Get description from GitHub or paper title
        description = github_info.get("description", "")
        if not description and data["paper_titles"]:
            description = data["paper_titles"][0][:200]
        
        tool_type = guess_tool_type(tool_name, description)
        
        # Generate tags
        tags = generate_tags(
            tool_name, 
            data["paper_titles"][0] if data["paper_titles"] else "",
            tool_type
        )
        
        # Calculate quality score
        quality_score = calculate_quality_score(
            github_info, 
            len(data["paper_titles"])
        )
        
        standardized_tool = {
            "name": tool_name,
            "description": description,
            "source": {
                "github": github_url,
                "paper_doi": data["paper_dois"][0] if data["paper_dois"] else "",
                "paper_title": data["paper_titles"][0] if data["paper_titles"] else ""
            },
            "type": tool_type,
            "interface": {
                "cli": True,  # Default assume CLI, can optimize later
                "api": False,
                "web": tool_type == "web_server"
            },
            "tags": tags,
            "quality_score": quality_score,
            "github_metadata": {
                "stars": github_info.get("stars", 0),
                "language": github_info.get("language", ""),
                "forks": github_info.get("forks", 0)
            },
            "discovered_at": datetime.now().isoformat()
        }
        
        standardized_tools.append(standardized_tool)
    
    # Sort by quality score
    standardized_tools.sort(key=lambda x: x["quality_score"], reverse=True)
    
    return standardized_tools

# ============================================================
# 7. Save standardized tool library
# ============================================================
def save_tool_library(standardized_tools, filename="tool_library.json"):
    """保存标准化工具库"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(standardized_tools, f, ensure_ascii=False, indent=2)
    print(f"📁 Saved {len(standardized_tools)} standardized tools to {filename}")

# ============================================================
# 8. Generate summary for downstream agents
# ============================================================
def generate_summary(standardized_tools):
    """生成一个简洁的工具摘要，方便下游Agent快速了解"""
    summary = {
        "total_tools": len(standardized_tools),
        "by_type": {},
        "high_quality": [],  # Tools with quality score >= 60
        "latest": []
    }
    
    for tool in standardized_tools:
        # Count by type
        tool_type = tool.get("type", "unknown")
        summary["by_type"][tool_type] = summary["by_type"].get(tool_type, 0) + 1
        
        # High quality tools
        if tool.get("quality_score", 0) >= 60:
            summary["high_quality"].append({
                "name": tool["name"],
                "description": tool["description"][:100],
                "github": tool["source"]["github"],
                "score": tool["quality_score"]
            })
    
    # Latest 5
    summary["latest"] = standardized_tools[:5]
    
    with open("tool_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📁 Saved tool summary to tool_summary.json")
    
    return summary

# ============================================================
# 9. Main program
# ============================================================
if __name__ == "__main__":
    print("🔄 Starting tool data conversion...")
    
    # Load raw data
    raw_results = load_raw_results("github_from_html.json")
    print(f"📄 Found {len(raw_results)} raw records")
    
    if not raw_results:
        print("⚠️ No data to convert, please run agent.py first")
        exit()
    
    # Convert
    standardized = convert_to_standard(raw_results)
    print(f"🛠️ Generated {len(standardized)} standardized tools")
    
    # Save
    save_tool_library(standardized)
    summary = generate_summary(standardized)
    
    print(f"\n📊 Statistics:")
    print(f"   - Total tools: {summary['total_tools']}")
    print(f"   - By type: {summary['by_type']}")
    print(f"   - High quality tools (>=60): {len(summary['high_quality'])}")
    print("\n✅ Conversion complete!")