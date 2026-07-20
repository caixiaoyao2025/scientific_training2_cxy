import json
import re
import requests
from datetime import datetime

# ============================================================
# 1. 加载原始数据
# ============================================================
def load_raw_results(filename="github_from_html.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 找不到 {filename}，请先运行 agent.py")
        return []

# ============================================================
# 2. 从仓库名猜测工具类型
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
# 3. 获取GitHub仓库信息（星星数等）
# ============================================================
def get_github_repo_info(repo_url):
    """从GitHub API获取仓库信息"""
    if not repo_url or "github.com" not in repo_url:
        return {"stars": 0, "description": "", "language": ""}
    
    try:
        # 解析 owner/repo
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
# 4. 生成标签
# ============================================================
def generate_tags(tool_name, paper_title, tool_type):
    """根据工具名、论文标题生成标签"""
    tags = set()
    
    # 从工具名提取
    words = re.sub(r'[^a-zA-Z ]', ' ', tool_name).split()
    for w in words:
        if len(w) > 2:
            tags.add(w.lower())
    
    # 从论文标题提取关键词
    title_words = re.sub(r'[^a-zA-Z ]', ' ', paper_title).split()
    bio_keywords = ["protein", "design", "fold", "structure", "genome", "sequence", 
                    "bio", "enzyme", "docking", "simulation", "molecular"]
    for w in title_words:
        if w.lower() in bio_keywords:
            tags.add(w.lower())
    
    # 添加类型标签
    tags.add(tool_type)
    
    return list(tags)[:10]  # 最多10个

# ============================================================
# 5. 计算质量评分
# ============================================================
def calculate_quality_score(github_info, paper_count=1):
    """计算工具质量评分（0-100）"""
    score = 0
    
    # GitHub星星：0星0分，50星以上满分
    stars = github_info.get("stars", 0)
    score += min(stars / 50 * 30, 30)
    
    # 有描述加分
    if github_info.get("description"):
        score += 10
    
    # 有语言标签加分
    if github_info.get("language"):
        score += 10
    
    # 最近更新（3个月内）加分
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
    
    # 被多篇论文提及加分
    score += min(paper_count * 5, 20)
    
    return min(score, 100)

# ============================================================
# 6. 主转换函数
# ============================================================
def convert_to_standard(raw_results):
    """把原始数据转换成标准化工具列表"""
    standardized_tools = []
    
    # 先按GitHub链接去重，合并多篇论文的信息
    tool_map = {}
    
    for item in raw_results:
        github_links = item.get("github_links", [])
        if not github_links:
            continue
        
        # 只取第一个GitHub链接（大多数情况只有一个）
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
    
    # 转换为标准化格式
    for github_url, data in tool_map.items():
        github_info = data["github_info"]
        tool_name = github_url.split("/")[-1] or "未知工具"
        
        # 从GitHub描述或论文标题取描述
        description = github_info.get("description", "")
        if not description and data["paper_titles"]:
            description = data["paper_titles"][0][:200]
        
        tool_type = guess_tool_type(tool_name, description)
        
        # 生成标签
        tags = generate_tags(
            tool_name, 
            data["paper_titles"][0] if data["paper_titles"] else "",
            tool_type
        )
        
        # 计算质量评分
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
                "cli": True,  # 默认假设有CLI，可以后续优化
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
    
    # 按质量评分排序
    standardized_tools.sort(key=lambda x: x["quality_score"], reverse=True)
    
    return standardized_tools

# ============================================================
# 7. 保存标准化工具库
# ============================================================
def save_tool_library(standardized_tools, filename="tool_library.json"):
    """保存标准化工具库"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(standardized_tools, f, ensure_ascii=False, indent=2)
    print(f"📁 保存了 {len(standardized_tools)} 个标准化工具到 {filename}")

# ============================================================
# 8. 生成下游Agent可直接读取的摘要
# ============================================================
def generate_summary(standardized_tools):
    """生成一个简洁的工具摘要，方便下游Agent快速了解"""
    summary = {
        "total_tools": len(standardized_tools),
        "by_type": {},
        "high_quality": [],  # 质量分>=60的工具
        "latest": []
    }
    
    for tool in standardized_tools:
        # 按类型统计
        tool_type = tool.get("type", "unknown")
        summary["by_type"][tool_type] = summary["by_type"].get(tool_type, 0) + 1
        
        # 高质量工具
        if tool.get("quality_score", 0) >= 60:
            summary["high_quality"].append({
                "name": tool["name"],
                "description": tool["description"][:100],
                "github": tool["source"]["github"],
                "score": tool["quality_score"]
            })
    
    # 最新的5个
    summary["latest"] = standardized_tools[:5]
    
    with open("tool_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📁 保存了工具摘要到 tool_summary.json")
    
    return summary

# ============================================================
# 9. 主程序
# ============================================================
if __name__ == "__main__":
    print("🔄 开始转换工具数据...")
    
    # 加载原始数据
    raw_results = load_raw_results("github_from_html.json")
    print(f"📄 找到 {len(raw_results)} 条原始记录")
    
    if not raw_results:
        print("⚠️ 没有数据可转换，请先运行 agent.py")
        exit()
    
    # 转换
    standardized = convert_to_standard(raw_results)
    print(f"🛠️ 生成了 {len(standardized)} 个标准化工具")
    
    # 保存
    save_tool_library(standardized)
    summary = generate_summary(standardized)
    
    print(f"\n📊 统计:")
    print(f"   - 总工具数: {summary['total_tools']}")
    print(f"   - 按类型: {summary['by_type']}")
    print(f"   - 高质量工具(>=60分): {len(summary['high_quality'])}")
    print("\n✅ 转换完成！")