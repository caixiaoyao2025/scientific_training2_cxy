import json
import requests
import re
import time
import random
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import json
import os

# ============================================================
# 记忆管理：记录已处理的论文
# ============================================================
SEEN_FILE = "seen_papers.json"

def load_seen_papers():
    """加载已处理论文的ID列表"""
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_seen_papers(seen_ids):
    """保存已处理论文的ID列表"""
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_ids), f, ensure_ascii=False, indent=2)

def is_paper_seen(paper_id):
    """检查某篇论文是否已处理过"""
    seen = load_seen_papers()
    return paper_id in seen

def mark_paper_as_seen(paper_id):
    """标记某篇论文为已处理"""
    seen = load_seen_papers()
    seen.add(paper_id)
    save_seen_papers(seen)
# ============================================================
# 1. 搜索论文（PubMed）
# ============================================================
def search_papers(query, max_results=10):
    """搜索PubMed，返回论文列表"""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    
    response = requests.get(search_url, params=params)
    data = response.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    
    if not ids:
        return []
    
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json"
    }
    
    response = requests.get(summary_url, params=params)
    data = response.json()
    
    papers = []
    for uid in ids:
        doc = data.get("result", {}).get(uid, {})
        doi = ""
        for article_id in doc.get("articleids", []):
            if article_id.get("idtype") == "doi":
                doi = article_id.get("value")
                break
        
        papers.append({
            "pmid": uid,
            "title": doc.get("title", "无标题"),
            "abstract": doc.get("abstract", ""),
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
        })
    
    return papers

# ============================================================
# 2. 使用Playwright获取JavaScript渲染后的内容
# ============================================================
def fetch_with_playwright(url):
    """使用Playwright获取JavaScript渲染后的内容（攻克复杂出版社）"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(60000)
            page.goto(url, timeout=60000)
            page.wait_for_selector('body', timeout=10000)
            page.wait_for_timeout(3000)  # 等待动态内容加载
            content = page.content()
            browser.close()
            if content and len(content) > 500:
                return content
            return None
    except Exception as e:
        print(f"   ⚠️ Playwright获取失败: {e}")
        return None

# ============================================================
# 3. 直接解析HTML（BeautifulSoup）
# ============================================================
def parse_html_directly(html, url):
    """使用BeautifulSoup直接解析HTML"""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        content = ""

        if 'mdpi.com' in url:
            print("   📡 检测到 MDPI 页面，使用专项解析...")
            # MDPI 的正文通常在这个div里
            main_div = soup.find('div', id='main-content') or soup.find('div', class_='article-content') or soup.find('div', class_='html-content')
            if main_div:
                content = main_div.get_text(separator='\n', strip=True)
            # 如果没找到，尝试用 <article> 标签
            if not content or len(content) < 200:
                article_tag = soup.find('article')
                if article_tag:
                    content = article_tag.get_text(separator='\n', strip=True)
            # 如果还是没有，尝试用包含 'content' 的 div 作为备选
            if not content or len(content) < 200:
                for div in soup.find_all('div', class_=re.compile(r'content')):
                    div_text = div.get_text(separator='\n', strip=True)
                    if len(div_text) > len(content):
                        content = div_text
        
        if not content or len(content) < 200:
            main_content = soup.find('article') or soup.find('main')
            if main_content:
                content = main_content.get_text(separator='\n', strip=True)
        
        # 策略2: 查找包含关键class的div
        if not content or len(content) < 200:
            for div in soup.find_all('div', class_=re.compile(r'fulltext|article|main|content|body|abstract|pdf')):
                div_text = div.get_text(separator='\n', strip=True)
                if len(div_text) > len(content):
                    content = div_text
        
        # 策略3: 收集所有段落
        if not content or len(content) < 200:
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 20])
        
        # 策略4: 兜底使用body
        if not content or len(content) < 100:
            body = soup.find('body')
            if body:
                content = body.get_text(separator='\n', strip=True)

        if content and len(content) > 100:
            print(f"   ✅ 直接解析成功，获取 {len(content)} 字符")
            return content
        else:
            print(f"   ⚠️ 直接解析内容过短 ({len(content) if content else 0} 字符)")
            return None

    except Exception as e:
        print(f"   ⚠️ 直接解析异常: {e}")
        return None

# ============================================================
# 4. 增强版：获取论文全文（整合所有策略）
# ============================================================
def fetch_html_from_doi(doi, retries=2):
    """增强版：使用Session + Playwright备选，获取论文全文"""
    if not doi:
        return None

    # 增强的请求头，模拟真实浏览器
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

    paper_url = f"https://doi.org/{doi}"
    
    # --- 第一阶段：使用Session获取重定向后的最终URL ---
    try:
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(paper_url, timeout=30, allow_redirects=True)
        final_url = response.url
        print(f"   📡 最终地址: {final_url}")
    except Exception as e:
        print(f"   ⚠️ 请求失败: {e}")
        return None

    # --- 第二阶段：判断出版社，选择策略 ---
    # 需要重点攻坚的复杂出版社列表
    complex_publishers = ['elsevier', 'springer', 'tandfonline', 'wiley', 'nature', 'science', 'oup', 'oxford']
    
    if any(domain in final_url for domain in complex_publishers):
        print(f"   📡 检测到复杂出版社({final_url.split('/')[2]})，使用多层策略...")
        
        # 策略A：尝试Playwright（模拟真实浏览器）
        print(f"   🌐 尝试Playwright...")
        playwright_content = fetch_with_playwright(final_url)
        if playwright_content and len(playwright_content) > 500:
            print(f"   ✅ Playwright成功获取 {len(playwright_content)} 字符")
            return playwright_content
        else:
            print(f"   ⚠️ Playwright获取失败或内容过短")
        
        # 策略B：尝试Jina Reader
        print(f"   📡 尝试Jina Reader...")
        for attempt in range(retries + 1):
            try:
                jina_url = f"https://r.jina.ai/{final_url}"
                jina_response = requests.get(jina_url, headers={"User-Agent": headers["User-Agent"]}, timeout=30)
                if jina_response.status_code == 200 and len(jina_response.text) > 500:
                    print(f"   ✅ Jina Reader成功获取 {len(jina_response.text)} 字符")
                    return jina_response.text
                else:
                    print(f"   ⚠️ Jina Reader尝试 {attempt+1}/{retries+1} 失败")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"   ⚠️ Jina Reader异常: {e}")
                time.sleep(2 ** attempt)
        
        # 策略C：最终备选，直接解析
        print(f"   📡 所有高级策略失败，尝试直接解析...")
        return parse_html_directly(response.text, final_url)

    # 简单出版社（如PMC、arXiv），直接解析
    elif any(domain in final_url for domain in ['ncbi.nlm.nih.gov', 'arxiv.org', 'pubmed']):
        print(f"   📡 检测到简单出版社({final_url.split('/')[2]})，直接解析...")
        return parse_html_directly(response.text, final_url)

    # 未知出版社，尝试Jina Reader，然后直接解析
    else:
        print(f"   📡 未知出版社({final_url.split('/')[2]})，先尝试Jina Reader...")
        for attempt in range(retries + 1):
            try:
                jina_url = f"https://r.jina.ai/{final_url}"
                jina_response = requests.get(jina_url, headers={"User-Agent": headers["User-Agent"]}, timeout=30)
                if jina_response.status_code == 200 and len(jina_response.text) > 500:
                    print(f"   ✅ Jina Reader成功获取 {len(jina_response.text)} 字符")
                    return jina_response.text
                time.sleep(2 ** attempt)
            except:
                pass
        
        print(f"   📡 Jina Reader失败，尝试直接解析...")
        return parse_html_directly(response.text, final_url)

# ============================================================
# 5. 从文本中提取GitHub链接
# ============================================================
def extract_github_links(text):
    """从文本中提取GitHub链接，并过滤出生物信息学相关的"""
    if not text:
        return []
    
    # 1. 先提取所有GitHub链接
    patterns = [
        r'(https?://github\.com/[^\s\)<>]+)',
        r'github\.com/([^\s\)<>]+)'
    ]
    
    raw_links = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.startswith('http'):
                raw_links.append(match)
            else:
                raw_links.append(f"https://github.com/{match}")
    
    # 去重
    raw_links = list(set(raw_links))
    
    # 2. 过滤：只保留生物信息学/蛋白质工程相关的仓库
    bio_keywords = [
        # 蛋白质相关
        'protein', 'peptide', 'enzyme', 'fold', 'structure', 'design',
        'alphafold', 'rosetta', 'mpnn', 'esm', 'proteinmpnn',
        # 序列/基因组
        'genome', 'sequence', 'alignment', 'blast', 'homology',
        # 生物信息学通用
        'bio', 'bioinfo', 'bioinformatics', 'compbio', 'computational',
        # 工具类型
        'tool', 'server', 'pipeline', 'workflow', 'package', 'library',
        # 特定方法
        'docking', 'simulation', 'molecular', 'dynamics', 'md',
        'machine learning', 'deep learning', 'neural', 'ai',
        # 数据格式
        'pdb', 'cif', 'mmcif', 'fasta', 'sam', 'bam', 'vcf'
    ]
    
    filtered_links = []
    for link in raw_links:
        # 把链接转小写，方便匹配
        link_lower = link.lower()
        
        # 检查链接里是否包含任何关键词
        is_relevant = any(kw in link_lower for kw in bio_keywords)
        
        # 额外过滤掉明显无关的（如前端库、通用工具）
        irrelevant = ['normalize.css', 'jquery', 'bootstrap', 'react', 'angular', 'vue', 'd3', 'three.js']
        is_irrelevant = any(ir in link_lower for ir in irrelevant)
        
        if is_relevant and not is_irrelevant:
            filtered_links.append(link)
    
    return filtered_links

# ============================================================
# 6. 主程序
# ============================================================
def run_html_agent():
    query = "bioinformatics tools"
    print(f"🔍 搜索: {query}")
    
    papers = search_papers(query, max_results=10)
    print(f"📄 找到 {len(papers)} 篇论文")
    
    seen = load_seen_papers()
    print(f"📌 历史已处理: {len(seen)} 篇")
    
    results = []
    new_count = 0
    skipped_count = 0
    
    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('pmid') or paper.get('doi')
        
        # 跳过已处理的
        if paper_id and paper_id in seen:
            print(f"\n📖 [{i}/{len(papers)}] ⏭️ 跳过: {paper['title'][:50]}...")
            skipped_count += 1
            continue
        
        print(f"\n📖 [{i}/{len(papers)}] {paper['title'][:70]}...")
        
        # --- 获取全文 ---
        html_content = None
        if paper.get('doi'):
            print(f"   🔗 DOI: {paper['doi']}")
            html_content = fetch_html_from_doi(paper['doi'])
        
        if not html_content:
            print("   ⚠️ 无法获取HTML全文，尝试用摘要")
            html_content = paper.get('abstract', '')
        
        if not html_content or len(html_content) < 50:
            print("   ❌ 没有可用内容")
            # 即使没内容也标记为已处理，避免反复尝试
            if paper_id:
                mark_paper_as_seen(paper_id)
            continue
        
        print(f"   📝 获取了 {len(html_content)} 字符")
        
        # --- 提取GitHub链接 ---
        github_links = extract_github_links(html_content)
        
        if github_links:
            print(f"   🔗 发现GitHub链接:")
            for link in github_links[:3]:
                print(f"      {link}")
            results.append({
                "title": paper['title'][:100],
                "doi": paper['doi'],
                "github_links": github_links,
                "url": paper['url']
            })
            new_count += 1
        else:
            print("   ⏭️ 未发现GitHub链接")
        
        # 标记为已处理
        if paper_id:
            mark_paper_as_seen(paper_id)
        
        time.sleep(1)
    
    # 保存结果
    with open("github_from_html.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 本次: {len(results)} 个新工具 | 跳过 {skipped_count} 篇 | 累计 {len(load_seen_papers())} 篇")
    print(f"📁 保存到 github_from_html.json")

# ============================================================
# 7. 运行
# ============================================================
if __name__ == "__main__":
    run_html_agent()