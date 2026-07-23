import json
import requests
import re
import time
import random
from bs4 import BeautifulSoup
import os

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# ============================================================
# Memory management: track processed papers
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
# 1. Search papers (PubMed)
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
            "title": doc.get("title", "No title"),
            "abstract": doc.get("abstract", ""),
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
        })
    
    return papers

# ============================================================
# 2. Fetch content rendered by JavaScript using Playwright
# ============================================================
def fetch_with_playwright(url):
    """使用Playwright获取JavaScript渲染后的内容（攻克复杂出版社）"""
    if not HAS_PLAYWRIGHT:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(60000)
            page.goto(url, timeout=60000)
            page.wait_for_selector('body', timeout=10000)
            page.wait_for_timeout(3000)  # Wait for dynamic content to load
            content = page.content()
            browser.close()
            if content and len(content) > 500:
                return content
            return None
    except Exception as e:
        print(f"   ⚠️ Playwright fetch failed: {e}")
        return None

# ============================================================
# 3. Parse HTML directly (BeautifulSoup)
# ============================================================
def parse_html_directly(html, url):
    """使用BeautifulSoup直接解析HTML"""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        content = ""

        if 'mdpi.com' in url:
            print("   📡 Detected MDPI page, using specialized parsing...")
            # MDPI main content is usually in this div
            main_div = soup.find('div', id='main-content') or soup.find('div', class_='article-content') or soup.find('div', class_='html-content')
            if main_div:
                content = main_div.get_text(separator='\n', strip=True)
            # If not found, try using <article> tag
            if not content or len(content) < 200:
                article_tag = soup.find('article')
                if article_tag:
                    content = article_tag.get_text(separator='\n', strip=True)
            # If still nothing, try div containing 'content' as fallback
            if not content or len(content) < 200:
                for div in soup.find_all('div', class_=re.compile(r'content')):
                    div_text = div.get_text(separator='\n', strip=True)
                    if len(div_text) > len(content):
                        content = div_text
        
        if not content or len(content) < 200:
            main_content = soup.find('article') or soup.find('main')
            if main_content:
                content = main_content.get_text(separator='\n', strip=True)
        
        # Strategy 2: Find divs with key CSS classes
        if not content or len(content) < 200:
            for div in soup.find_all('div', class_=re.compile(r'fulltext|article|main|content|body|abstract|pdf')):
                div_text = div.get_text(separator='\n', strip=True)
                if len(div_text) > len(content):
                    content = div_text
        
        # Strategy 3: Collect all paragraphs
        if not content or len(content) < 200:
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 20])
        
        # Strategy 4: Fallback to body
        if not content or len(content) < 100:
            body = soup.find('body')
            if body:
                content = body.get_text(separator='\n', strip=True)

        if content and len(content) > 100:
            print(f"   ✅ Direct parsing succeeded, fetched {len(content)} chars")
            return content
        else:
            print(f"   ⚠️ Direct parsing content too short ({len(content) if content else 0} chars)")
            return None

    except Exception as e:
        print(f"   ⚠️ Direct parsing exception: {e}")
        return None

# ============================================================
# 4. Enhanced: fetch full paper text (integrated strategies)
# ============================================================
def fetch_html_from_doi(doi, retries=2):
    """增强版：使用Session + Playwright备选，获取论文全文"""
    if not doi:
        return None

    # Enhanced request headers to mimic real browser
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
    
    # --- Phase 1: Get final URL after redirect using Session ---
    try:
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(paper_url, timeout=30, allow_redirects=True)
        final_url = response.url
        print(f"   📡 Final URL: {final_url}")
    except Exception as e:
        print(f"   ⚠️ Request failed: {e}")
        return None

    # --- Phase 2: Identify publisher, choose strategy ---
    # List of complex publishers requiring special handling
    complex_publishers = ['elsevier', 'springer', 'tandfonline', 'wiley', 'nature', 'science', 'oup', 'oxford']
    
    if any(domain in final_url for domain in complex_publishers):
        print(f"   📡 Detected complex publisher ({final_url.split('/')[2]}), using multi-layer strategy...")
        
        # Strategy A: Try Playwright (mimic real browser)
        print(f"   🌐 Trying Playwright...")
        playwright_content = fetch_with_playwright(final_url)
        if playwright_content and len(playwright_content) > 500:
            print(f"   ✅ Playwright successfully fetched {len(playwright_content)} chars")
            return playwright_content
        else:
            print(f"   ⚠️ Playwright fetch failed or content too short")
        
        # Strategy B: Try Jina Reader
        print(f"   📡 Trying Jina Reader...")
        for attempt in range(retries + 1):
            try:
                jina_url = f"https://r.jina.ai/{final_url}"
                jina_response = requests.get(jina_url, headers={"User-Agent": headers["User-Agent"]}, timeout=30)
                if jina_response.status_code == 200 and len(jina_response.text) > 500:
                    print(f"   ✅ Jina Reader successfully fetched {len(jina_response.text)} chars")
                    return jina_response.text
                else:
                    print(f"   ⚠️ Jina Reader attempt {attempt+1}/{retries+1} failed")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"   ⚠️ Jina Reader exception: {e}")
                time.sleep(2 ** attempt)
        
        # Strategy C: Final fallback, direct parsing
        print(f"   📡 All advanced strategies failed, trying direct parsing...")
        return parse_html_directly(response.text, final_url)

    # Simple publishers (e.g. PMC, arXiv), parse directly
    elif any(domain in final_url for domain in ['ncbi.nlm.nih.gov', 'arxiv.org', 'pubmed']):
        print(f"   📡 Detected simple publisher ({final_url.split('/')[2]}), parsing directly...")
        return parse_html_directly(response.text, final_url)

    # Unknown publisher, try Jina Reader then parse directly
    else:
        print(f"   📡 Unknown publisher ({final_url.split('/')[2]}), trying Jina Reader first...")
        for attempt in range(retries + 1):
            try:
                jina_url = f"https://r.jina.ai/{final_url}"
                jina_response = requests.get(jina_url, headers={"User-Agent": headers["User-Agent"]}, timeout=30)
                if jina_response.status_code == 200 and len(jina_response.text) > 500:
                    print(f"   ✅ Jina Reader successfully fetched {len(jina_response.text)} chars")
                    return jina_response.text
                time.sleep(2 ** attempt)
            except:
                pass
        
        print(f"   📡 Jina Reader failed, trying direct parsing...")
        return parse_html_directly(response.text, final_url)

# ============================================================
# 5. Extract GitHub links from text
# ============================================================
def extract_github_links(text):
    """从文本中提取GitHub链接，并过滤出生物信息学相关的"""
    if not text:
        return []
    
    # 1. First extract all GitHub links
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
    
    # Deduplicate
    raw_links = list(set(raw_links))
    
    # 2. Filter: keep only bioinformatics/protein engineering related repos
    bio_keywords = [
        # Protein-related
        'protein', 'peptide', 'enzyme', 'fold', 'structure', 'design',
        'alphafold', 'rosetta', 'mpnn', 'esm', 'proteinmpnn',
        # Sequence/genome
        'genome', 'sequence', 'alignment', 'blast', 'homology',
        # Bioinformatics general
        'bio', 'bioinfo', 'bioinformatics', 'compbio', 'computational',
        # Tool types
        'tool', 'server', 'pipeline', 'workflow', 'package', 'library',
        # Specific methods
        'docking', 'simulation', 'molecular', 'dynamics', 'md',
        'machine learning', 'deep learning', 'neural', 'ai',
        # Data formats
        'pdb', 'cif', 'mmcif', 'fasta', 'sam', 'bam', 'vcf'
    ]
    
    filtered_links = []
    for link in raw_links:
        # Convert link to lowercase for matching
        link_lower = link.lower()
        
        # Check if link contains any keywords
        is_relevant = any(kw in link_lower for kw in bio_keywords)
        
        # Additionally filter out obviously irrelevant ones (e.g. frontend libraries, general tools)
        irrelevant = ['normalize.css', 'jquery', 'bootstrap', 'react', 'angular', 'vue', 'd3', 'three.js']
        is_irrelevant = any(ir in link_lower for ir in irrelevant)
        
        if is_relevant and not is_irrelevant:
            filtered_links.append(link)
    
    return filtered_links

# ============================================================
# 6. Main program
# ============================================================
def run_html_agent():
    query = "bioinformatics tools"
    print(f"🔍 Searching: {query}")
    
    papers = search_papers(query, max_results=10)
    print(f"📄 Found {len(papers)} papers")
    
    seen = load_seen_papers()
    print(f"📌 Previously processed: {len(seen)} papers")
    
    results = []
    new_count = 0
    skipped_count = 0
    
    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('pmid') or paper.get('doi')
        
        # Skip already processed
        if paper_id and paper_id in seen:
            print(f"\n📖 [{i}/{len(papers)}] ⏭️ Skipping: {paper['title'][:50]}...")
            skipped_count += 1
            continue
        
        print(f"\n📖 [{i}/{len(papers)}] {paper['title'][:70]}...")
        
        # --- Fetch full text ---
        html_content = None
        if paper.get('doi'):
            print(f"   🔗 DOI: {paper['doi']}")
            html_content = fetch_html_from_doi(paper['doi'])
        
        if not html_content:
            print("   ⚠️ Cannot fetch HTML full text, trying abstract")
            html_content = paper.get('abstract', '')
        
        if not html_content or len(html_content) < 50:
            print("   ❌ No usable content")
            # Mark as processed even with no content to avoid retrying
            if paper_id:
                mark_paper_as_seen(paper_id)
            continue
        
        print(f"   📝 Fetched {len(html_content)} chars")
        
        # --- Extract GitHub links ---
        github_links = extract_github_links(html_content)
        
        if github_links:
            print(f"   🔗 Found GitHub links:")
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
            print("   ⏭️ No GitHub links found")
        
        # Mark as processed
        if paper_id:
            mark_paper_as_seen(paper_id)
        
        time.sleep(1)
    
    # Save results
    with open("github_from_html.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 This run: {len(results)} new tools | Skipped {skipped_count} papers | Total {len(load_seen_papers())} papers")
    print(f"📁 Saved to github_from_html.json")

# ============================================================
# 7. Run
# ============================================================
if __name__ == "__main__":
    run_html_agent()