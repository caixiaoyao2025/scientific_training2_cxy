# 生物信息学工具发现 Agent 项目讲稿

## 一、项目背景

大家好，我们小组的项目是"生物信息学工具发现 Agent"。

生物信息学领域的工具数量庞大且更新极快，每天都有新的分析工具发表。研究者面临两个痛点：

1. **工具发现困难** — 论文里提到的工具分散在 PubMed、GitHub 等不同平台，靠人工检索效率低
2. **工具集成繁琐** — 每个工具有不同的命令行参数、不同的输入输出格式、不同的调用方式，接入复杂流程时需要写大量胶水代码

我们的目标是：**让 AI Agent 自动发现工具、标准化工具接口、并直接使用工具完成分析任务**，把研究者从繁琐的工具查找和集成工作中解放出来。

---

## 二、核心设计思路：ToolSpec

我们的核心设计理念来自一个关键观察：

**LLM 从论文/GitHub 抽取工具时，不应只抽取 `command`（命令），而应该抽取：**

```
execution type         执行类型
+
execution specification 执行规范
```

具体来说，一个工具应该被描述成：

```
ToolSpec
    |
    ↓
Execution Engine        执行引擎
    |
    +--- CLI Runner      命令行执行器
    |
    +--- Python Runner    Python 脚本执行器
    |
    +--- API Runner       API 调用执行器
    |
    +--- Docker Runner    Docker 容器执行器
```

这就是为什么像 MCP 这类系统强调 **tool interface standardization（工具接口标准化）**：

> Agent 不需要知道工具内部是什么，只需要知道统一的输入、输出和调用方式。

在我们的项目中，这个理念已经落地为 `registry.yaml` 中的工具规范格式：

```yaml
- name: "fastp_qc"
  type: "cli"                          # ← execution type
  command: "fastp -i {fq_path} -o {filtered_fq_path} ..."   # ← execution spec
  inputs:                              # ← 统一输入接口
    fq_path: {type: "string", ...}
  expected_outputs:                    # ← 统一输出接口
    - name: "json_report_path"
      render_as: "text"
```

**接口统一带来的好处**：Agent 只要看 `inputs` 和 `expected_outputs` 就知道怎么调用工具，不需要理解 fastp、samtools、Picard 这些工具的内部实现差异。

---

## 三、系统架构

我们的系统由三个部分组成：

### 1. MCP Server（工具注册中心）

- 基于 FastMCP 框架实现
- 通过 `registry.yaml` 声明式注册工具，目前内置 7 个生物信息学工具
- 工具类型涵盖 `cli`（fastp、samtools、bedtools）、`script`（Python 脚本）、`java`（Picard）等

| 工具 | 类型 | 用途 |
|------|------|------|
| samtools_flagstat | CLI | 比对统计 |
| bedtools_intersect | CLI | 区间重叠分析 |
| blastn_tabular | CLI | 序列比对搜索 |
| fastp_qc | CLI | 测序数据质控 |
| render_qc_png | Script | 质控可视化 |
| extract_pdf_summary | Script | PDF 摘要提取 |
| picard_collect_alignment_summary | Java | 比对质量指标 |

### 2. 工具发现 Pipeline（自动扩充工具库）

每天自动运行的发现 Agent：

```
PubMed 搜索论文
    ↓
HTML 解析（多策略：直接解析 / Jina Reader / Playwright）
    ↓
提取 GitHub 链接
    ↓
标准化为 ToolSpec（execution type + specification）
    ↓
合并到 MCP Server 注册表
```

- 支持本地 `run_discover.bat`（Windows 任务计划程序）和云端 GitHub Actions 每日定时运行
- 论文解析采用多策略降级机制，应对不同出版社的反爬虫措施

### 3. Biomni Agent（下游消费方）

- 基于 Biomni A1（LangGraph 状态机框架）
- 通过 MCP 协议连接工具注册中心
- 用户用自然语言提出分析需求，Agent 自主规划步骤、调用工具、汇总结果
- 提供 Gradio 双面板 GUI：左侧展示执行进度和最终结果，右侧展示详细执行日志

---

## 四、关键技术点

### 1. LLM 与执行框架的协议对齐

Biomni 要求模型输出 `<execute>` 标签来执行代码，但 Qwen 等模型习惯输出 markdown 代码块。我们实现了 **LLMProxy**，在 LLM 响应层拦截并自动转换：

```
Qwen 输出 ```bash fastp ...```
    ↓ LLMProxy 自动包装
<execute>#!bash fastp ...</execute>
```

### 2. 空输出导致的循环问题

wget、fastp 等工具的输出走 stderr 而非 stdout，导致执行引擎捕获不到结果，模型陷入"解析错误"无限循环。我们 monkey-patch 了 `run_bash_script`，强制合并 stdout + stderr，任何命令都返回非空观察结果。

### 3. 限流保护

SiliconFlow API 有 TPM 速率限制，我们实现了指数退避重试机制，遇到 429 错误自动等待后重试，避免任务中断。

### 4. 执行环境

- Google Colab 上使用 Python 3.11 venv 隔离环境
- apt 安装 fastp、samtools、bedtools 等核心工具
- 通过 Gradio share 链接分享给用户访问

---

## 五、Demo 演示

启动 notebook 后，用户只需在聊天框输入自然语言提示词，例如：

> "Download these files from GitHub and run the analysis:
> 1. Download sample.fastq and run fastp on it
> 2. Download sample.sam and run samtools flagstat on it
> 3. Download sample.bed and sample_annotation.gff, then use bedtools intersect
> Summarize all results."

Agent 会自动完成：

1. **fastp 质控**：50 reads → 37 通过（74%），Q20 提升至 77.5%，Q30 提升至 62.7%
2. **samtools flagstat**：30 条 reads，100% 比对成功
3. **bedtools intersect**：发现 20 个基因区间重叠（lac operon、ara operon、recA、lexA 等）
4. 自动汇总所有结果

整个过程 Agent 自主规划 7 个步骤、逐步执行、实时反馈。

---

## 六、总结与展望

### 已实现

- ✅ MCP 标准化的工具注册中心（7 个生物信息学工具）
- ✅ 每日自动运行的论文工具发现 Pipeline
- ✅ Biomni 与 MCP Server 的无缝对接
- ✅ 自然语言驱动的多工具分析编排
- ✅ 稳定的执行环境与容错机制

### 未来方向

1. **丰富 Execution Engine**：当前主要支持 CLI/脚本，未来可扩展 API Runner 和 Docker Runner，覆盖更多工具类型
2. **工具质量校验**：对发现的工具增加自动测试与验证机制，确保可用性
3. **多模型适配**：当前针对 Qwen 优化，未来支持更多 LLM 后端
4. **规模化**：工具发现从每天 5 篇论文扩展到更大规模，建立更完整的生物信息学工具生态

---

**一句话总结**：我们构建了一个"会自己找工具、自己学工具、自己用工具"的生物信息学 Agent 系统，通过工具接口标准化，让 AI 真正成为研究者的得力助手。
