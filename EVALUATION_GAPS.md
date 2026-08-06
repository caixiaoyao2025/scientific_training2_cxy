# 9 项评估 Gap 对照表

评审提出的 9 项工具质量问题，与当前 pipeline 的覆盖情况对照。

| # | 评估项 | 状态 | 实现位置 / 证据 | 说明 |
|---|--------|------|-----------------|------|
| 1 | 依赖冲突（dependencies clash） | ⚠️ 部分 | `verify_repo.py` 扫描 requirements/pyproject/environment.yml | 检测到依赖文件并据此选安装方式，但**没有解析/报告版本冲突**。安装失败时 `execute_test.py` 记录 "install failed" 作为间接信号 |
| 2 | repo 跑不起来（not runnable） | ✅ 已覆盖 | `verify_repo.py`（entry scripts + install cmd）→ `execute_test.py`（venv 安装 + 冒烟运行） | step3.5 结构性验证 + step3.6 执行级验证，跑不起来会被 `failed` 挡住不进 registry |
| 3 | 代码不完整（incomplete code） | ❌ 未做 | — | 未做编译/语法完整性检查（`python -m compileall`、import smoke） |
| 4 | 无 license | ✅ 已覆盖 | `verify_repo.py` `_find_license()` + `discovery_to_registry.py` 写 `verified_license`/`verified_license_path` | 无 license 的 repo 在 registry 里明确标记 `verified_license: false` |
| 5 | 不可复现（not reproducible） | ✅ 已覆盖 | `execute_test.py` 隔离 venv 安装 + 固定超时 + 记录 exit code/evidence | 每个 repo 用干净 venv 实测安装+运行，结果进 `tool_execution.json` 作为可复现证据 |
| 6 | 反幻觉（anti-hallucination） | ❌ 未做 | — | 未做工具名/描述与真实 repo 的交叉校验；当前靠 blobless clone 实际验证缓解（真实克隆，不是靠 LLM 生成） |
| 7 | 文档过期（stale docs） | ⚠️ 规避 | convert/verify 全部用 AST 级文件操作，不依赖 README 文字 | 不从 README 推断行为，从 pyproject/entry_points/实际安装结果取证，规避文档过期问题 |
| 8 | wrapper 出错（broken wrapper） | ✅ 已覆盖 | `execute_test.py` 对每个候选命令做真实冒烟运行（含 `--help` fallback） | wrapper 挂掉会 exit≠0，被 `failed` 挡住 |
| 9 | IO 格式（I/O format） | ✅ 已解决 | `execute_test.py` `_find_sample_input` 用 repo 内真实示例文件；无则 fallback 到内置 `sample.fasta` | 已解决，见 step3.6 |

## 剩余未做（❌）的两项

### ③ 代码不完整检查
`verify_repo.py` 可在 clone 后对 Python 包做 `python -m compileall`（纯 AST/语法编译，无副作用），
失败即 `unverified`。当前未实现，因为多数生信工具含编译期依赖（pysam 等），
编译检查易误伤。可作为可选门控。

### ⑥ 反幻觉校验
当前设计已用"真实 blobless clone + 执行测试"替代 LLM 生成，本身抗幻觉。
可补充：把 GitHub API 返回的 `description` 与论文标题做关键词重叠校验，
显著不匹配的标记 `low_confidence`，不进高优先级。未实现。
