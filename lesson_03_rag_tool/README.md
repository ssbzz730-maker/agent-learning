# 第3课：把完整 RAG 接成 Agent 工具

> 当前状态：代码、样例文档和离线测试已完成。

## 学习目标

- 把复杂检索封装成 `search_company_documents` 只读工具。
- 理解 RAG 工具负责证据，Agent 模型负责最终生成。
- 组合向量召回、BM25、RRF、Reranker、阈值过滤和父块取回。
- 返回带来源、页码、标题和分数的结构化证据。
- 让 Agent 在一次任务中继续调用计算器或人工确认写工具。

## 完整数据流

```text
用户任务
  → Agent判断需要制度事实
  → search_company_documents
      → 向量召回
      → BM25召回
      → RRF融合
      → 可选CrossEncoder精排
      → 相似度/关键词阈值过滤
      → 子块去重并取回父块
      → 返回[E1]、来源和分数
  → Agent读取证据
  ├─ 需要算术 → calculator
  ├─ 需要写入 → create_ticket + 人工确认
  └─ 信息足够 → 带引用回答
```

## 为什么工具只返回证据

工具返回：

```python
{
    "success": True,
    "results": [
        {
            "evidence_id": "E1",
            "content": "上海住宿标准为每晚500元……",
            "source": "company_policies.md",
            "heading": "差旅报销制度",
            "scores": {...},
        }
    ]
}
```

它不在内部再次调用 DeepSeek 生成答案。这样 Agent 可以继续计算、比较或调用其他工具，也避免“RAG 生成一次、Agent 又生成一次”的重复费用和信息损失。

## 目录结构

```text
lesson_03_rag_tool/
├── agent_app/
│   ├── agent.py                    # 组合第2课控制器和RAG注册表
│   ├── main.py                     # ask/confirm/reject/status
│   └── tools/
│       ├── rag_tool.py             # 混合检索和Chroma适配器
│       └── registry.py             # Agent可见Schema和参数校验
├── rag/
│   ├── document_loaders.py         # TXT/MD/PDF/DOCX读取
│   └── build_index.py              # 父子分块和Chroma建库
├── data/documents/                 # 可替换或增加自己的知识文档
└── tests/                          # 不下载模型的内存后端测试
```

## 运行离线测试

不调用 DeepSeek，不加载 Embedding 和 Reranker：

```powershell
.\.venv\Scripts\python.exe `
  -m unittest discover `
  -s lesson_03_rag_tool\tests `
  -p "test_*.py" -v
```

测试覆盖：

- 混合检索返回结构化父块证据。
- 向量相似度较低时，精确关键词仍能召回错误码。
- 无关查询通过阈值过滤并返回空证据。
- `source` 来源过滤和不存在来源。
- Reranker 改变最终顺序。
- Markdown 多格式入口、父子分块和重叠校验。
- Agent 先检索、再计算、最后引用证据。

## 安装 RAG 运行依赖

基础 Agent 测试不需要这些重型依赖。真正建库时再安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
```

首次加载模型可能需要联网下载，之后保存在被 Git 忽略的模型缓存目录。

## 建立索引

样例知识文档位于 `lesson_03_rag_tool/data/documents/`。可以加入自己的 TXT、Markdown、PDF 或 DOCX。

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_03_rag_tool.rag.build_index
```

默认生成：

```text
lesson_03_rag_tool/rag_database/
```

数据库、模型缓存和运行状态都不会提交 GitHub。

## 运行 RAG Agent

以下 `ask` 命令会调用 DeepSeek并产生 API 用量：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_03_rag_tool.agent_app.main `
  ask "根据上海住宿标准，出差三晚最多报销多少？" `
  --session-id travel-demo
```

启用 Reranker：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_03_rag_tool.agent_app.main `
  --rerank `
  ask "取消会议室预约需要提前多久？" `
  --session-id meeting-demo
```

`--rerank` 是主命令参数，因此必须写在 `ask` 前面。

## 关键工具边界

| 组件 | 负责 | 不负责 |
|---|---|---|
| Chroma后端 | 子块向量召回、读取父块 | 最终回答 |
| BM25 | 精确词和编号召回 | 语义同义表达 |
| RRF | 融合不同排名 | 判断事实真假 |
| Reranker | 精排少量候选 | 全库召回 |
| RAG工具 | 返回可靠结构化证据 | 执行写操作 |
| Agent | 选择和组合工具、生成答案 | 绕过程序确认 |

## 空结果和工具失败不同

没有相关制度：

```python
{"success": True, "results": [], "message": "没有足够可靠的相关证据"}
```

工具本身异常：

```python
{"success": False, "error": "数据库不可用"}
```

前者表示检索正常完成但没有证据；后者表示检索流程没有正常完成。Agent 对两者都不能编造答案，但向用户解释的原因不同。

## 本课需要能够解释

- 为什么完整 RAG 适合作为 Agent 的只读工具。
- 为什么工具返回证据而不是再生成最终答案。
- 向量检索、BM25、RRF 和 Reranker 各自处于哪一层。
- 为什么小子块负责召回、父块负责提供上下文。
- 为什么 `success=True` 仍可能没有结果。
- Agent 如何在 RAG 之后继续调用计算器。
