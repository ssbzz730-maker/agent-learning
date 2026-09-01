# 第6课：Agent质量、可靠性与可观测性

> 本课把第5课状态图扩展为可以追踪、重试和量化评估的Agent。

## 目录

```text
lesson_06_agent_quality/
├── observability.py        # request_id、JSONL日志、敏感字段脱敏
├── reliability.py          # 错误分类、指数退避、模型调用观测
├── quality_agent.py        # 对第5课Agent做横切增强
├── evaluation.py           # 工具、检索、答案和性能评估
├── main.py                 # ask/resume/evaluate/live-evaluate
├── data/
│   ├── eval_cases.json     # 期望行为
│   └── example_runs.json   # 仅用于验证评估器的示例结果
└── tests/
```

## request_id与thread_id

```text
thread_id：一段多轮会话，对应LangGraph Checkpoint
request_id：其中一次ask或resume，用来关联日志和Token
```

一次请求的JSONL事件可能是：

```text
request_started
model_started
model_finished
tool_started
tool_finished
model_started
model_finished
request_finished
```

每条事件都有相同的`request_id`和`thread_id`，因此能还原整条调用链。
日志不记录完整问题和工具参数，`api_key`、`authorization`、`password`、
`secret`和`token`字段会被替换为`***REDACTED***`。

## 模型超时与重试

真实`ChatOpenAI`通过`timeout`设置HTTP超时，并关闭SDK内部重试；
`ObservedRetryModel`统一记录每一次尝试。

只重试：

- `TimeoutError`、`ConnectionError`
- HTTP 408、429、500、502、503、504

参数错误和认证错误立即失败。等待时间按指数退避增长。写工具不会在
`QualityLangGraphAgent`中自动重试，避免创建两张工单；写重试还必须依赖幂等键。

## 评估分层

评估器同时计算：

- Tool precision、recall和顺序准确率
- Retrieval Hit Rate与MRR
- 答案关键词命中
- 拒答准确率
- 引用是否属于真实Evidence ID
- 平均延迟与总Token

`eval_cases.json`描述期望行为，运行记录描述实际行为。案例按`id`对齐，缺失
任何案例会报错，防止报告看似很好但实际漏跑失败案例。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s lesson_06_agent_quality\tests `
  -p "test_*.py" -v
```

## 离线验证评估器

下面使用的是标明为`example_runs.json`的示例结果，只验证指标计算，不能代表
真实模型质量：

```powershell
.\.venv\Scripts\python.exe -m lesson_06_agent_quality.main evaluate
```

报告写入被Git忽略的：

```text
lesson_06_agent_quality/runtime/evaluation-report.json
```

## 真实API调用

下面会调用DeepSeek并产生费用：

```powershell
.\.venv\Scripts\python.exe -m lesson_06_agent_quality.main ask `
  "请使用计算器计算125乘0.8，只根据工具结果回答。" `
  --thread-id lesson6-live-calculator
```

输出包含`request_id`，结构化日志位于：

```text
lesson_06_agent_quality/runtime/events.jsonl
```

## 真实批量评估

真实评估会逐条调用DeepSeek，并默认加载第3课RAG索引，因此会产生多次API调用
和费用。每个案例使用不同`thread_id`，避免历史相互污染：

```powershell
.\.venv\Scripts\python.exe -m lesson_06_agent_quality.main live-evaluate `
  --local-files-only `
  --thread-prefix baseline-v1
```

它会生成：

```text
runtime/live-runs.json
runtime/live-evaluation-report.json
runtime/events.jsonl
```

修改Prompt、检索阈值或Reranker后更换`thread-prefix`再次运行，比较两份报告，
才可以判断新版本是否真的更好。

## 本课掌握标准

- 区分单元测试和Agent评估。
- 区分`thread_id`和`request_id`。
- 说明为什么429可重试而参数错误不可重试。
- 说明写工具为什么不能盲目重试。
- 能通过JSONL把一次请求的模型和工具事件串起来。
- 能解释Tool Recall、Hit@K、MRR、拒答准确率和引用真实性。
- 知道示例运行结果不能冒充真实模型基线。
