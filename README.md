# Agent Learning：从原生 Tool Calling 到可控 Agent

这是一个按课程推进的 Agent 学习仓库。目标不是背框架 API，而是理解模型、工具、状态、人工确认、RAG 和控制器之间的运行机制，并能借助 AI 开发、审查和调试 Agent 项目。

## 课程进度

| 课程 | 内容 | 状态 |
|---|---|---|
| [第1课](lesson_01_native_tool_calling/) | 原生 Tool Calling、安全计算器、知识库工具、控制循环、离线测试 | 已完成 |
| [第2课](lesson_02_state_and_confirmation/) | 会话状态、暂停与恢复、高风险工具人工确认、幂等写入 | 已完成 |
| [第3课](lesson_03_rag_tool/) | 混合检索、RRF、Reranker、父块证据接入 Agent | 已完成 |
| [第4课](lesson_04_langchain_basics/) | LangChain：Prompt、Runnable、Tool、RAG适配 | 已完成 |
| 第5课 | LangGraph：StateGraph、条件边、Checkpoint | 待学习 |
| 第6课 | Agent 评估、日志、超时、重试和可观测性 | 待学习 |
| 第7课 | FastAPI、流式网页和 Docker 部署 | 待学习 |

## 仓库结构

```text
native_agent_project/
├── lesson_01_native_tool_calling/      # 每课代码、数据、测试和说明
├── lesson_02_state_and_confirmation/
├── lesson_03_rag_tool/
├── lesson_04_langchain_basics/
├── docs/
│   ├── rag_core_concepts.md            # RAG核心概念复习
│   ├── agent_core_concepts.md          # Agent核心运行机制
│   ├── questions_and_answers.md        # 学习过程中问过的问题
│   ├── learning_journal.md              # 跨设备学习路线与故障记录
│   └── interview_checklist.md          # 实习面试自检
├── .github/workflows/tests.yml         # GitHub自动测试
├── .env.example                        # 环境变量示例，不含真实Key
└── requirements.txt
```

跨设备复习可以从 [RAG 与 Agent 学习日志](docs/learning_journal.md) 开始，再按链接进入各专题文档。

## 创建环境

```powershell
cd C:\Users\22729\Documents\Codex\2026-08-16\w\work\native_agent_project

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

支持 Python 3.11 和 3.12。

## 运行全部已完成课程的测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
```

当前测试全部为离线测试，不调用 DeepSeek。

## 运行第1课 Agent

```powershell
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
  "DEEPSEEK_API_KEY",
  "Machine"
)

.\.venv\Scripts\python.exe -m lesson_01_native_tool_calling.agent_app.main `
  "根据上海住宿标准，计算出差三晚最多能报销多少钱？"
```

预期决策轨迹：

```text
search_knowledge_base
  ↓ 返回每晚500元
calculator
  ↓ 计算500×3
最终答案
  ↓ 三晚最多报销1500元
```

## 安全原则

- 模型只提出工具调用，Python控制器决定是否执行。
- 工具注册表是白名单，未知工具一律拒绝。
- 工具参数必须由程序再次校验，不能只相信Schema。
- 不使用 `eval` 执行用户或模型生成的字符串。
- 使用最大步骤数和重复调用检测防止死循环。
- 写入、发送、删除等高风险操作必须等待用户确认。
- `.env`、API Key、虚拟环境和运行数据不会提交Git。

## 学习方法

按一条任务的数据流阅读代码：

```text
用户问题
  ↓ main.py
NativeAgent.run()
  ↓ 模型返回tool_calls
ToolRegistry.execute()
  ↓ 工具结果作为role=tool消息
模型继续决策
  ↓
最终答案或达到安全限制
```

代码可以由 AI 辅助，但学习者需要能够解释输入、输出、状态变化、安全边界和失败位置。
