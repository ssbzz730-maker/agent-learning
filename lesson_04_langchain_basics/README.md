# 第4课：LangChain 基础与手动工具 Agent

> 当前状态：Prompt、Runnable、Tool、RAG适配、手动工具循环和离线测试已完成。

## 学习目标

- 理解 LangChain 对原生模型调用进行了哪些标准化。
- 明确 `prompt | model | parser` 每一步的输入和输出。
- 使用 `invoke()`、`batch()` 和 `stream()`。
- 使用 `@tool` 和 `StructuredTool` 生成模型可见 Schema。
- 把第3课 RAG 对象适配成 LangChain Tool。
- 使用 `bind_tools()`、`AIMessage.tool_calls` 和 `ToolMessage` 走完工具调用流程。
- 在不依赖 AgentExecutor 的情况下理解多步工具循环和 `max_steps`。
- 继续保留参数校验和安全边界，不迷信框架。

## 与原生项目的对应关系

| 原生实现 | LangChain抽象 |
|---|---|
| 手动构造消息字典 | `ChatPromptTemplate` |
| OpenAI兼容客户端 | `ChatOpenAI` |
| 依次调用多个函数 | Runnable管道 |
| 提取模型文本 | `StrOutputParser` |
| 手写工具Schema | `@tool` / `StructuredTool` |
| 工具参数校验 | Pydantic `args_schema` + 真实函数校验 |
| 手动解析工具调用字典 | `AIMessage.tool_calls` |
| 手动构造tool角色消息 | `ToolMessage` |

LangChain减少样板代码，但不会自动保证工具安全、证据可靠或写操作获得授权。

## Runnable 数据流

项目中的基础 Chain：

```python
chain = (
    RunnableLambda(normalize_input)
    | prompt
    | model
    | StrOutputParser()
)
```

实际类型变化：

```text
{"question": " 什么是RAG？ "}       dict
  → normalize_input
{"question": "什么是RAG？"}         dict
  → ChatPromptTemplate
ChatPromptValue / 消息列表
  → ChatOpenAI或假模型
AIMessage
  → StrOutputParser
"RAG是……"                           str
```

`A | B` 表示把 A 的输出作为 B 的输入，并不是跳过中间类型。

## 三种调用方式

单次：

```python
chain.invoke({"question": "什么是RAG？"})
```

批量：

```python
chain.batch([
    {"question": "问题A"},
    {"question": "问题B"},
])
```

流式：

```python
for chunk in chain.stream({"question": "什么是Agent？"}):
    print(chunk, end="", flush=True)
```

统一的 Runnable 接口让 Prompt、模型、解析器和普通 Python 函数可以用相同方式组合。

## Tool 的两层校验

计算器使用：

```python
@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> dict:
    return {"success": True, "result": calculate(expression)}
```

第一层是 Pydantic Schema：

```text
告诉模型expression是字符串
检查字段是否存在和类型是否基本正确
```

第二层是原来的 `calculate()`：

```text
AST白名单
只允许数字和基础运算符
拒绝任意Python代码
```

Schema不能替代真实函数的安全校验。

## RAG Tool 适配

`build_rag_search_tool(rag_engine)` 不修改第3课检索算法，只把它包装成统一 Tool：

```text
LangChain Tool输入
  → Pydantic RAGSearchInput
  → rag_engine.search()
  → 第3课结构化证据
  → Tool输出
```

RAG仍然只返回证据，不在工具内部生成最终答案。

## 手动工具 Agent

`manual_agent.py` 没有隐藏控制循环，而是把每一步直接写出来：

```text
Prompt生成system/history/user消息
  → 绑定工具后的模型
  → AIMessage
      ├─ 没有tool_calls：返回最终答案
      └─ 存在tool_calls
           → 校验工具名、参数和call_id
           → 执行真实LangChain Tool
           → 构造对应的ToolMessage
           → 把完整消息再次发送给模型
```

关键代码关系：

```python
bound_model = model.bind_tools(tools)       # 只让模型看到工具Schema
message = bound_model.invoke(messages)      # 模型决定是否请求工具
result = tool.invoke(call["args"])          # Python程序执行真实工具
messages.append(ToolMessage(...))           # 把结果对应到tool_call_id
```

这四步不能混为一谈：定义 Tool、绑定 Tool、模型选择 Tool、程序执行 Tool
是四件不同的事。`max_steps` 限制的是模型调用次数，防止模型持续请求工具。

第4课 Agent 只保存本次 `ask()` 的内存消息，不提供跨进程状态、人工确认或
Checkpoint；这些能力在第5课使用 LangGraph 实现。

## 为什么本课不使用AgentExecutor

本课故意使用普通 `for` 循环实现工具 Agent，使工具调用的数据变化完全可见。
下一课将使用 LangGraph 把同一个流程改写为：

```text
模型节点
工具节点
条件边
持久化Checkpoint
人工确认中断
```

对比手动循环和状态图后，才能理解 LangGraph 解决了什么问题。这里不使用
隐藏内部流转的旧式 Agent 封装。

## 目录结构

```text
lesson_04_langchain_basics/
├── prompts.py                    # Prompt模板和消息格式化
├── chains.py                     # Runnable管道和DeepSeek模型工厂
├── tools.py                      # calculator与RAG Tool
├── manual_agent.py               # bind_tools与显式工具控制循环
├── main.py                       # chain/batch/stream/agent命令行入口
└── tests/test_langchain_basics.py
```

## 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-langchain.txt
```

## 离线测试

假模型不会调用 DeepSeek：

```powershell
.\.venv\Scripts\python.exe `
  -m unittest discover `
  -s lesson_04_langchain_basics\tests `
  -p "test_*.py" -v
```

显示计算器 Schema，同样不调用模型：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  --show-tool-schema
```

## 真实模型运行

下面的命令都会调用 DeepSeek 并产生 API 用量。

基础 Chain：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  --mode chain `
  "用一句话解释Runnable"
```

流式输出：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  --mode stream `
  "解释Tool Schema"
```

批量调用（每个带引号的参数是一个问题）：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  --mode batch `
  "什么是Runnable？" `
  "什么是Tool？"
```

计算器 Tool Agent：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  --mode agent `
  "计算125乘0.8"
```

RAG与计算器组合（需要第3课索引和 `requirements-rag.txt`）：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  --mode agent `
  --with-rag `
  --local-files-only `
  "根据上海住宿标准计算三晚最多报销多少"
```

## 本课需要能够解释

- `ChatPromptTemplate.invoke()` 为什么不是直接返回字符串。
- `ChatOpenAI.invoke()` 为什么返回 `AIMessage`。
- `StrOutputParser` 在管道中的作用。
- `invoke()`、`batch()`、`stream()` 的差别。
- `@tool` 根据哪些信息生成 Schema。
- 为什么有了 `args_schema` 仍然不能使用 `eval()`。
- 如何在不改检索算法的情况下把 RAG 适配成 Tool。
- `bind_tools()` 为什么不会自动执行工具。
- 为什么必须把带相同 `tool_call_id` 的 `ToolMessage` 返回模型。
- 手动 Agent 循环与下一课 LangGraph 状态图的对应关系。
