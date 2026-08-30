# 第4课：LangChain 基础

> 当前状态：Prompt、Runnable、Tool、RAG适配和离线测试已完成。

## 学习目标

- 理解 LangChain 对原生模型调用进行了哪些标准化。
- 明确 `prompt | model | parser` 每一步的输入和输出。
- 使用 `invoke()`、`batch()` 和 `stream()`。
- 使用 `@tool` 和 `StructuredTool` 生成模型可见 Schema。
- 把第3课 RAG 对象适配成 LangChain Tool。
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

## 为什么本课不使用AgentExecutor

本课只学习基础组件，不急着重新搭建完整 Agent。你已经在前几课理解了原生控制循环；下一课将使用 LangGraph 明确表达：

```text
模型节点
工具节点
条件边
持久化Checkpoint
人工确认中断
```

这比直接套一个看不见内部状态流转的旧式 Agent 封装更适合当前学习目标。

## 目录结构

```text
lesson_04_langchain_basics/
├── prompts.py                    # Prompt模板和消息格式化
├── chains.py                     # Runnable管道和DeepSeek模型工厂
├── tools.py                      # calculator与RAG Tool
├── main.py                       # 基础Chain命令行入口
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

下面的命令会调用 DeepSeek并产生 API 用量：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_04_langchain_basics.main `
  "用一句话解释Runnable"
```

## 本课需要能够解释

- `ChatPromptTemplate.invoke()` 为什么不是直接返回字符串。
- `ChatOpenAI.invoke()` 为什么返回 `AIMessage`。
- `StrOutputParser` 在管道中的作用。
- `invoke()`、`batch()`、`stream()` 的差别。
- `@tool` 根据哪些信息生成 Schema。
- 为什么有了 `args_schema` 仍然不能使用 `eval()`。
- 如何在不改检索算法的情况下把 RAG 适配成 Tool。
