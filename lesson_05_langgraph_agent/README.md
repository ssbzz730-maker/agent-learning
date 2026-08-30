# 第5课：LangGraph状态图Agent

> 当前状态：State、Node、Edge、Checkpoint、thread_id和interrupt人工确认均已实现。

## 本课解决什么问题

第4课使用`for`循环手动决定下一步。第5课把同一运行机制表达成图：

```text
START → agent
          ├─ 无tool_calls ───────────────→ END
          ├─ 达到max_steps → limit ─────→ END
          ├─ 安全工具 → tools ──────────→ agent
          └─ 写工具 → approval（暂停）
                           ├─ 同意 → tools → agent
                           └─ 拒绝 → rejected → agent
```

LangGraph不替代模型和工具。它负责保存State，并根据Edge决定下一个Node。

## 核心文件

```text
lesson_05_langgraph_agent/
├── state.py                 # AgentState与Reducer
├── tools.py                 # 需要人工确认的create_ticket
├── graph_agent.py           # 节点、条件边、interrupt和图编译
├── main.py                  # ask/resume/status CLI
└── tests/test_langgraph_agent.py
```

## State与Reducer

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int
    max_steps: int
    approved: bool | None
    tool_events: Annotated[list[dict], operator.add]
```

- `add_messages`：节点返回新消息时追加或按消息ID更新，不覆盖全部历史。
- `operator.add`：把新工具事件追加到旧列表。
- 没有Reducer的字段：新值直接覆盖旧值。

## 图是怎样建立的

```python
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", call_tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_model, {...})
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=checkpointer)
```

`compile()`只生成可运行图，不会调用模型。`graph.invoke()`才真正执行。

## Checkpoint与thread_id

编译时传入`SqliteSaver`后，每个节点执行前后的State都会存入SQLite。
调用图时必须提供：

```python
config = {"configurable": {"thread_id": "meeting-demo"}}
```

相同`thread_id`继续原消息历史；不同ID彼此隔离。由于使用SQLite，关闭程序并
重新运行后，旧线程仍然存在。

`thread_id`不是用户问题，也不会自动发送给模型；它是Checkpointer查找状态的键。

## interrupt人工确认

`approval`节点执行：

```python
decision = interrupt({"tool_calls": calls})
```

第一次运行到这里时，LangGraph保存Checkpoint并返回待确认内容，不会继续执行
`create_ticket`。用户确认后：

```python
graph.invoke(Command(resume=True), config=同一个thread_id)
```

图从中断位置恢复。同意进入`tools`，拒绝进入`rejected`。恢复时必须使用原来的
`thread_id`，否则LangGraph找不到等待中的Checkpoint。

## 安装

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-langchain.txt
```

## 离线测试

测试使用脚本假模型，不调用DeepSeek：

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s lesson_05_langgraph_agent\tests `
  -p "test_*.py" -v
```

## 真实运行

以下`ask`与`resume`命令会调用DeepSeek并产生API用量。

安全计算器：

```powershell
.\.venv\Scripts\python.exe -m lesson_05_langgraph_agent.main `
  ask "计算125乘0.8" `
  --thread-id math-demo
```

请求创建工单，程序会暂停：

```powershell
.\.venv\Scripts\python.exe -m lesson_05_langgraph_agent.main `
  ask "显示器无法点亮，请创建高优先级工单" `
  --thread-id ticket-demo
```

确认并恢复：

```powershell
.\.venv\Scripts\python.exe -m lesson_05_langgraph_agent.main `
  resume --thread-id ticket-demo --approve
```

拒绝并恢复：

```powershell
.\.venv\Scripts\python.exe -m lesson_05_langgraph_agent.main `
  resume --thread-id ticket-demo --reject
```

只读取状态，不需要API Key，也不会调用模型：

```powershell
.\.venv\Scripts\python.exe -m lesson_05_langgraph_agent.main `
  status --thread-id ticket-demo
```

接入第3课RAG时，把`--with-rag --local-files-only`放在`ask`子命令之前。

## 本课掌握标准

- 能说明State、Node、Edge和Conditional Edge各自负责什么。
- 能解释Reducer为什么决定“覆盖”还是“追加”。
- 能沿图说明`agent → tools → agent`循环。
- 能解释Checkpoint与普通Python变量的区别。
- 能说明`thread_id`为什么能隔离会话。
- 能解释`interrupt()`为什么在确认前不会执行写工具。
- 能说明`Command(resume=...)`必须使用原thread_id。
