# 第8课：LangGraph真实流式Agent

本课把第7课的“生命周期SSE”升级为LangGraph真实运行事件流。网页可以在图运行期间看到模型文本、节点变化、工具调用和人工确认，而不是等完整答案生成后模拟打字。

## 学习目标

完成本课后，应能解释：

1. `graph.astream()`与`graph.invoke()`的区别。
2. `messages`和`updates`两种流模式分别提供什么。
3. 为什么Token流仍然需要最终`result`事件。
4. 为什么异步图应使用`AsyncSqliteSaver`。
5. SSE生产者、队列和HTTP消费者如何协作。
6. 浏览器断线后为什么必须取消生产任务。
7. 人工确认为什么需要用同一`thread_id`建立一条新事件流。
8. 为什么不能把LangGraph内部事件原样暴露给前端。

## 目录结构

```text
lesson_08_streaming_agent/
├── streaming_agent.py  # 运行图并映射公开事件
├── service.py          # 异步会话锁和AsyncSqliteSaver
├── api.py              # SSE队列、心跳、取消和REST兼容接口
├── static/             # 中文流式网页
└── tests/              # 图事件和HTTP协议离线测试
```

## 核心数据流

```text
浏览器fetch POST
  → FastAPI创建SSE响应
  → StreamingAgentService取得thread_id锁
  → StreamingQualityAgent调用graph.astream
  → LangGraph产生messages与updates
  → 白名单映射为公开事件
  → asyncio.Queue解耦生产与发送
  → StreamingResponse发送SSE
  → 前端TextDecoder解析并更新页面
```

## 公开事件协议

| 事件 | 含义 | 主要字段 |
|---|---|---|
| `started` | 本次ask或resume开始 | `request_id`、`thread_id` |
| `token` | 模型产生文本块 | `text`、`message_id` |
| `node_finished` | 一个图节点执行完成 | `node` |
| `tool_requested` | 模型申请工具 | `name`、`call_id` |
| `tool_finished` | 工具执行结束 | `name`、`success` |
| `approval_required` | 风险工具等待确认 | `tools`、`count` |
| `heartbeat` | 长时间无事件时保持连接 | `status` |
| `result` | 可持久化的最终业务结果 | 答案、状态、调用次数 |
| `done` | 本次事件流正常结束 | `status` |
| `error` | 流开始后的失败 | 通用错误类型和消息 |

公开的工具事件有意不包含完整工具结果。最终RAG证据或业务数据应通过经过校验的业务响应字段返回，不能直接转发框架内部事件。

## messages与updates

本课同时订阅：

```python
async for chunk in graph.astream(
    graph_input,
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    ...
```

- `messages`包含模型消息块和工具消息，可从模型消息块提取文本。
- `updates`包含节点对State的部分更新，可判断工具申请、工具完成和`__interrupt__`。

Token只是展示中的增量文本，不能代替最终业务状态。`result`事件从最新Checkpoint重建，包含`completed`或`awaiting_confirmation`等权威状态。

## 异步SQLite Checkpoint

同步`SqliteSaver`适合`graph.invoke()`；本课的`graph.astream()`使用：

```python
async with AsyncSqliteSaver.from_conn_string(path) as saver:
    await saver.setup()
```

它让Checkpoint读写与异步图兼容。`thread_id`仍然是会话索引，相同会话通过`asyncio.Lock`串行执行，不同会话可以并发。

## 断线取消

API使用独立生产任务把图事件写入队列，HTTP消费者从队列读取并发送。浏览器点击“停止”时，`AbortController`关闭连接；响应生成器进入`finally`并取消生产任务，取消信号继续传递到`graph.astream()`。

取消不等于回滚已经完成的外部写操作。因此写工具仍然需要人工确认和幂等键，不能把HTTP断线当成事务机制。

## 人工确认恢复

第一次事件流可能结束为：

```text
tool_requested → approval_required → result(awaiting_confirmation) → done
```

用户确认后，网页请求`POST /api/approvals/stream`。服务使用相同`thread_id`执行`Command(resume=True)`，并建立一条新的SSE连接继续输出工具和模型事件。

## 启动

```powershell
cd C:\Users\22729\Documents\Codex\2026-08-16\w\work\native_agent_project

$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
  "DEEPSEEK_API_KEY",
  "Machine"
)

.\.venv\Scripts\python.exe -m uvicorn `
  lesson_08_streaming_agent.api:app `
  --host 127.0.0.1 `
  --port 8000 `
  --reload
```

浏览器打开：<http://127.0.0.1:8000>

## PowerShell测试SSE

`curl.exe -N`关闭客户端输出缓冲，便于观察事件逐条到达：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"question":"请使用计算器计算36乘2.5","thread_id":"stream-demo","max_steps":6}'
```

## 运行离线测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s lesson_08_streaming_agent\tests `
  -p "test_*.py" `
  -v
```

离线测试使用脚本模型和假服务，不调用DeepSeek，也不消耗Token。

## 安全与生产边界

- 网页只使用`textContent`展示模型内容，避免把模型输出当HTML执行。
- 公开事件由白名单映射产生，不转发完整Prompt和内部状态。
- 同进程内用`asyncio.Lock`保护同一会话；多进程部署仍需分布式锁。
- 当前没有用户认证和租户授权，不能直接暴露到公网。
- 客户端取消只能阻止后续工作，不能撤销已经发生的外部副作用。
- 真正生产部署还需要限流、审计、TLS、跨实例状态存储和指标监控。

## 自测问题

1. 为什么模型Token和最终`result`需要同时存在？
2. `messages`流和`updates`流分别负责什么？
3. 为什么工具事件只公开名称和成功状态？
4. 客户端关闭连接后，服务端如果不取消任务会怎样？
5. 为什么恢复确认必须继续使用原来的`thread_id`？
