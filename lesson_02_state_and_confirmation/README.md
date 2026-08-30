# 第2课：状态与人工确认

> 当前状态：代码、说明和离线测试已完成。

## 学习目标

- 使用 `session_id` 隔离不同会话。
- 持久化消息、步骤、状态和待确认操作。
- 区分只读工具与写入工具。
- 高风险工具执行前暂停，用户确认后恢复。
- 用户拒绝后不执行工具，并把拒绝结果返回模型。
- 使用幂等键保证相同写操作最多执行一次。

## 目录结构

```text
lesson_02_state_and_confirmation/
├── agent_app/
│   ├── agent.py                 # 可暂停和恢复的控制循环
│   ├── state_store.py           # 按 session_id 保存 JSON 状态
│   ├── main.py                  # ask/confirm/reject/status 命令
│   └── tools/
│       ├── registry.py          # Schema、风险等级、参数校验和执行入口
│       └── ticket_store.py      # 带幂等键的模拟工单系统
└── tests/test_stateful_agent.py # 6 个离线测试
```

## 状态结构

```python
{
    "session_id": "demo",
    "messages": [],
    "step": 1,
    "status": "waiting_confirmation",
    "pending_action": {
        "action_id": "...",
        "tool_call_id": "call_123",
        "name": "create_ticket",
        "arguments": {},
    },
    "tool_results": [],
    "executed_calls": [],
    "answer": None,
    "last_error": None,
}
```

状态默认保存在本课的 `agent_states/`，模拟工单保存在 `tickets/`。这两个目录都已被 `.gitignore` 排除，不会上传运行数据。

## 状态流转

```text
running
  ├─ 只读工具 → 立即执行 → 保存结果 → running
  ├─ create_ticket → 只保存 pending_action → waiting_confirmation
  │                                        ├─ confirm → 执行一次 → running
  │                                        └─ reject  → 不执行   → running
  ├─ 得到最终答案 → completed
  └─ 达到限制或异常 → failed
```

最重要的一点：

```text
模型返回 create_ticket
        ≠
工单已经创建
```

模型只能提出操作建议。程序验证参数并保存 `pending_action` 后立即暂停；只有用户执行 `confirm`，控制器才会调用真实写入函数。

## 为什么状态必须由程序保存

模型不会自动记住上次进程发生了什么。程序重启后，新的模型请求只有重新收到保存的 `messages`、`pending_action` 和工具结果，才能继续原任务。

状态存储承担的是：

```text
记住发生过什么
  +
告诉控制器下一步允许做什么
```

## 为什么需要幂等键

假设工单已经创建，但程序在保存结果前突然退出。重启后再次确认，如果直接执行就可能创建两张工单。

项目为每个待确认操作生成稳定的 `action_id`，工单系统把它当作幂等键：

```text
第一次 action_id → 创建工单
相同 action_id 再次到达 → 返回原工单，不重复写入
```

重复调用检测保护一次 Agent 任务，幂等键保护真实写入系统。两者解决的层次不同。

## 命令行运行

以下命令可能调用 DeepSeek，会产生 API 用量。先确保当前进程能读取环境变量。

### 1. 提出创建工单任务

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_02_state_and_confirmation.agent_app.main `
  ask "三号会议室投影仪无法开机，帮我创建高优先级维修工单" `
  --session-id ticket-demo
```

模型提出 `create_ticket` 后，程序应显示：

```text
状态：waiting_confirmation
确认前不会执行
```

### 2. 查看保存的状态

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_02_state_and_confirmation.agent_app.main `
  status --session-id ticket-demo
```

### 3. 确认执行

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_02_state_and_confirmation.agent_app.main `
  confirm --session-id ticket-demo
```

如果不允许创建，则使用：

```powershell
.\.venv\Scripts\python.exe `
  -m lesson_02_state_and_confirmation.agent_app.main `
  reject --session-id ticket-demo
```

## 运行离线测试

测试不调用 DeepSeek，也不会产生 API 费用：

```powershell
.\.venv\Scripts\python.exe `
  -m unittest lesson_02_state_and_confirmation.tests.test_stateful_agent -v
```

覆盖以下场景：

1. 不同 `session_id` 的历史互不影响。
2. 未确认时不创建工单。
3. 确认后只执行一次。
4. 拒绝后不执行。
5. 程序重启后能恢复等待确认的任务。
6. 模型重复请求相同写操作时不会重复创建。

## 本课需要能够解释

- `messages`、`status` 和 `pending_action` 分别保存什么。
- 为什么模型提出工具调用不等于工具已经执行。
- 为什么确认结果必须作为 `role=tool` 消息返回模型。
- 重复调用检测和幂等键有什么区别。
- 为什么状态文件和工单运行数据不能提交到 GitHub。
