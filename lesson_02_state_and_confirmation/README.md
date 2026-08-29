# 第2课：状态与人工确认

> 当前状态：机制设计已整理，代码将在下一步实现。

## 学习目标

- 使用`session_id`隔离不同会话。
- 持久化消息、步骤、状态和待确认操作。
- 区分只读工具与写入工具。
- 高风险工具执行前暂停，用户确认后恢复。
- 用户拒绝后不执行工具，并把拒绝结果返回模型。

## 计划状态结构

```python
{
    "session_id": "demo",
    "messages": [],
    "step": 0,
    "status": "running",
    "pending_action": None,
    "tool_results": [],
}
```

状态流转：

```text
running
  ├─ 只读工具 → 执行 → running
  ├─ 高风险工具 → waiting_confirmation
  │                    ├─ 同意 → 执行 → running
  │                    └─ 拒绝 → 不执行 → running
  ├─ 得到答案 → completed
  └─ 达到限制或异常 → failed
```

## 计划新增工具

```text
create_ticket
```

它会模拟创建工单，属于写操作。模型可以建议调用，但程序必须保存为`pending_action`并等待用户明确确认，不能由模型代替用户授权。

## 计划测试

- 不同`session_id`的历史相互隔离。
- 未确认时不创建工单。
- 确认后只执行一次。
- 拒绝后不执行。
- 程序重启后能恢复等待确认的任务。
- 模型重复请求相同写操作时不会重复执行。
