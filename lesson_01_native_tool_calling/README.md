# 第1课：原生 Tool Calling

## 学习目标

- 理解“模型选择工具，Python执行工具”。
- 理解工具Schema与工具注册表的区别。
- 理解 `tool_call_id` 和 `role=tool` 消息。
- 理解Agent控制循环、终止条件和安全限制。
- 使用假模型响应进行离线测试。

## 目录

```text
lesson_01_native_tool_calling/
├── agent_app/
│   ├── agent.py                 # Agent控制循环
│   ├── config.py                # DeepSeek配置
│   ├── main.py                  # 命令行入口
│   └── tools/
│       ├── calculator.py        # AST安全计算器
│       ├── knowledge_base.py    # TF-IDF知识库
│       └── registry.py          # Schema、白名单和参数校验
├── data/knowledge_base.json     # 示例制度
└── tests/test_agent.py          # 详细注释的离线测试
```

## 推荐阅读顺序

1. `agent_app/main.py`：用户问题怎样进入Agent。
2. `agent_app/agent.py`：模型调用、工具执行和循环终止。
3. `agent_app/tools/registry.py`：工具Schema和执行白名单。
4. `agent_app/tools/calculator.py`：为什么不使用`eval`。
5. `agent_app/tools/knowledge_base.py`：知识库工具的输入和输出。
6. `tests/test_agent.py`：如何不调用API测试控制循环。

## 运行

从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m lesson_01_native_tool_calling.agent_app.main `
  "125元打八折是多少？"
```

运行本课测试：

```powershell
.\.venv\Scripts\python.exe -m unittest `
  lesson_01_native_tool_calling.tests.test_agent `
  -v
```

## 核心流程

```text
messages + TOOL_SCHEMAS
  ↓
模型返回content或tool_calls
  ├─ content → 最终答案，结束
  └─ tool_calls
       ↓
     注册表校验工具和参数
       ↓
     Python执行工具
       ↓
     结果通过tool_call_id加入messages
       ↓
     下一轮模型决策
```
