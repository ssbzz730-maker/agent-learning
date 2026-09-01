# 第7课：FastAPI、SSE网页与Docker部署

> 本课把第6课可观测LangGraph Agent变成浏览器和其他程序可调用的服务。

## 架构

```text
中文网页 / API客户端
        ↓ HTTP或SSE
FastAPI输入校验与错误映射
        ↓
AgentService（相同thread_id并发互斥）
        ↓
QualityLangGraphAgent
        ├─ DeepSeek
        ├─ calculator / create_ticket / 可选RAG
        ├─ SQLite Checkpoint
        └─ JSONL可观测日志
```

## 目录

```text
lesson_07_fastapi_deployment/
├── api.py                 # 应用工厂、REST、SSE和静态文件
├── service.py             # 生产运行时与并发边界
├── static/
│   ├── index.html         # 中文聊天与确认页面
│   ├── styles.css
│   └── app.js             # fetch、SSE解析和Checkpoint读取
└── tests/test_api.py      # 注入假服务的离线API契约测试
```

根目录还新增：

```text
requirements-web.txt
Dockerfile
docker-compose.yml
.dockerignore
```

## 为什么使用应用工厂

```python
app = create_app(service=None)
```

生产环境不传服务，由`lifespan`在启动时创建DeepSeek、SQLite和Agent；测试传入
`FakeAgentService`，不会读取API Key或调用模型。模块导入也不会立刻产生API调用。

## API

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/api/chat` | 普通完整响应 |
| POST | `/api/chat/stream` | SSE生命周期事件流 |
| POST | `/api/approvals` | 同意或拒绝中断工具 |
| GET | `/api/sessions/{thread_id}` | 读取最新Checkpoint |
| GET | `/docs` | FastAPI自动接口文档 |

请求由Pydantic限制问题长度、`thread_id`和`max_steps`。服务层为每个
`thread_id`建立锁，避免同一Checkpoint被两个请求同时更新；不同会话仍可并发。

## SSE说明

当前接口发送：

```text
started → result → done
```

浏览器无需等HTTP连接结束才知道请求已经启动。Agent在`asyncio.to_thread()`中
运行，不阻塞FastAPI事件循环。

必须准确区分：当前模型节点仍调用完整`model.invoke()`，所以这是“请求生命周期
事件流”，不是DeepSeek逐Token原生流。要实现逐Token输出，需要下一步把模型节点
改为异步流并消费LangGraph消息事件，不能仅把完整答案切成字符来冒充。

## 本地运行

安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
  lesson_07_fastapi_deployment.api:app `
  --host 127.0.0.1 `
  --port 8000 `
  --reload
```

访问：

```text
网页：http://127.0.0.1:8000
文档：http://127.0.0.1:8000/docs
健康：http://127.0.0.1:8000/health
```

启用第3课RAG：

```powershell
$env:ENABLE_RAG="true"
$env:LOCAL_FILES_ONLY="true"
```

需要已经安装`requirements-rag.txt`并建立索引。

## REST示例

```powershell
$body = @{
  question = "请使用计算器计算36乘2.5"
  thread_id = "api-demo"
  max_steps = 4
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/chat `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

确认写工具：

```powershell
$body = @{ thread_id="ticket-demo"; approved=$true } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/approvals `
  -ContentType "application/json" -Body $body
```

## Docker

当前PowerShell会话先取得系统环境变量，但不要把Key写进Compose文件：

```powershell
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
  "DEEPSEEK_API_KEY", "Machine"
)

docker compose up --build
```

运行数据保存在Docker命名卷`agent-runtime`，容器重建后Checkpoint仍然存在。
镜像使用非root用户，`.dockerignore`排除Key、缓存、数据库和运行日志。

停止：

```powershell
docker compose down
```

只有明确希望删除所有容器运行状态时才执行：

```powershell
docker compose down -v
```

## 安全边界

这是本地学习服务，尚未实现登录、租户授权、限流、CSRF保护和HTTPS，不能直接
暴露到公网。生产部署至少需要反向代理、TLS、身份认证、用户与thread_id归属校验、
请求限流和审计日志。前端只能改善交互，不能代替后端权限检查。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s lesson_07_fastapi_deployment\tests `
  -p "test_*.py" -v
```

## 掌握标准

- 区分Pydantic请求模型、API路由和AgentService。
- 解释为什么同步Agent要放到线程中，不能阻塞异步事件循环。
- 解释REST完整响应和SSE事件流的区别。
- 说明为什么相同`thread_id`并发写Checkpoint需要互斥。
- 能沿`/api/approvals`解释HTTP请求如何恢复LangGraph interrupt。
- 理解Docker镜像、容器、端口映射和命名卷的关系。
- 知道当前SSE不是逐Token模型流，也知道服务不能直接暴露公网。
