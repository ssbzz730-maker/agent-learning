"""异步流式Agent服务、同会话互斥和生产依赖创建。"""

import asyncio
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_04_langchain_basics.chains import create_deepseek_model
from lesson_06_agent_quality.observability import EventLogger
from lesson_06_agent_quality.reliability import RetryPolicy
from lesson_07_fastapi_deployment.service import WebSettings, build_tools
from lesson_08_streaming_agent.streaming_agent import StreamingQualityAgent


class StreamingAgentService:
    """对同一thread_id串行执行图，不同会话可以并发流式运行。"""

    def __init__(self, agent, logger=None):
        self.agent = agent
        self.logger = logger
        self._locks = {}
        self._locks_guard = asyncio.Lock()

    async def _thread_lock(self, thread_id):
        async with self._locks_guard:
            return self._locks.setdefault(thread_id, asyncio.Lock())

    async def stream_chat(self, question, thread_id, max_steps=6):
        """锁定当前会话，并转发新问题产生的全部公开事件。"""

        lock = await self._thread_lock(thread_id)
        async with lock:
            async for event in self.agent.astream_ask(
                question,
                thread_id,
                max_steps,
            ):
                yield event

    async def stream_resume(self, thread_id, approved):
        """锁定当前会话，并流式恢复人工确认中断。"""

        lock = await self._thread_lock(thread_id)
        async with lock:
            async for event in self.agent.astream_resume(thread_id, approved):
                yield event

    async def get_session(self, thread_id):
        """异步读取Checkpoint，不调用模型。"""

        snapshot = await self.agent.graph.aget_state(self.agent._config(thread_id))
        if not snapshot.values:
            raise KeyError(thread_id)
        messages = []
        for message in snapshot.values.get("messages", []):
            item = {"type": message.type, "content": message.content}
            if getattr(message, "tool_calls", None):
                item["tool_calls"] = message.tool_calls
            if getattr(message, "tool_call_id", None):
                item["tool_call_id"] = message.tool_call_id
            messages.append(item)
        return {
            "thread_id": thread_id,
            "next_nodes": list(snapshot.next),
            "model_calls": snapshot.values.get("model_calls", 0),
            "messages": messages,
        }


@asynccontextmanager
async def default_streaming_service(settings=None):
    """创建真实流式模型和异步SQLite Saver，并在退出时安全关闭。"""

    settings = settings or WebSettings.from_env()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    database = settings.runtime_dir / "checkpoints.sqlite3"
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        logger = EventLogger(settings.runtime_dir / "events.jsonl")
        ticket_store = TicketStore(settings.runtime_dir / "tickets")
        model = create_deepseek_model(
            timeout=settings.model_timeout,
            max_retries=0,
            streaming=True,
        )
        agent = StreamingQualityAgent(
            model,
            build_tools(settings, ticket_store),
            logger,
            saver,
            retry_policy=RetryPolicy(max_attempts=settings.max_attempts),
        )
        yield StreamingAgentService(agent, logger)
