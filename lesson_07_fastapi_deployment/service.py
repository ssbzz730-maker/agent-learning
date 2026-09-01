"""把可观测LangGraph Agent封装成线程安全的Web服务。"""

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_04_langchain_basics.chains import create_deepseek_model
from lesson_04_langchain_basics.tools import build_rag_search_tool, calculator_tool
from lesson_05_langgraph_agent.tools import build_create_ticket_tool
from lesson_06_agent_quality.observability import EventLogger
from lesson_06_agent_quality.quality_agent import QualityLangGraphAgent
from lesson_06_agent_quality.reliability import RetryPolicy


BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WebSettings:
    runtime_dir: Path = BASE_DIR / "runtime"
    enable_rag: bool = False
    local_files_only: bool = False
    enable_reranker: bool = False
    model_timeout: float = 30.0
    max_attempts: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            runtime_dir=Path(os.getenv("AGENT_RUNTIME_DIR", BASE_DIR / "runtime")),
            enable_rag=env_bool("ENABLE_RAG"),
            local_files_only=env_bool("LOCAL_FILES_ONLY"),
            enable_reranker=env_bool("ENABLE_RERANKER"),
            model_timeout=float(os.getenv("MODEL_TIMEOUT", "30")),
            max_attempts=int(os.getenv("MODEL_MAX_ATTEMPTS", "3")),
        )


class AgentService:
    """管理Agent、SQLite连接，以及相同thread_id的并发互斥。"""

    def __init__(self, agent, connection=None, logger=None):
        self.agent = agent
        self.connection = connection
        self.logger = logger
        self._locks = {}
        self._locks_guard = threading.Lock()

    def _thread_lock(self, thread_id):
        with self._locks_guard:
            return self._locks.setdefault(thread_id, threading.RLock())

    def chat(self, question, thread_id, max_steps=6):
        with self._thread_lock(thread_id):
            return self.agent.ask(question, thread_id, max_steps)

    def resume(self, thread_id, approved):
        with self._thread_lock(thread_id):
            return self.agent.resume(thread_id, approved)

    def get_session(self, thread_id):
        snapshot = self.agent.get_state(thread_id)
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

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def build_tools(settings, ticket_store):
    tools = [calculator_tool, build_create_ticket_tool(ticket_store)]
    if not settings.enable_rag:
        return tools

    from lesson_03_rag_tool.agent_app.tools.rag_tool import (
        ChromaParentChildBackend,
        CrossEncoderReranker,
        HybridRAGTool,
    )

    lesson_3 = PROJECT_DIR / "lesson_03_rag_tool"
    backend = ChromaParentChildBackend(
        database_dir=lesson_3 / "rag_database",
        model_cache_dir=lesson_3 / ".model-cache",
        local_files_only=settings.local_files_only,
    )
    reranker = (
        CrossEncoderReranker(local_files_only=settings.local_files_only)
        if settings.enable_reranker
        else None
    )
    tools.append(build_rag_search_tool(HybridRAGTool(backend, reranker=reranker)))
    return tools


def build_default_service(settings=None):
    """创建生产服务；只有应用启动时执行，不在模块导入时加载模型。"""

    settings = settings or WebSettings.from_env()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        settings.runtime_dir / "checkpoints.sqlite3",
        check_same_thread=False,
    )
    saver = SqliteSaver(connection)
    saver.setup()
    logger = EventLogger(settings.runtime_dir / "events.jsonl")
    ticket_store = TicketStore(settings.runtime_dir / "tickets")
    model = create_deepseek_model(
        timeout=settings.model_timeout,
        max_retries=0,
    )
    agent = QualityLangGraphAgent(
        model,
        build_tools(settings, ticket_store),
        logger,
        saver,
        retry_policy=RetryPolicy(max_attempts=settings.max_attempts),
    )
    return AgentService(agent, connection, logger)
