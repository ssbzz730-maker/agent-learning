"""第5课CLI：提问、恢复人工确认和读取SQLite Checkpoint。"""

import argparse
import json
import sqlite3
from pathlib import Path

from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_04_langchain_basics.chains import create_deepseek_model
from lesson_04_langchain_basics.tools import (
    build_rag_search_tool,
    calculator_tool,
)
from lesson_05_langgraph_agent.graph_agent import LangGraphAgent
from lesson_05_langgraph_agent.tools import build_create_ticket_tool


BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_CHECKPOINT_DB = BASE_DIR / "runtime" / "checkpoints.sqlite3"
DEFAULT_TICKET_DIR = BASE_DIR / "runtime" / "tickets"
DEFAULT_DATABASE_DIR = PROJECT_DIR / "lesson_03_rag_tool" / "rag_database"
DEFAULT_MODEL_CACHE_DIR = PROJECT_DIR / "lesson_03_rag_tool" / ".model-cache"


class StateOnlyModel:
    """status命令只需编译图和读状态，不需要API Key或真实模型。"""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("status命令不应调用模型")


def build_tools(args):
    """创建计算器、工单和可选RAG工具。"""

    tools = [calculator_tool, build_create_ticket_tool(TicketStore(args.ticket_dir))]
    if not args.with_rag:
        return tools

    from lesson_03_rag_tool.agent_app.tools.rag_tool import (
        ChromaParentChildBackend,
        CrossEncoderReranker,
        HybridRAGTool,
    )

    backend = ChromaParentChildBackend(
        database_dir=args.database_dir,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    )
    reranker = (
        CrossEncoderReranker(local_files_only=args.local_files_only)
        if args.rerank
        else None
    )
    tools.append(build_rag_search_tool(HybridRAGTool(backend, reranker=reranker)))
    return tools


def show_result(result):
    """输出图运行后的完成或等待确认状态。"""

    print(f"线程：{result.thread_id}")
    print(f"状态：{result.status}")
    print(f"本轮模型调用：{result.model_calls}")
    if result.answer:
        print(f"答案：{result.answer}")
    if result.pending_action:
        print("待确认操作：")
        print(json.dumps(result.pending_action, ensure_ascii=False, indent=2))


def show_checkpoint(agent, thread_id):
    """读取Checkpoint并以便于学习的格式显示消息。"""

    snapshot = agent.get_state(thread_id)
    if not snapshot.values:
        raise ValueError(f"线程不存在：{thread_id}")
    messages = []
    for message in snapshot.values.get("messages", []):
        item = {"type": message.type, "content": message.content}
        if getattr(message, "tool_calls", None):
            item["tool_calls"] = message.tool_calls
        if getattr(message, "tool_call_id", None):
            item["tool_call_id"] = message.tool_call_id
        messages.append(item)
    output = {
        "thread_id": thread_id,
        "next_nodes": list(snapshot.next),
        "model_calls": snapshot.values.get("model_calls", 0),
        "messages": messages,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


def build_parser():
    parser = argparse.ArgumentParser(description="LangGraph状态与人工确认Agent")
    parser.add_argument("--checkpoint-db", type=Path, default=DEFAULT_CHECKPOINT_DB)
    parser.add_argument("--ticket-dir", type=Path, default=DEFAULT_TICKET_DIR)
    parser.add_argument("--with-rag", action="store_true")
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="向指定线程提问")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--thread-id", default="lesson-5-demo")
    ask_parser.add_argument("--max-steps", type=int, default=6)

    resume_parser = subparsers.add_parser("resume", help="恢复待确认的线程")
    resume_parser.add_argument("--thread-id", default="lesson-5-demo")
    decision = resume_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")

    status_parser = subparsers.add_parser("status", help="读取最新Checkpoint")
    status_parser.add_argument("--thread-id", default="lesson-5-demo")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    connection = None
    try:
        connection = sqlite3.connect(args.checkpoint_db, check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver

        saver = SqliteSaver(connection)
        saver.setup()
        model = (
            StateOnlyModel()
            if args.command == "status"
            else create_deepseek_model()
        )
        agent = LangGraphAgent(model, build_tools(args), saver)
        if args.command == "ask":
            show_result(agent.ask(args.question, args.thread_id, args.max_steps))
        elif args.command == "resume":
            show_result(agent.resume(args.thread_id, approved=args.approve))
        else:
            show_checkpoint(agent, args.thread_id)
    except (RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
