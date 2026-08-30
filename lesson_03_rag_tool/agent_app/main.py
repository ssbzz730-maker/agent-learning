"""第3课 CLI：完整 RAG Agent 的提问、确认、拒绝和状态查看。"""

import argparse
import json
from pathlib import Path

from lesson_02_state_and_confirmation.agent_app.agent import AgentError
from lesson_02_state_and_confirmation.agent_app.state_store import StateStore
from lesson_03_rag_tool.agent_app.agent import DEFAULT_STATE_DIR, RAGAgent
from lesson_03_rag_tool.agent_app.tools.rag_tool import (
    ChromaParentChildBackend,
    CrossEncoderReranker,
    HybridRAGTool,
)
from lesson_03_rag_tool.rag.build_index import (
    DEFAULT_DATABASE_DIR,
    DEFAULT_MODEL_CACHE_DIR,
)


def build_agent(args):
    """根据命令行配置加载索引、Embedding 和可选 Reranker。"""

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
    return RAGAgent(HybridRAGTool(backend, reranker=reranker))


def show_result(result):
    print(f"会话：{result.session_id}")
    print(f"状态：{result.status}")
    if result.answer:
        print(f"答案：{result.answer}")
    if result.pending_action:
        print("待确认操作：")
        print(json.dumps(result.pending_action, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="完整RAG工具 Agent")
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--rerank", action="store_true", help="启用CrossEncoder精排")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用已经下载到缓存的模型",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--session-id", default="rag-demo")
    ask_parser.add_argument("--max-steps", type=int, default=6)
    for command in ("confirm", "reject", "status"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--session-id", default="rag-demo")
        if command in {"confirm", "reject"}:
            command_parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    try:
        if args.command == "status":
            state = StateStore(DEFAULT_STATE_DIR).load(args.session_id)
            if state is None:
                raise AgentError("会话不存在")
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return
        agent = build_agent(args)
        if args.command == "ask":
            result = agent.ask(
                args.question,
                session_id=args.session_id,
                max_steps=args.max_steps,
            )
        else:
            result = agent.confirm(
                args.session_id,
                approved=args.command == "confirm",
                max_steps=args.max_steps,
            )
        show_result(result)
    except (AgentError, RuntimeError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
