"""第4课命令行入口：体验 Chain、批量、流式和手动 Tool Agent。"""

import argparse
import json
from pathlib import Path

from lesson_04_langchain_basics.chains import build_answer_chain, create_deepseek_model
from lesson_04_langchain_basics.manual_agent import ManualToolAgent
from lesson_04_langchain_basics.tools import (
    build_rag_search_tool,
    calculator_tool,
    tool_summary,
)


PROJECT_DIR = Path(__file__).parents[1]
DEFAULT_DATABASE_DIR = PROJECT_DIR / "lesson_03_rag_tool" / "rag_database"
DEFAULT_MODEL_CACHE_DIR = PROJECT_DIR / "lesson_03_rag_tool" / ".model-cache"


def build_cli_tools(args):
    """创建CLI需要的工具；只有指定 --with-rag 时才加载重型RAG依赖。"""

    tools = [calculator_tool]
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


def main():
    parser = argparse.ArgumentParser(description="LangChain完整基础项目")
    parser.add_argument(
        "question",
        nargs="*",
        help="问题；batch模式可提供多个带引号的问题",
    )
    parser.add_argument(
        "--mode",
        choices=("chain", "batch", "stream", "agent"),
        default="chain",
        help="选择基础、批量、流式或手动工具Agent模式",
    )
    parser.add_argument(
        "--show-tool-schema",
        action="store_true",
        help="显示calculator的模型可见Schema，不调用模型",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--no-trace", action="store_true", help="隐藏Agent工具轨迹")
    parser.add_argument("--with-rag", action="store_true", help="Agent同时绑定第3课RAG")
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--rerank", action="store_true", help="RAG启用CrossEncoder精排")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.show_tool_schema:
        print(json.dumps(tool_summary(calculator_tool), ensure_ascii=False, indent=2))
        return
    if not args.question:
        parser.error("请提供问题，或使用 --show-tool-schema")
    try:
        if args.mode == "batch":
            answers = build_answer_chain().batch(
                [{"question": question} for question in args.question]
            )
            for number, answer in enumerate(answers, start=1):
                print(f"[{number}] {answer}")
            return

        question = " ".join(args.question)
        if args.mode == "stream":
            for chunk in build_answer_chain().stream({"question": question}):
                print(chunk, end="", flush=True)
            print()
            return
        if args.mode == "agent":
            agent = ManualToolAgent(create_deepseek_model(), build_cli_tools(args))
            result = agent.ask(
                question,
                max_steps=args.max_steps,
                show_trace=not args.no_trace,
            )
            print(result.answer)
            return

        answer = build_answer_chain().invoke({"question": question})
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(answer)


if __name__ == "__main__":
    main()
