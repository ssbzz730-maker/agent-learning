"""第6课CLI：可观测Agent真实调用与离线评估报告。"""

import argparse
import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from lesson_04_langchain_basics.chains import create_deepseek_model
from lesson_05_langgraph_agent.main import (
    DEFAULT_DATABASE_DIR,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_TICKET_DIR,
    build_tools,
)
from lesson_06_agent_quality.evaluation import (
    evaluate_dataset,
    load_records,
    run_live_dataset,
    write_report,
)
from lesson_06_agent_quality.observability import EventLogger
from lesson_06_agent_quality.quality_agent import QualityLangGraphAgent
from lesson_06_agent_quality.reliability import RetryPolicy


BASE_DIR = Path(__file__).parent
DEFAULT_CASES = BASE_DIR / "data" / "eval_cases.json"
DEFAULT_RUNS = BASE_DIR / "data" / "example_runs.json"
DEFAULT_RUNTIME = BASE_DIR / "runtime"


def build_parser():
    parser = argparse.ArgumentParser(description="Agent质量、可靠性与可观测性")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="离线计算评估指标")
    evaluate.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    evaluate.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    evaluate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUNTIME / "evaluation-report.json",
    )

    for command in ("ask", "resume", "live-evaluate"):
        current = subparsers.add_parser(command)
        current.add_argument("--thread-id", default="quality-demo")
        current.add_argument(
            "--checkpoint-db",
            type=Path,
            default=DEFAULT_RUNTIME / "checkpoints.sqlite3",
        )
        current.add_argument(
            "--log-file",
            type=Path,
            default=DEFAULT_RUNTIME / "events.jsonl",
        )
        current.add_argument("--ticket-dir", type=Path, default=DEFAULT_TICKET_DIR)
        current.add_argument("--with-rag", action="store_true")
        current.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
        current.add_argument(
            "--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR
        )
        current.add_argument("--rerank", action="store_true")
        current.add_argument("--local-files-only", action="store_true")
        current.add_argument("--model-timeout", type=float, default=30.0)
        current.add_argument("--max-attempts", type=int, default=3)
        if command == "ask":
            current.add_argument("question")
            current.add_argument("--max-steps", type=int, default=6)
        elif command == "resume":
            decision = current.add_mutually_exclusive_group(required=True)
            decision.add_argument("--approve", action="store_true")
            decision.add_argument("--reject", action="store_true")
        else:
            current.set_defaults(with_rag=True)
            current.add_argument("--cases", type=Path, default=DEFAULT_CASES)
            current.add_argument(
                "--runs-output",
                type=Path,
                default=DEFAULT_RUNTIME / "live-runs.json",
            )
            current.add_argument(
                "--report-output",
                type=Path,
                default=DEFAULT_RUNTIME / "live-evaluation-report.json",
            )
            current.add_argument("--thread-prefix", default="live-eval")
    return parser


def show_result(result, log_file):
    print(f"request_id：{result.request_id}")
    print(f"thread_id：{result.thread_id}")
    print(f"状态：{result.status}")
    if result.answer:
        print(f"答案：{result.answer}")
    if result.pending_action:
        print(json.dumps(result.pending_action, ensure_ascii=False, indent=2))
    print(f"结构化日志：{log_file}")


def run_agent(args):
    args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.checkpoint_db, check_same_thread=False)
    try:
        saver = SqliteSaver(connection)
        saver.setup()
        logger = EventLogger(args.log_file)
        # 禁用SDK内置重试，由ObservedRetryModel统一记录每次尝试。
        model = create_deepseek_model(timeout=args.model_timeout, max_retries=0)
        agent = QualityLangGraphAgent(
            model,
            build_tools(args),
            logger,
            saver,
            retry_policy=RetryPolicy(max_attempts=args.max_attempts),
        )
        if args.command == "ask":
            result = agent.ask(args.question, args.thread_id, args.max_steps)
        elif args.command == "resume":
            result = agent.resume(args.thread_id, approved=args.approve)
        else:
            cases = load_records(args.cases)
            runs = run_live_dataset(agent, cases, logger, args.thread_prefix)
            write_report(runs, args.runs_output)
            report = evaluate_dataset(cases, runs)
            write_report(report, args.report_output)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            print(f"真实运行记录：{args.runs_output}")
            print(f"评估报告：{args.report_output}")
            return
        show_result(result, args.log_file)
    finally:
        connection.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            report = evaluate_dataset(
                load_records(args.cases),
                load_records(args.runs),
            )
            path = write_report(report, args.output)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            print(f"完整报告：{path}")
            return
        run_agent(args)
    except (RuntimeError, TypeError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
