"""第2课命令行入口：提问、确认、拒绝和查看状态。"""

import argparse
import json

from lesson_02_state_and_confirmation.agent_app.agent import AgentError, StatefulAgent
from lesson_02_state_and_confirmation.agent_app.state_store import StateStore


def print_result(result):
    """用适合学习者阅读的格式展示 Agent 当前状态。"""

    print(f"会话：{result.session_id}")
    print(f"状态：{result.status}")
    if result.answer:
        print(f"答案：{result.answer}")
    if result.pending_action:
        print("待确认操作：")
        print(json.dumps(result.pending_action, ensure_ascii=False, indent=2))
        print("确认前不会执行。请使用 confirm 或 reject 子命令。")


def main():
    """解析子命令，并把操作交给持久化 Agent。"""

    parser = argparse.ArgumentParser(description="状态与人工确认 Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="提出一个新问题")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--session-id", default="default")
    ask_parser.add_argument("--max-steps", type=int, default=5)

    for command in ("confirm", "reject", "status"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--session-id", default="default")
        if command in {"confirm", "reject"}:
            command_parser.add_argument("--max-steps", type=int, default=5)

    args = parser.parse_args()
    try:
        if args.command == "status":
            state = StateStore().load(args.session_id)
            if state is None:
                raise AgentError("会话不存在")
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return

        agent = StatefulAgent()
        if args.command == "ask":
            result = agent.ask(args.question, args.session_id, args.max_steps)
            print_result(result)
        elif args.command in {"confirm", "reject"}:
            result = agent.confirm(
                args.session_id,
                approved=args.command == "confirm",
                max_steps=args.max_steps,
            )
            print_result(result)
    except (AgentError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
