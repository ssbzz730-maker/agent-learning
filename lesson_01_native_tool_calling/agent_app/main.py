"""原生Agent命令行入口。"""

import argparse

from lesson_01_native_tool_calling.agent_app.agent import AgentError, NativeAgent


def main():
    """解析命令行参数，支持单次提问和连续交互。"""

    parser = argparse.ArgumentParser(description="原生 Tool Calling Agent")
    parser.add_argument("question", nargs="?", help="需要Agent完成的任务")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--no-trace", action="store_true", help="隐藏工具调用轨迹")
    args = parser.parse_args()

    try:
        agent = NativeAgent()
    except AgentError as error:
        parser.error(str(error))

    def run(question):
        try:
            agent.run(
                question,
                max_steps=args.max_steps,
                show_trace=not args.no_trace,
            )
        except (AgentError, ValueError) as error:
            print(f"执行失败：{error}")

    if args.question:
        run(args.question)
        return
    print("原生Agent已就绪；输入任务开始，输入 q 退出。")
    while True:
        question = input("\n你：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            run(question)


if __name__ == "__main__":
    main()
