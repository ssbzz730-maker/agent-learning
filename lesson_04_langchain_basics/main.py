"""第4课命令行入口：运行基础 Runnable Chain 或查看工具 Schema。"""

import argparse
import json

from lesson_04_langchain_basics.chains import build_answer_chain
from lesson_04_langchain_basics.tools import calculator_tool, tool_summary


def main():
    parser = argparse.ArgumentParser(description="LangChain基础示例")
    parser.add_argument("question", nargs="?", help="发送给基础Chain的问题")
    parser.add_argument(
        "--show-tool-schema",
        action="store_true",
        help="显示calculator的模型可见Schema，不调用模型",
    )
    args = parser.parse_args()
    if args.show_tool_schema:
        print(json.dumps(tool_summary(calculator_tool), ensure_ascii=False, indent=2))
        return
    if not args.question:
        parser.error("请提供问题，或使用 --show-tool-schema")
    try:
        answer = build_answer_chain().invoke({"question": args.question})
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(answer)


if __name__ == "__main__":
    main()
