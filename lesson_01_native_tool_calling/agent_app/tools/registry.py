"""定义模型可见的工具Schema，并把工具名称映射到真实函数。"""

import json

from lesson_01_native_tool_calling.agent_app.tools.calculator import calculate
from lesson_01_native_tool_calling.agent_app.tools.knowledge_base import KnowledgeBase


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算基础数学表达式。涉及金额、比例或算术时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "只包含数字和基础运算符的表达式，例如 500 * 3",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索公司制度知识库。询问会议室、报销、休假或设备制度时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "完整、独立的检索问题"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRegistry:
    """校验模型生成的参数并执行对应工具。"""

    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base or KnowledgeBase()

    def execute(self, name, raw_arguments):
        """执行指定工具，并返回可安全序列化成JSON的结果。"""

        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            raise ValueError("工具参数不是有效JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是JSON对象")

        if name == "calculator":
            if set(arguments) != {"expression"}:
                raise ValueError("calculator 只接受 expression 参数")
            return {"result": calculate(arguments["expression"])}
        if name == "search_knowledge_base":
            if not set(arguments) <= {"query", "top_k"} or "query" not in arguments:
                raise ValueError("search_knowledge_base 参数不正确")
            results = self.knowledge_base.search(
                arguments["query"], arguments.get("top_k", 3)
            )
            return {
                "results": results,
                "message": None if results else "知识库中没有找到相关规定",
            }
        raise ValueError(f"不允许调用未知工具：{name}")
