"""定义工具 Schema、风险等级、参数校验和真实函数映射。"""

import json

from lesson_01_native_tool_calling.agent_app.tools.calculator import calculate
from lesson_01_native_tool_calling.agent_app.tools.knowledge_base import KnowledgeBase
from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算基础数学表达式。涉及金额、比例或算术时使用。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索公司制度知识库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
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
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "创建一张需要人工确认的内部支持工单。一次只创建一张。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "简短标题"},
                    "description": {"type": "string", "description": "问题详情"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium",
                    },
                },
                "required": ["title", "description"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRegistry:
    """统一管理只读工具和需要确认的写工具。"""

    CONFIRMATION_REQUIRED = {"create_ticket"}

    def __init__(self, knowledge_base=None, ticket_store=None):
        # 延迟加载知识库，使后续课程可以复用计算器和写工具，
        # 而不必初始化第1课的简单检索器。
        self.knowledge_base = knowledge_base
        self.ticket_store = ticket_store or TicketStore()

    def requires_confirmation(self, name):
        """返回工具是否会改变外部状态。"""

        return name in self.CONFIRMATION_REQUIRED

    def validate(self, name, raw_arguments):
        """解析模型 JSON，并进行程序侧的严格参数校验。"""

        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            raise ValueError("工具参数不是有效JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是JSON对象")

        if name == "calculator":
            if set(arguments) != {"expression"} or not isinstance(
                arguments["expression"], str
            ):
                raise ValueError("calculator 只接受字符串 expression 参数")
        elif name == "search_knowledge_base":
            if not set(arguments) <= {"query", "top_k"} or "query" not in arguments:
                raise ValueError("search_knowledge_base 参数不正确")
            if not isinstance(arguments["query"], str) or not arguments[
                "query"
            ].strip():
                raise ValueError("query 必须是非空字符串")
            top_k = arguments.get("top_k", 3)
            if (
                not isinstance(top_k, int)
                or isinstance(top_k, bool)
                or not 1 <= top_k <= 5
            ):
                raise ValueError("top_k 必须是 1 到 5 之间的整数")
        elif name == "create_ticket":
            if not set(arguments) <= {"title", "description", "priority"} or not {
                "title",
                "description",
            } <= set(arguments):
                raise ValueError("create_ticket 参数不正确")
            title = arguments["title"]
            description = arguments["description"]
            priority = arguments.get("priority", "medium")
            if not isinstance(title, str) or not 1 <= len(title.strip()) <= 100:
                raise ValueError("title 长度必须为 1 到 100")
            if not isinstance(description, str) or not 1 <= len(
                description.strip()
            ) <= 1000:
                raise ValueError("description 长度必须为 1 到 1000")
            if priority not in {"low", "medium", "high"}:
                raise ValueError("priority 必须是 low、medium 或 high")
            arguments = {
                "title": title.strip(),
                "description": description.strip(),
                "priority": priority,
            }
        else:
            raise ValueError(f"不允许调用未知工具：{name}")
        return arguments

    def execute(self, name, arguments, idempotency_key=None):
        """执行已经通过 ``validate`` 的参数。"""

        if name == "calculator":
            return {"result": calculate(arguments["expression"]), "success": True}
        if name == "search_knowledge_base":
            if self.knowledge_base is None:
                self.knowledge_base = KnowledgeBase()
            results = self.knowledge_base.search(
                arguments["query"], arguments.get("top_k", 3)
            )
            return {
                "results": results,
                "message": None if results else "知识库中没有找到相关规定",
                "success": True,
            }
        if name == "create_ticket":
            ticket = self.ticket_store.create(
                **arguments,
                idempotency_key=idempotency_key,
            )
            return {"success": True, "ticket": ticket}
        raise ValueError(f"不允许调用未知工具：{name}")
