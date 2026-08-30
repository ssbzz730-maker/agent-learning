"""第3课工具注册表：完整 RAG、计算器和人工确认写工具。"""

from lesson_02_state_and_confirmation.agent_app.tools.registry import (
    ToolRegistry as BaseToolRegistry,
)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_company_documents",
            "description": (
                "从公司文档中检索可靠证据。回答制度、流程、标准、规定等"
                "事实问题前必须使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "脱离对话历史也能理解的完整问题",
                    },
                    "source": {
                        "type": "string",
                        "description": "可选，只检索指定来源文件",
                    },
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
            "name": "create_ticket",
            "description": "创建内部支持工单；这是写操作，执行前需要用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
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


class RAGToolRegistry(BaseToolRegistry):
    """把完整 RAG 工具接入第2课的安全注册表。"""

    def __init__(self, rag_tool, ticket_store=None):
        super().__init__(ticket_store=ticket_store)
        self.rag_tool = rag_tool

    def validate(self, name, raw_arguments):
        """校验 RAG 参数；其他工具复用第2课规则。"""

        if name != "search_company_documents":
            return super().validate(name, raw_arguments)

        import json

        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            raise ValueError("工具参数不是有效JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是JSON对象")
        if not set(arguments) <= {"query", "source", "top_k"} or "query" not in arguments:
            raise ValueError("search_company_documents 参数不正确")
        query = arguments["query"]
        source = arguments.get("source")
        top_k = arguments.get("top_k", 3)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if source is not None and (
            not isinstance(source, str) or not source.strip()
        ):
            raise ValueError("source 必须是非空字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
            raise ValueError("top_k 必须是 1 到 5 之间的整数")
        result = {"query": query.strip(), "top_k": top_k}
        if source:
            result["source"] = source.strip()
        return result

    def execute(self, name, arguments, idempotency_key=None):
        """RAG 只返回证据；计算和写工具交给父类。"""

        if name == "search_company_documents":
            return self.rag_tool.search(**arguments)
        return super().execute(name, arguments, idempotency_key=idempotency_key)
