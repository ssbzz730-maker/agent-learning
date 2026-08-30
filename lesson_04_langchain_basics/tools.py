"""把安全计算器和第3课 RAG 封装成 LangChain Tool。"""

from typing import Optional

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from lesson_01_native_tool_calling.agent_app.tools.calculator import calculate


class CalculatorInput(BaseModel):
    """模型调用计算器时必须提供的结构化参数。"""

    expression: str = Field(
        description="只包含数字和基础运算符的表达式，例如 500 * 3"
    )


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> dict:
    """安全计算基础数学表达式；涉及金额、比例或算术时使用。"""

    return {"success": True, "result": calculate(expression)}


class RAGSearchInput(BaseModel):
    """完整 RAG 工具的输入 Schema。"""

    query: str = Field(description="脱离历史也能理解的完整检索问题")
    source: Optional[str] = Field(
        default=None,
        description="可选，只检索指定来源文件",
    )
    top_k: int = Field(default=3, ge=1, le=5, description="最终证据数量")


def build_rag_search_tool(rag_engine):
    """把具有 ``search`` 方法的第3课 RAG 对象适配成 Tool。"""

    def search_company_documents(
        query: str,
        source: Optional[str] = None,
        top_k: int = 3,
    ) -> dict:
        """从公司文档检索结构化证据，不在工具内部生成答案。"""

        return rag_engine.search(query=query, source=source, top_k=top_k)

    return StructuredTool.from_function(
        func=search_company_documents,
        name="search_company_documents",
        description=(
            "从公司文档检索可靠证据。回答制度、标准、流程和规定前使用。"
        ),
        args_schema=RAGSearchInput,
    )


def tool_summary(tool_object):
    """返回模型可见的工具名称、描述和 JSON Schema，便于对照学习。"""

    return {
        "name": tool_object.name,
        "description": tool_object.description,
        "args_schema": tool_object.args_schema.model_json_schema(),
    }
