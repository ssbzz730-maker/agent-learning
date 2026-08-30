"""第5课写工具：创建工单，并依靠稳定幂等键避免重复写入。"""

import hashlib
import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CreateTicketInput(BaseModel):
    """模型创建内部支持工单时需要提供的参数。"""

    title: str = Field(min_length=1, max_length=100, description="简短工单标题")
    description: str = Field(min_length=1, max_length=2000, description="问题详情")
    priority: str = Field(
        default="medium",
        pattern="^(low|medium|high)$",
        description="优先级：low、medium或high",
    )


def build_create_ticket_tool(ticket_store):
    """把第2课TicketStore适配成需要人工确认的LangChain Tool。"""

    def create_ticket(title: str, description: str, priority: str = "medium"):
        """创建内部支持工单；这是写操作，执行前必须获得用户确认。"""

        normalized = json.dumps(
            {
                "title": title.strip(),
                "description": description.strip(),
                "priority": priority,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        idempotency_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return ticket_store.create(
            title=title.strip(),
            description=description.strip(),
            priority=priority,
            idempotency_key=idempotency_key,
        )

    return StructuredTool.from_function(
        func=create_ticket,
        name="create_ticket",
        description="创建内部支持工单。属于写操作，程序会在执行前要求人工确认。",
        args_schema=CreateTicketInput,
    )
