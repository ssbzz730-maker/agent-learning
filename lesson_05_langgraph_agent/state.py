"""定义LangGraph所有节点共享的状态结构。"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """消息用add_messages合并；其余字段由节点直接覆盖。"""

    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int
    max_steps: int
    approved: bool | None
    tool_events: Annotated[list[dict], operator.add]
