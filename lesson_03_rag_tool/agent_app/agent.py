"""把第2课可暂停 Agent 与第3课混合检索工具组合起来。"""

from pathlib import Path

from lesson_02_state_and_confirmation.agent_app.agent import StatefulAgent
from lesson_02_state_and_confirmation.agent_app.state_store import StateStore
from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_03_rag_tool.agent_app.tools.registry import RAGToolRegistry, TOOL_SCHEMAS


BASE_DIR = Path(__file__).parents[1]
DEFAULT_STATE_DIR = BASE_DIR / "agent_states"
DEFAULT_TICKET_DIR = BASE_DIR / "tickets"

RAG_SYSTEM_PROMPT = """你是公司办公 Agent，可以检索公司文档、计算和建议创建工单。
制度、标准、流程等事实必须先调用 search_company_documents，不能凭常识编造。
检索结果为空时必须明确说没有足够证据。回答事实时使用 [E1] 等 evidence_id，
并说明 source；涉及算术时继续调用 calculator。
create_ticket 是写操作，程序会等待用户确认。工具拒绝或失败后不能声称成功。"""


class RAGAgent(StatefulAgent):
    """复用状态和确认控制器，仅替换 Prompt、Schema 和注册表。"""

    def __init__(
        self,
        rag_tool,
        client=None,
        state_store=None,
        ticket_store=None,
        model=None,
    ):
        registry = RAGToolRegistry(
            rag_tool,
            ticket_store=ticket_store or TicketStore(DEFAULT_TICKET_DIR),
        )
        kwargs = {
            "client": client,
            "registry": registry,
            "state_store": state_store or StateStore(DEFAULT_STATE_DIR),
            "tool_schemas": TOOL_SCHEMAS,
            "system_prompt": RAG_SYSTEM_PROMPT,
        }
        if model is not None:
            kwargs["model"] = model
        super().__init__(**kwargs)
