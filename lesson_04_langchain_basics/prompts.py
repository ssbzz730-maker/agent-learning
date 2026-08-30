"""使用 ChatPromptTemplate 声明消息结构。"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


SYSTEM_PROMPT = """你是公司办公助手。
回答时区分事实证据与推测；没有可靠证据时明确说明。"""

AGENT_SYSTEM_PROMPT = """你是公司办公工具 Agent。
涉及算术时使用 calculator；涉及公司制度、标准、流程和规定时，必须先使用
search_company_documents（如果该工具可用）。工具只返回数据或证据，你需要根据
工具结果生成最终答案。工具失败或证据不足时如实说明，不能声称操作成功。"""


def build_basic_prompt():
    """创建只接收 ``question`` 的基础聊天 Prompt。"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "{question}"),
        ]
    )


def build_conversation_prompt(system_prompt=SYSTEM_PROMPT):
    """创建支持可选历史消息和自定义系统指令的 Prompt。"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history", optional=True),
            ("user", "{question}"),
        ]
    )


def build_agent_prompt():
    """创建要求模型根据问题选择已绑定工具的 Agent Prompt。"""

    return build_conversation_prompt(AGENT_SYSTEM_PROMPT)


def render_basic_messages(question):
    """把输入字典格式化为模型能够接收的消息列表。"""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question 必须是非空字符串")
    return build_basic_prompt().invoke({"question": question.strip()}).to_messages()
