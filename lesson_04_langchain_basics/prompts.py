"""使用 ChatPromptTemplate 声明消息结构。"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


SYSTEM_PROMPT = """你是公司办公助手。
回答时区分事实证据与推测；没有可靠证据时明确说明。"""


def build_basic_prompt():
    """创建只接收 ``question`` 的基础聊天 Prompt。"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "{question}"),
        ]
    )


def build_conversation_prompt():
    """创建支持可选历史消息的 Prompt。"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("user", "{question}"),
        ]
    )


def render_basic_messages(question):
    """把输入字典格式化为模型能够接收的消息列表。"""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question 必须是非空字符串")
    return build_basic_prompt().invoke({"question": question.strip()}).to_messages()
