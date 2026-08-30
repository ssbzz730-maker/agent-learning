"""通过 Runnable 管道组合输入清洗、Prompt、模型和输出解析器。"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from lesson_01_native_tool_calling.agent_app.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    get_api_key,
)
from lesson_04_langchain_basics.prompts import build_basic_prompt


def normalize_input(payload):
    """校验 Runnable 输入，并去掉问题首尾空白。"""

    if not isinstance(payload, dict):
        raise TypeError("chain 输入必须是字典")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question 必须是非空字符串")
    return {"question": question.strip()}


def create_deepseek_model():
    """创建 DeepSeek 的 OpenAI 兼容 LangChain 模型对象。"""

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        temperature=0,
        max_tokens=1000,
    )


def build_answer_chain(model=None):
    """返回 ``输入清洗 | Prompt | 模型 | 字符串解析`` 管道。"""

    model = model or create_deepseek_model()
    return (
        RunnableLambda(normalize_input)
        | build_basic_prompt()
        | model
        | StrOutputParser()
    )
