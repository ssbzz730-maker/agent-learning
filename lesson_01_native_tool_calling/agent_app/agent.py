"""原生Tool Calling控制循环：模型决策，程序校验并执行工具。"""

import json
from dataclasses import dataclass

from openai import OpenAI

from lesson_01_native_tool_calling.agent_app.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    get_api_key,
)
from lesson_01_native_tool_calling.agent_app.tools.registry import (
    TOOL_SCHEMAS,
    ToolRegistry,
)


SYSTEM_PROMPT = """你是公司办公助手，可以使用计算器和公司制度知识库。
制度事实必须先调用 search_knowledge_base，不能凭常识编造。
如果知识库没有相关结果，要明确说知识库中没有相关规定。
涉及计算时使用 calculator，不要自己心算。
需要多个工具时可以分步骤调用。最终答案简洁，并标明知识来源。"""


class AgentError(RuntimeError):
    """Agent配置错误或控制循环无法安全继续。"""


@dataclass
class AgentResult:
    """一次Agent任务的最终答案、消息轨迹和执行步数。"""

    answer: str
    messages: list
    steps: int


def assistant_message_to_dict(message):
    """把SDK消息转换成下一轮API能够接受的普通字典。"""

    payload = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return payload


class NativeAgent:
    """限制步骤数、检测重复调用并保存完整消息轨迹的Agent。"""

    def __init__(self, client=None, registry=None, model=DEEPSEEK_MODEL):
        api_key = get_api_key()
        if client is None and not api_key:
            raise AgentError("未设置 DEEPSEEK_API_KEY")
        self.client = client or OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.registry = registry or ToolRegistry()
        self.model = model

    def run(self, question, max_steps=5, show_trace=True):
        """循环调用模型和工具，直到得到答案或达到最大步骤数。"""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 10:
            raise ValueError("max_steps 必须是 1 到 10 之间的整数")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ]
        executed_calls = set()

        for step in range(1, max_steps + 1):
            if show_trace:
                print(f"\n[步骤 {step}] 正在让模型决定下一步……")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1000,
            )
            message = response.choices[0].message
            messages.append(assistant_message_to_dict(message))

            if not message.tool_calls:
                answer = (message.content or "").strip()
                if not answer:
                    raise AgentError("模型既没有返回答案，也没有调用工具")
                if show_trace:
                    print(f"[最终答案] {answer}")
                return AgentResult(answer=answer, messages=messages, steps=step)

            for call in message.tool_calls:
                name = call.function.name
                arguments = call.function.arguments
                fingerprint = (name, arguments)
                if fingerprint in executed_calls:
                    raise AgentError(f"检测到重复工具调用：{name}({arguments})")
                executed_calls.add(fingerprint)
                if show_trace:
                    print(f"[工具选择] {name}")
                    print(f"[工具参数] {arguments}")
                try:
                    result = self.registry.execute(name, arguments)
                    content = json.dumps(result, ensure_ascii=False)
                except Exception as error:
                    content = json.dumps(
                        {"error": str(error), "success": False}, ensure_ascii=False
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )
                if show_trace:
                    print(f"[工具结果] {content}")

        raise AgentError(f"任务在 {max_steps} 步内没有完成")
