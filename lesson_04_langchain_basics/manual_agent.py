"""用 LangChain 消息和 Tool 手动实现一个可观察的 Agent 控制循环。"""

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, ToolMessage

from lesson_04_langchain_basics.prompts import build_agent_prompt


class AgentLoopError(RuntimeError):
    """Agent 达到最大模型调用次数、但仍未生成最终答案。"""


@dataclass
class AgentResult:
    """一次 Agent 运行的最终答案、完整消息和工具执行轨迹。"""

    answer: str
    messages: list
    steps: int
    tool_events: list


class ManualToolAgent:
    """显式演示 Tool 绑定、选择、执行和 ToolMessage 回传。"""

    def __init__(self, model, tools):
        if not tools:
            raise ValueError("Agent 至少需要一个工具")
        self.tools = {}
        for tool_object in tools:
            if tool_object.name in self.tools:
                raise ValueError(f"工具名称重复：{tool_object.name}")
            self.tools[tool_object.name] = tool_object

        # bind_tools 只把工具说明和参数 Schema 交给模型，不会执行工具。
        self.model = model.bind_tools(list(self.tools.values()))

    def ask(self, question, history=None, max_steps=6, show_trace=True):
        """运行手动工具循环，直到模型返回普通文本或达到步数上限。"""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 必须是非空字符串")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps 必须是正整数")
        if history is not None and not isinstance(history, (list, tuple)):
            raise TypeError("history 必须是消息列表")

        messages = build_agent_prompt().invoke(
            {
                "history": list(history or []),
                "question": question.strip(),
            }
        ).to_messages()
        tool_events = []

        for step in range(1, max_steps + 1):
            message = self.model.invoke(messages)
            if not isinstance(message, AIMessage):
                raise TypeError("绑定工具后的模型必须返回 AIMessage")
            messages.append(message)

            if not message.tool_calls:
                answer = self._message_text(message)
                if show_trace:
                    print(f"[第{step}步] 模型生成最终答案")
                return AgentResult(answer, messages, step, tool_events)

            if show_trace:
                names = ", ".join(call["name"] for call in message.tool_calls)
                print(f"[第{step}步] 模型请求工具：{names}")

            for call in message.tool_calls:
                tool_message, event = self._execute_tool(call)
                messages.append(tool_message)
                tool_events.append(event)
                if show_trace:
                    status = "成功" if event["success"] else "失败"
                    print(f"  └─ {event['name']}：{status}")

        raise AgentLoopError(f"达到最大步骤 {max_steps}，模型仍在请求工具")

    def _execute_tool(self, call):
        """校验工具名、调用真实 Tool，并构造与调用 ID 对应的 ToolMessage。"""

        name = call.get("name")
        arguments = call.get("args", {})
        call_id = call.get("id")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_call 缺少工具名称")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call 的 args 必须是字典")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool_call 缺少调用 ID")

        tool_object = self.tools.get(name)
        if tool_object is None:
            payload = {"success": False, "error": f"未注册工具：{name}"}
        else:
            try:
                result = tool_object.invoke(arguments)
                payload = {"success": True, "result": result}
            except Exception as error:  # 工具错误必须返回模型，避免循环猜测。
                payload = {"success": False, "error": str(error)}

        content = json.dumps(payload, ensure_ascii=False, default=str)
        event = {
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            **payload,
        }
        return (
            ToolMessage(content=content, tool_call_id=call_id, name=name),
            event,
        )

    @staticmethod
    def _message_text(message):
        """把最终 AIMessage 的文本内容转换成命令行可输出的字符串。"""

        if isinstance(message.content, str):
            return message.content
        return json.dumps(message.content, ensure_ascii=False, default=str)
