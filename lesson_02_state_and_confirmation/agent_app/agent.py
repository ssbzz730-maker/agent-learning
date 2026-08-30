"""持久化 Agent 控制循环，以及写工具的暂停、确认和恢复。"""

import hashlib
import json
from dataclasses import dataclass

from openai import OpenAI

from lesson_01_native_tool_calling.agent_app.agent import assistant_message_to_dict
from lesson_01_native_tool_calling.agent_app.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    get_api_key,
)
from lesson_02_state_and_confirmation.agent_app.state_store import StateStore
from lesson_02_state_and_confirmation.agent_app.tools.registry import (
    TOOL_SCHEMAS,
    ToolRegistry,
)


SYSTEM_PROMPT = """你是公司办公助手，可以查询制度、计算并建议创建内部工单。
制度事实必须先查询知识库；涉及计算必须使用计算器。
create_ticket 会改变外部状态，一次只能调用一个写工具。程序会暂停并等待用户确认，
你不能替用户确认。工具被拒绝或失败后，应如实向用户说明，不能声称已经创建。"""


class AgentError(RuntimeError):
    """会话状态或控制循环无法安全继续。"""


@dataclass
class AgentRunResult:
    """一次运行可能完成，也可能暂停等待人工确认。"""

    session_id: str
    status: str
    answer: str | None
    pending_action: dict | None
    messages: list
    steps: int


def _new_state(session_id, system_prompt=SYSTEM_PROMPT):
    """创建可被 JSON 序列化的初始状态。"""

    return {
        "session_id": session_id,
        "messages": [{"role": "system", "content": system_prompt}],
        "step": 0,
        "status": "running",
        "pending_action": None,
        "tool_results": [],
        "executed_calls": [],
        "answer": None,
        "last_error": None,
    }


class StatefulAgent:
    """保存每一步状态，并在写工具执行前暂停。"""

    def __init__(
        self,
        client=None,
        registry=None,
        state_store=None,
        model=DEEPSEEK_MODEL,
        tool_schemas=None,
        system_prompt=SYSTEM_PROMPT,
    ):
        api_key = get_api_key()
        if client is None and not api_key:
            raise AgentError("未设置 DEEPSEEK_API_KEY")
        self.client = client or OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.registry = registry or ToolRegistry()
        self.state_store = state_store or StateStore()
        self.model = model
        self.tool_schemas = TOOL_SCHEMAS if tool_schemas is None else tool_schemas
        self.system_prompt = system_prompt

    @staticmethod
    def _result(state):
        return AgentRunResult(
            session_id=state["session_id"],
            status=state["status"],
            answer=state.get("answer"),
            pending_action=state.get("pending_action"),
            messages=state["messages"],
            steps=state["step"],
        )

    @staticmethod
    def _fingerprint(name, arguments):
        canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"{name}:{canonical}"

    def ask(self, question, session_id="default", max_steps=5, show_trace=True):
        """开始新一轮问题，同时保留同一 session 的历史消息。"""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 10:
            raise ValueError("max_steps 必须是 1 到 10 之间的整数")

        state = self.state_store.load(session_id)
        if state and state["status"] == "waiting_confirmation":
            raise AgentError("当前会话正在等待确认，请先执行 confirm 或 reject")
        if state and state["status"] == "running":
            raise AgentError("当前会话有未完成任务，不能直接加入新问题")
        if state is None:
            state = _new_state(
                StateStore.validate_session_id(session_id),
                system_prompt=self.system_prompt,
            )
        else:
            state.update(
                {
                    "step": 0,
                    "status": "running",
                    "pending_action": None,
                    "executed_calls": [],
                    "answer": None,
                    "last_error": None,
                }
            )
        state["messages"].append({"role": "user", "content": question.strip()})
        self.state_store.save(state)
        return self._continue(state, max_steps=max_steps, show_trace=show_trace)

    def confirm(self, session_id, approved, max_steps=5, show_trace=True):
        """确认或拒绝待执行写操作，然后从持久化消息继续运行。"""

        if not isinstance(approved, bool):
            raise TypeError("approved 必须是布尔值")
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 10:
            raise ValueError("max_steps 必须是 1 到 10 之间的整数")
        state = self.state_store.load(session_id)
        if not state or state["status"] != "waiting_confirmation":
            raise AgentError("当前会话没有等待确认的操作")

        action = state["pending_action"]
        if approved:
            try:
                result = self.registry.execute(
                    action["name"],
                    action["arguments"],
                    idempotency_key=action["action_id"],
                )
                result["approved"] = True
            except Exception as error:
                result = {"success": False, "approved": True, "error": str(error)}
        else:
            result = {
                "success": False,
                "approved": False,
                "error": "用户拒绝执行此写操作",
            }

        content = json.dumps(result, ensure_ascii=False)
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": action["tool_call_id"],
                "content": content,
            }
        )
        state["tool_results"].append(
            {
                "tool_call_id": action["tool_call_id"],
                "name": action["name"],
                "result": result,
            }
        )
        state["pending_action"] = None
        state["status"] = "running"
        self.state_store.save(state)
        if show_trace:
            decision = "已确认并执行" if approved else "已拒绝，未执行"
            print(f"[人工确认] {decision} {action['name']}")
        return self._continue(state, max_steps=max_steps, show_trace=show_trace)

    def get_state(self, session_id):
        """读取状态，便于 CLI 查看和程序重启后恢复。"""

        state = self.state_store.load(session_id)
        if state is None:
            raise AgentError("会话不存在")
        return state

    def _append_tool_result(self, state, call, name, result, show_trace):
        content = json.dumps(result, ensure_ascii=False)
        state["messages"].append(
            {"role": "tool", "tool_call_id": call.id, "content": content}
        )
        state["tool_results"].append(
            {"tool_call_id": call.id, "name": name, "result": result}
        )
        self.state_store.save(state)
        if show_trace:
            print(f"[工具结果] {content}")

    def _continue(self, state, max_steps, show_trace):
        """从保存的消息继续模型循环，直到完成、暂停或失败。"""

        while state["step"] < max_steps:
            state["step"] += 1
            self.state_store.save(state)
            if show_trace:
                print(f"\n[步骤 {state['step']}] 正在让模型决定下一步……")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=state["messages"],
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=1000,
            )
            message = response.choices[0].message
            state["messages"].append(assistant_message_to_dict(message))

            if not message.tool_calls:
                answer = (message.content or "").strip()
                if not answer:
                    return self._fail(state, "模型既没有返回答案，也没有调用工具")
                state["answer"] = answer
                state["status"] = "completed"
                self.state_store.save(state)
                if show_trace:
                    print(f"[最终答案] {answer}")
                return self._result(state)

            risky_calls = [
                call
                for call in message.tool_calls
                if self.registry.requires_confirmation(call.function.name)
            ]
            if risky_calls and len(message.tool_calls) != 1:
                for call in message.tool_calls:
                    self._append_tool_result(
                        state,
                        call,
                        call.function.name,
                        {
                            "success": False,
                            "error": "写工具必须单独调用，不能与其他工具并行执行",
                        },
                        show_trace,
                    )
                continue

            for call in message.tool_calls:
                name = call.function.name
                try:
                    arguments = self.registry.validate(name, call.function.arguments)
                except Exception as error:
                    self._append_tool_result(
                        state,
                        call,
                        name,
                        {"success": False, "error": str(error)},
                        show_trace,
                    )
                    continue

                fingerprint = self._fingerprint(name, arguments)
                if fingerprint in state["executed_calls"]:
                    self._append_tool_result(
                        state,
                        call,
                        name,
                        {"success": False, "error": "重复工具调用已被控制器阻止"},
                        show_trace,
                    )
                    continue
                state["executed_calls"].append(fingerprint)

                if self.registry.requires_confirmation(name):
                    action_source = f"{state['session_id']}:{call.id}:{fingerprint}"
                    state["pending_action"] = {
                        "action_id": hashlib.sha256(
                            action_source.encode("utf-8")
                        ).hexdigest(),
                        "tool_call_id": call.id,
                        "name": name,
                        "arguments": arguments,
                    }
                    state["status"] = "waiting_confirmation"
                    self.state_store.save(state)
                    if show_trace:
                        print(f"[等待确认] {name}: {arguments}")
                    return self._result(state)

                try:
                    result = self.registry.execute(name, arguments)
                except Exception as error:
                    result = {"success": False, "error": str(error)}
                self._append_tool_result(
                    state, call, name, result, show_trace=show_trace
                )

        return self._fail(state, f"任务在 {max_steps} 步内没有完成")

    def _fail(self, state, message):
        state["status"] = "failed"
        state["last_error"] = message
        self.state_store.save(state)
        raise AgentError(message)
