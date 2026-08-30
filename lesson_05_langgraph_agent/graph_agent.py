"""LangGraph Agent：状态图、条件边、Checkpoint与人工确认中断。"""

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from lesson_04_langchain_basics.prompts import AGENT_SYSTEM_PROMPT
from lesson_05_langgraph_agent.state import AgentState


GRAPH_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT + """
需要创建工单时使用create_ticket。该工具是写操作，程序会暂停并询问用户；
没有收到工具结果前不能声称工单已创建。"""


@dataclass
class GraphAgentResult:
    """图运行或恢复后的用户可见结果。"""

    thread_id: str
    status: str
    answer: str | None
    pending_action: dict | None
    messages: list
    model_calls: int


class LangGraphAgent:
    """把第4课手动循环改写成可暂停、可恢复的状态图。"""

    def __init__(self, model, tools, checkpointer=None, risky_tools=None):
        if not tools:
            raise ValueError("Agent至少需要一个工具")
        self.tools = {}
        for tool_object in tools:
            if tool_object.name in self.tools:
                raise ValueError(f"工具名称重复：{tool_object.name}")
            self.tools[tool_object.name] = tool_object
        self.risky_tools = set(risky_tools or {"create_ticket"})
        self.model = model.bind_tools(list(self.tools.values()))
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """注册节点和边，并使用Checkpointer编译图。"""

        builder = StateGraph(AgentState)
        builder.add_node("agent", self._call_model)
        builder.add_node("tools", self._call_tools)
        builder.add_node("approval", self._request_approval)
        builder.add_node("rejected", self._reject_tools)
        builder.add_node("limit", self._stop_at_limit)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_model,
            {
                "tools": "tools",
                "approval": "approval",
                "limit": "limit",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "approval",
            self._route_after_approval,
            {"approved": "tools", "rejected": "rejected"},
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("rejected", "agent")
        builder.add_edge("limit", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _call_model(self, state):
        """模型节点：读取全部历史消息，并追加一个AIMessage。"""

        response = self.model.invoke(
            [SystemMessage(content=GRAPH_SYSTEM_PROMPT), *state["messages"]]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("模型节点必须返回AIMessage")
        return {
            "messages": [response],
            "model_calls": state.get("model_calls", 0) + 1,
        }

    def _call_tools(self, state):
        """工具节点：执行最后一个AIMessage中的全部tool_calls。"""

        messages = []
        events = []
        for call in self._last_tool_calls(state):
            tool_message, event = self._execute_tool(call)
            messages.append(tool_message)
            events.append(event)
        return {"messages": messages, "tool_events": events, "approved": None}

    def _request_approval(self, state):
        """风险节点：保存Checkpoint并暂停，恢复时接收用户的布尔决定。"""

        calls = self._last_tool_calls(state)
        decision = interrupt(
            {
                "type": "tool_approval",
                "message": "以下写操作需要你的确认",
                "tool_calls": calls,
            }
        )
        return {"approved": bool(decision)}

    def _reject_tools(self, state):
        """拒绝节点：不执行工具，只把拒绝结果作为ToolMessage返回模型。"""

        messages = []
        events = []
        for call in self._last_tool_calls(state):
            payload = {"success": False, "error": "用户拒绝执行该工具"}
            messages.append(self._tool_message(call, payload))
            events.append(self._event(call, payload))
        return {"messages": messages, "tool_events": events, "approved": None}

    def _stop_at_limit(self, state):
        """达到步数上限时补齐工具响应协议，并给出诚实的终止信息。"""

        messages = []
        events = []
        for call in self._last_tool_calls(state):
            payload = {"success": False, "error": "达到最大模型调用次数，工具未执行"}
            messages.append(self._tool_message(call, payload))
            events.append(self._event(call, payload))
        messages.append(AIMessage(content="达到最大步骤，未继续执行工具。"))
        return {"messages": messages, "tool_events": events, "approved": None}

    def _route_after_model(self, state):
        """条件边：按最终回答、步数、风险工具或安全工具选择下一节点。"""

        calls = self._last_tool_calls(state, allow_empty=True)
        if not calls:
            return "end"
        if state["model_calls"] >= state["max_steps"]:
            return "limit"
        if any(call["name"] in self.risky_tools for call in calls):
            return "approval"
        return "tools"

    @staticmethod
    def _route_after_approval(state):
        return "approved" if state["approved"] else "rejected"

    def _execute_tool(self, call):
        name = call.get("name")
        arguments = call.get("args", {})
        tool_object = self.tools.get(name)
        if tool_object is None:
            payload = {"success": False, "error": f"未注册工具：{name}"}
        else:
            try:
                payload = {"success": True, "result": tool_object.invoke(arguments)}
            except Exception as error:
                payload = {"success": False, "error": str(error)}
        return self._tool_message(call, payload), self._event(call, payload)

    @staticmethod
    def _tool_message(call, payload):
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool_call缺少调用ID")
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, default=str),
            tool_call_id=call_id,
            name=call.get("name"),
        )

    @staticmethod
    def _event(call, payload):
        return {
            "call_id": call.get("id"),
            "name": call.get("name"),
            "arguments": call.get("args", {}),
            **payload,
        }

    @staticmethod
    def _last_tool_calls(state, allow_empty=False):
        if not state.get("messages") or not isinstance(
            state["messages"][-1], AIMessage
        ):
            raise ValueError("当前状态最后一条消息不是AIMessage")
        calls = state["messages"][-1].tool_calls
        if not calls and not allow_empty:
            raise ValueError("当前AIMessage不包含tool_calls")
        return calls

    @staticmethod
    def _config(thread_id):
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id必须是非空字符串")
        return {"configurable": {"thread_id": thread_id.strip()}}

    def ask(self, question, thread_id="default", max_steps=6):
        """向指定线程追加问题；Checkpoint会自动合并旧消息。"""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question必须是非空字符串")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps必须是正整数")
        config = self._config(thread_id)
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question.strip())],
                "model_calls": 0,
                "max_steps": max_steps,
                "approved": None,
                "tool_events": [],
            },
            config=config,
        )
        return self._to_result(thread_id, result)

    def resume(self, thread_id, approved):
        """用同一thread_id确认或拒绝Checkpoint中的待执行工具。"""

        if not isinstance(approved, bool):
            raise TypeError("approved必须是布尔值")
        result = self.graph.invoke(
            Command(resume=approved),
            config=self._config(thread_id),
        )
        return self._to_result(thread_id, result)

    def get_state(self, thread_id):
        """读取指定线程的最新Checkpoint，不调用模型。"""

        return self.graph.get_state(self._config(thread_id))

    @staticmethod
    def _to_result(thread_id, state):
        interrupts = state.get("__interrupt__", [])
        if interrupts:
            return GraphAgentResult(
                thread_id, "awaiting_confirmation", None,
                interrupts[0].value, state.get("messages", []),
                state.get("model_calls", 0),
            )
        answer = None
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )
                break
        return GraphAgentResult(
            thread_id, "completed", answer, None,
            state.get("messages", []), state.get("model_calls", 0),
        )
