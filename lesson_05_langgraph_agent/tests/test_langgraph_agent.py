"""LangGraph状态图、Checkpoint、条件边与人工确认的离线测试。"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_04_langchain_basics.tools import calculator_tool
from lesson_05_langgraph_agent.graph_agent import LangGraphAgent
from lesson_05_langgraph_agent.tools import build_create_ticket_tool


def tool_call(name, arguments, call_id):
    """构造LangChain标准tool_call，供假模型按脚本返回。"""

    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


class ScriptedToolModel:
    """离线模型：记录输入消息，并按顺序返回预设AIMessage。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("假模型没有剩余响应")
        return self.responses.pop(0)


class LangGraphAgentTests(unittest.TestCase):
    def test_safe_tool_follows_agent_tools_agent_path(self):
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[tool_call("calculator", {"expression": "25*4"}, "c1")],
                ),
                AIMessage(content="答案是100。"),
            ]
        )
        agent = LangGraphAgent(model, [calculator_tool], InMemorySaver())
        result = agent.ask("计算25乘4", thread_id="math", max_steps=3)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "答案是100。")
        self.assertEqual(result.model_calls, 2)
        self.assertEqual([tool.name for tool in model.bound_tools], ["calculator"])
        tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
        self.assertEqual(tool_messages[0].tool_call_id, "c1")
        self.assertIn("100", tool_messages[0].content)

    def test_checkpoint_preserves_history_for_same_thread(self):
        model = ScriptedToolModel(
            [AIMessage(content="第一答"), AIMessage(content="第二答")]
        )
        agent = LangGraphAgent(model, [calculator_tool], InMemorySaver())

        agent.ask("第一问", thread_id="same")
        result = agent.ask("第二问", thread_id="same")

        contents = [message.content for message in result.messages]
        self.assertEqual(contents, ["第一问", "第一答", "第二问", "第二答"])
        second_model_input = model.calls[1]
        self.assertTrue(any(m.content == "第一问" for m in second_model_input))

    def test_thread_ids_isolate_conversations(self):
        model = ScriptedToolModel(
            [AIMessage(content="A答"), AIMessage(content="B答")]
        )
        agent = LangGraphAgent(model, [calculator_tool], InMemorySaver())

        result_a = agent.ask("A问", thread_id="thread-a")
        result_b = agent.ask("B问", thread_id="thread-b")

        self.assertEqual([m.content for m in result_a.messages], ["A问", "A答"])
        self.assertEqual([m.content for m in result_b.messages], ["B问", "B答"])

    def test_risky_tool_interrupts_then_approval_executes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TicketStore(directory)
            create_ticket = build_create_ticket_tool(store)
            model = ScriptedToolModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            tool_call(
                                "create_ticket",
                                {
                                    "title": "显示器故障",
                                    "description": "屏幕无法点亮",
                                    "priority": "high",
                                },
                                "ticket-1",
                            )
                        ],
                    ),
                    AIMessage(content="工单已创建。"),
                ]
            )
            agent = LangGraphAgent(
                model,
                [calculator_tool, create_ticket],
                InMemorySaver(),
            )

            waiting = agent.ask("帮我创建工单", thread_id="approval")
            self.assertEqual(waiting.status, "awaiting_confirmation")
            self.assertEqual(
                waiting.pending_action["tool_calls"][0]["name"],
                "create_ticket",
            )
            self.assertEqual(store.list_tickets(), [])

            completed = agent.resume("approval", approved=True)
            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.answer, "工单已创建。")
            self.assertEqual(len(store.list_tickets()), 1)

    def test_rejection_returns_tool_message_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TicketStore(directory)
            create_ticket = build_create_ticket_tool(store)
            model = ScriptedToolModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            tool_call(
                                "create_ticket",
                                {"title": "测试", "description": "不应创建"},
                                "ticket-2",
                            )
                        ],
                    ),
                    AIMessage(content="用户已拒绝，没有创建工单。"),
                ]
            )
            agent = LangGraphAgent(model, [create_ticket], InMemorySaver())

            agent.ask("创建测试工单", thread_id="reject")
            completed = agent.resume("reject", approved=False)

            self.assertEqual(completed.status, "completed")
            self.assertEqual(store.list_tickets(), [])
            rejected = [m for m in completed.messages if isinstance(m, ToolMessage)]
            self.assertIn("用户拒绝", rejected[0].content)

    def test_max_steps_stops_before_requested_tool_execution(self):
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("calculator", {"expression": "2+2"}, "limit")
                    ],
                )
            ]
        )
        agent = LangGraphAgent(model, [calculator_tool], InMemorySaver())
        result = agent.ask("计算", thread_id="limit", max_steps=1)

        self.assertEqual(result.status, "completed")
        self.assertIn("达到最大步骤", result.answer)
        self.assertFalse(
            any(
                "4" in message.content
                for message in result.messages
                if isinstance(message, ToolMessage)
            )
        )

    def test_sqlite_checkpoint_survives_agent_recreation(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite3"
            first_connection = sqlite3.connect(database, check_same_thread=False)
            first_saver = SqliteSaver(first_connection)
            first_saver.setup()
            first_agent = LangGraphAgent(
                ScriptedToolModel([AIMessage(content="持久化回答")]),
                [calculator_tool],
                first_saver,
            )
            first_agent.ask("持久化问题", thread_id="persist")
            first_connection.close()

            second_connection = sqlite3.connect(database, check_same_thread=False)
            second_saver = SqliteSaver(second_connection)
            second_saver.setup()
            second_agent = LangGraphAgent(
                ScriptedToolModel([AIMessage(content="新回答")]),
                [calculator_tool],
                second_saver,
            )
            snapshot = second_agent.get_state("persist")
            second_connection.close()

            contents = [message.content for message in snapshot.values["messages"]]
            self.assertEqual(contents, ["持久化问题", "持久化回答"])

    def test_graph_contains_explicit_learning_nodes(self):
        model = ScriptedToolModel([AIMessage(content="不用运行")])
        agent = LangGraphAgent(model, [calculator_tool], InMemorySaver())
        node_names = set(agent.graph.get_graph().nodes)

        expected = {"agent", "tools", "approval", "rejected", "limit"}
        self.assertTrue(expected <= node_names)


if __name__ == "__main__":
    unittest.main()
