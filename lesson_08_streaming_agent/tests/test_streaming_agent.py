"""真实LangGraph消息流、工具事件和人工确认恢复测试。"""

import tempfile
import unittest

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_04_langchain_basics.tools import calculator_tool
from lesson_05_langgraph_agent.tests.test_langgraph_agent import (
    ScriptedToolModel,
    tool_call,
)
from lesson_05_langgraph_agent.tools import build_create_ticket_tool
from lesson_06_agent_quality.observability import EventLogger
from lesson_08_streaming_agent.streaming_agent import StreamingQualityAgent


async def collect(source):
    return [item async for item in source]


class StreamingAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_contains_tool_token_result_and_done(self):
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call("calculator", {"expression": "6*7"}, "calc-1")
                    ],
                ),
                AIMessage(content="答案是42。"),
            ]
        )
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            await saver.setup()
            agent = StreamingQualityAgent(
                model,
                [calculator_tool],
                EventLogger(),
                saver,
            )
            events = await collect(agent.astream_ask("计算6乘7", "stream-test"))

        names = [item["event"] for item in events]
        self.assertEqual(names[0], "started")
        self.assertIn("tool_requested", names)
        self.assertIn("tool_finished", names)
        self.assertIn("token", names)
        self.assertEqual(names[-2:], ["result", "done"])
        result = next(item["data"] for item in events if item["event"] == "result")
        self.assertEqual(result["answer"], "答案是42。")
        public_tool = next(
            item["data"] for item in events if item["event"] == "tool_finished"
        )
        self.assertNotIn("result", public_tool)

    async def test_confirmation_stream_resumes_same_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TicketStore(directory)
            model = ScriptedToolModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            tool_call(
                                "create_ticket",
                                {"title": "网络故障", "description": "无法联网"},
                                "ticket-1",
                            )
                        ],
                    ),
                    AIMessage(content="工单已创建。"),
                ]
            )
            async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
                await saver.setup()
                agent = StreamingQualityAgent(
                    model,
                    [build_create_ticket_tool(store)],
                    EventLogger(),
                    saver,
                )
                waiting = await collect(agent.astream_ask("创建工单", "approval"))
                completed = await collect(agent.astream_resume("approval", True))
                waiting_result = next(
                    item["data"] for item in waiting if item["event"] == "result"
                )
                completed_result = next(
                    item["data"] for item in completed if item["event"] == "result"
                )
                self.assertIn(
                    "approval_required",
                    [item["event"] for item in waiting],
                )
                self.assertEqual(
                    waiting_result["status"],
                    "awaiting_confirmation",
                )
                self.assertEqual(completed_result["status"], "completed")
                self.assertEqual(len(store.list_tickets()), 1)


if __name__ == "__main__":
    unittest.main()
