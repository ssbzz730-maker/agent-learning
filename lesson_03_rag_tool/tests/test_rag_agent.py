"""验证 Agent 能先检索证据、再计算并形成带引用答案。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lesson_02_state_and_confirmation.agent_app.state_store import StateStore
from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore
from lesson_03_rag_tool.agent_app.agent import RAGAgent
from lesson_03_rag_tool.tests.test_rag_tool import FakeBackend
from lesson_03_rag_tool.agent_app.tools.rag_tool import HybridRAGTool


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeCompletions:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self.messages))]
        )


class RAGAgentTests(unittest.TestCase):
    def test_agent_combines_rag_and_calculator(self):
        responses = [
            SimpleNamespace(
                content=None,
                tool_calls=[
                    tool_call(
                        "rag_call",
                        "search_company_documents",
                        {"query": "上海住宿一晚最多报销多少？", "top_k": 1},
                    )
                ],
            ),
            SimpleNamespace(
                content=None,
                tool_calls=[
                    tool_call("calc_call", "calculator", {"expression": "500*3"})
                ],
            ),
            SimpleNamespace(
                content="上海住宿三晚最多1500元，依据policy.md [E1]。",
                tool_calls=None,
            ),
        ]
        completions = FakeCompletions(responses)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = RAGAgent(
                HybridRAGTool(FakeBackend()),
                client=client,
                state_store=StateStore(root / "states"),
                ticket_store=TicketStore(root / "tickets"),
            )
            result = agent.ask(
                "上海出差住三晚最多报销多少？",
                session_id="rag-agent-demo",
                show_trace=False,
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, 3)
        self.assertIn("1500元", result.answer)
        tool_messages = [
            message for message in result.messages if message["role"] == "tool"
        ]
        rag_result = json.loads(tool_messages[0]["content"])
        self.assertEqual(rag_result["results"][0]["evidence_id"], "E1")
        self.assertEqual(json.loads(tool_messages[1]["content"])["result"], 1500)
        visible_tools = {
            item["function"]["name"]
            for item in completions.requests[0]["tools"]
        }
        self.assertIn("search_company_documents", visible_tools)
        self.assertNotIn("search_knowledge_base", visible_tools)


if __name__ == "__main__":
    unittest.main()
