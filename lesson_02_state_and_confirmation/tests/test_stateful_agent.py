"""状态持久化和人工确认流程的离线测试，不调用真实模型 API。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lesson_02_state_and_confirmation.agent_app.agent import AgentError, StatefulAgent
from lesson_02_state_and_confirmation.agent_app.state_store import StateStore
from lesson_02_state_and_confirmation.agent_app.tools.registry import ToolRegistry
from lesson_02_state_and_confirmation.agent_app.tools.ticket_store import TicketStore


def tool_call(call_id, name, arguments):
    """构造与 OpenAI SDK 结构相同的假工具调用。"""

    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def assistant(content=None, calls=None):
    """构造假的 assistant 消息。"""

    return SimpleNamespace(content=content, tool_calls=calls)


class FakeCompletions:
    """按顺序返回预设消息，并记录 Agent 发来的请求。"""

    def __init__(self, messages):
        self.responses = iter(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self.responses))]
        )


def fake_client(messages):
    """创建具有 ``chat.completions.create`` 属性层级的假客户端。"""

    completions = FakeCompletions(messages)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class StatefulAgentTests(unittest.TestCase):
    """验证状态隔离、暂停、恢复、拒绝和幂等性。"""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.state_store = StateStore(root / "states")
        self.ticket_store = TicketStore(root / "tickets")
        self.registry = ToolRegistry(ticket_store=self.ticket_store)

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def create_ticket_call(call_id="ticket_call"):
        return tool_call(
            call_id,
            "create_ticket",
            {
                "title": "投影仪故障",
                "description": "三号会议室投影仪无法开机",
                "priority": "high",
            },
        )

    def build_agent(self, responses):
        return StatefulAgent(
            client=fake_client(responses),
            registry=self.registry,
            state_store=self.state_store,
        )

    def test_sessions_are_isolated(self):
        """两个 session_id 应保存各自的问题和答案。"""

        agent = self.build_agent(
            [assistant("会话A答案"), assistant("会话B答案")]
        )
        agent.ask("问题A", session_id="session-a", show_trace=False)
        agent.ask("问题B", session_id="session-b", show_trace=False)

        state_a = self.state_store.load("session-a")
        state_b = self.state_store.load("session-b")
        self.assertIn("问题A", json.dumps(state_a, ensure_ascii=False))
        self.assertNotIn("问题B", json.dumps(state_a, ensure_ascii=False))
        self.assertIn("问题B", json.dumps(state_b, ensure_ascii=False))

    def test_ticket_is_not_created_before_confirmation(self):
        """模型提出写操作后必须暂停，不能提前创建工单。"""

        agent = self.build_agent([assistant(calls=[self.create_ticket_call()])])
        result = agent.ask("帮我报修投影仪", "pending-demo", show_trace=False)

        self.assertEqual(result.status, "waiting_confirmation")
        self.assertEqual(result.pending_action["name"], "create_ticket")
        self.assertEqual(self.ticket_store.list_tickets(), [])

    def test_confirmation_executes_ticket_once(self):
        """确认后写入一次，并把真实结果交给模型形成答案。"""

        agent = self.build_agent(
            [
                assistant(calls=[self.create_ticket_call()]),
                assistant("工单已经创建。"),
            ]
        )
        agent.ask("帮我报修投影仪", "confirm-demo", show_trace=False)
        result = agent.confirm("confirm-demo", approved=True, show_trace=False)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "工单已经创建。")
        self.assertEqual(len(self.ticket_store.list_tickets()), 1)
        state = self.state_store.load("confirm-demo")
        ticket_result = state["tool_results"][-1]["result"]
        self.assertTrue(ticket_result["approved"])
        self.assertTrue(ticket_result["ticket"]["created_now"])

        with self.assertRaises(AgentError):
            agent.confirm("confirm-demo", approved=True, show_trace=False)
        self.assertEqual(len(self.ticket_store.list_tickets()), 1)

    def test_rejection_does_not_create_ticket(self):
        """拒绝后不写入工单，并让模型如实说明。"""

        agent = self.build_agent(
            [
                assistant(calls=[self.create_ticket_call()]),
                assistant("已取消创建，没有产生工单。"),
            ]
        )
        agent.ask("帮我报修投影仪", "reject-demo", show_trace=False)
        result = agent.confirm("reject-demo", approved=False, show_trace=False)

        self.assertEqual(result.status, "completed")
        self.assertEqual(self.ticket_store.list_tickets(), [])
        state = self.state_store.load("reject-demo")
        self.assertFalse(state["tool_results"][-1]["result"]["approved"])

    def test_new_agent_restores_waiting_session(self):
        """重建 Agent 对象后仍能读取 pending_action 并继续。"""

        first_agent = self.build_agent(
            [assistant(calls=[self.create_ticket_call("restart_call")])]
        )
        first_agent.ask("创建维修工单", "restart-demo", show_trace=False)

        restarted_agent = self.build_agent([assistant("恢复后创建成功。")])
        result = restarted_agent.confirm(
            "restart-demo", approved=True, show_trace=False
        )

        self.assertEqual(result.answer, "恢复后创建成功。")
        self.assertEqual(len(self.ticket_store.list_tickets()), 1)

    def test_repeated_write_call_is_blocked(self):
        """确认后模型再次请求相同写操作时不能重复创建。"""

        same_arguments_call = self.create_ticket_call
        agent = self.build_agent(
            [
                assistant(calls=[same_arguments_call("first_call")]),
                assistant(calls=[same_arguments_call("second_call")]),
                assistant("重复请求已被阻止，原工单保留。"),
            ]
        )
        agent.ask("创建维修工单", "repeat-demo", show_trace=False)
        result = agent.confirm("repeat-demo", approved=True, show_trace=False)

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(self.ticket_store.list_tickets()), 1)
        repeated_result = self.state_store.load("repeat-demo")["tool_results"][-1]
        self.assertIn("重复工具调用", repeated_result["result"]["error"])


if __name__ == "__main__":
    unittest.main()
