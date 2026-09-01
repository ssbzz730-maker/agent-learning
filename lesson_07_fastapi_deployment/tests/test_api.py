"""FastAPI契约、SSE、审批和静态网页的离线测试。"""

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from lesson_07_fastapi_deployment.api import create_app


class FakeAgentService:
    def __init__(self):
        self.chat_calls = []
        self.resume_calls = []

    def chat(self, question, thread_id, max_steps=6):
        self.chat_calls.append((question, thread_id, max_steps))
        return SimpleNamespace(
            request_id="req-fake",
            thread_id=thread_id,
            status="completed",
            answer="离线答案",
            pending_action=None,
            model_calls=2,
        )

    def resume(self, thread_id, approved):
        self.resume_calls.append((thread_id, approved))
        return SimpleNamespace(
            request_id="req-resume",
            thread_id=thread_id,
            status="completed",
            answer="已确认" if approved else "已拒绝",
            pending_action=None,
            model_calls=2,
        )

    def get_session(self, thread_id):
        if thread_id == "missing":
            raise KeyError(thread_id)
        return {
            "thread_id": thread_id,
            "next_nodes": [],
            "model_calls": 2,
            "messages": [{"type": "human", "content": "你好"}],
        }


class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.service = FakeAgentService()
        self.client_context = TestClient(create_app(self.service))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health_and_chinese_page(self):
        health = self.client.get("/health")
        page = self.client.get("/")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertIn("企业知识与工单 Agent", page.text)

    def test_chat_contract(self):
        response = self.client.post(
            "/api/chat",
            json={"question": "测试", "thread_id": "web-1", "max_steps": 4},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "离线答案")
        self.assertEqual(self.service.chat_calls, [("测试", "web-1", 4)])

    def test_chat_validation_rejects_empty_question(self):
        response = self.client.post(
            "/api/chat",
            json={"question": "", "thread_id": "web-1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_sse_stream_contains_lifecycle_events(self):
        response = self.client.post(
            "/api/chat/stream",
            json={"question": "流式测试", "thread_id": "stream-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: started", response.text)
        self.assertIn("event: result", response.text)
        self.assertIn("event: done", response.text)

    def test_approval_and_session(self):
        approval = self.client.post(
            "/api/approvals",
            json={"thread_id": "ticket-1", "approved": True},
        )
        session = self.client.get("/api/sessions/ticket-1")
        missing = self.client.get("/api/sessions/missing")

        self.assertEqual(approval.json()["answer"], "已确认")
        self.assertEqual(self.service.resume_calls, [("ticket-1", True)])
        self.assertEqual(session.status_code, 200)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
