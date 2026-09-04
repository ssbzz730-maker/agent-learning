"""第8课FastAPI流式协议、REST兼容和异步会话测试。"""

import asyncio
import unittest

from fastapi.testclient import TestClient

from lesson_08_streaming_agent.api import create_app, event_response


class FakeStreamingService:
    def __init__(self):
        self.chat_calls = []
        self.resume_calls = []

    async def stream_chat(self, question, thread_id, max_steps=6):
        self.chat_calls.append((question, thread_id, max_steps))
        yield {
            "event": "started",
            "data": {
                "request_id": "req-stream",
                "thread_id": thread_id,
                "operation": "ask",
            },
        }
        yield {"event": "tool_requested", "data": {"name": "calculator"}}
        yield {"event": "token", "data": {"text": "流式答案", "message_id": "m1"}}
        yield {"event": "result", "data": self._result(thread_id, "流式答案")}
        yield {"event": "done", "data": {"status": "completed"}}

    async def stream_resume(self, thread_id, approved):
        self.resume_calls.append((thread_id, approved))
        yield {
            "event": "started",
            "data": {
                "request_id": "req-resume",
                "thread_id": thread_id,
                "operation": "resume",
            },
        }
        answer = "已确认" if approved else "已拒绝"
        yield {"event": "token", "data": {"text": answer, "message_id": "m2"}}
        yield {"event": "result", "data": self._result(thread_id, answer)}
        yield {"event": "done", "data": {"status": "completed"}}

    async def get_session(self, thread_id):
        if thread_id == "missing":
            raise KeyError(thread_id)
        return {
            "thread_id": thread_id,
            "next_nodes": [],
            "model_calls": 2,
            "messages": [{"type": "human", "content": "你好"}],
        }

    @staticmethod
    def _result(thread_id, answer):
        return {
            "request_id": "req-stream",
            "thread_id": thread_id,
            "status": "completed",
            "answer": answer,
            "pending_action": None,
            "model_calls": 2,
        }


class StreamingAPITests(unittest.TestCase):
    def setUp(self):
        self.service = FakeStreamingService()
        self.context = TestClient(create_app(self.service))
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_stream_exposes_real_event_types(self):
        response = self.client.post(
            "/api/chat/stream",
            json={"question": "计算", "thread_id": "stream-1", "max_steps": 4},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: tool_requested", response.text)
        self.assertIn("event: token", response.text)
        self.assertIn("event: result", response.text)
        self.assertIn("event: done", response.text)
        self.assertEqual(self.service.chat_calls, [("计算", "stream-1", 4)])

    def test_rest_chat_remains_compatible(self):
        response = self.client.post(
            "/api/chat",
            json={"question": "普通请求", "thread_id": "rest-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "流式答案")

    def test_approval_can_continue_as_stream(self):
        response = self.client.post(
            "/api/approvals/stream",
            json={"thread_id": "approval-1", "approved": True},
        )

        self.assertIn("event: token", response.text)
        self.assertIn("已确认", response.text)
        self.assertEqual(self.service.resume_calls, [("approval-1", True)])

    def test_session_and_chinese_page(self):
        page = self.client.get("/")
        session = self.client.get("/api/sessions/session-1")
        missing = self.client.get("/api/sessions/missing")

        self.assertIn("流式企业知识与工单 Agent", page.text)
        self.assertEqual(session.status_code, 200)
        self.assertEqual(missing.status_code, 404)

    def test_cancelled_source_runs_async_cleanup(self):
        async def cancel_response():
            cleaned = asyncio.Event()

            class ConnectedRequest:
                @staticmethod
                async def is_disconnected():
                    return False

            async def source():
                try:
                    yield {"event": "started", "data": {}}
                    await asyncio.sleep(60)
                finally:
                    cleaned.set()

            response = event_response(ConnectedRequest(), source())
            first = await anext(response)
            await response.aclose()
            await asyncio.wait_for(cleaned.wait(), timeout=1)
            return first

        self.assertIn("event: started", asyncio.run(cancel_response()))


if __name__ == "__main__":
    unittest.main()
