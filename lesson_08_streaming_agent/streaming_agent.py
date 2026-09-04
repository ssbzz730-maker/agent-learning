"""把LangGraph内部消息与节点更新映射成安全的前端事件。"""

import asyncio
import time

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from lesson_06_agent_quality.observability import request_scope
from lesson_06_agent_quality.quality_agent import QualityLangGraphAgent


def message_text(message):
    """兼容字符串和内容块格式，只提取可展示文本。"""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "text_delta"}:
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class StreamingQualityAgent(QualityLangGraphAgent):
    """保留第6课日志与重试，并增加LangGraph异步事件流入口。"""

    async def astream_ask(
        self,
        question,
        thread_id="default",
        max_steps=6,
        request_id=None,
    ):
        """向图加入新问题，并逐条产生公开事件。"""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question必须是非空字符串")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps必须是正整数")
        graph_input = {
            "messages": [HumanMessage(content=question.strip())],
            "model_calls": 0,
            "max_steps": max_steps,
            "approved": None,
            "tool_events": [],
        }
        async for event in self._astream_run(
            graph_input,
            thread_id,
            "ask",
            request_id,
            question_length=len(question),
        ):
            yield event

    async def astream_resume(self, thread_id, approved, request_id=None):
        """从同一thread_id的人工确认中断处恢复，并继续产生事件。"""

        if not isinstance(approved, bool):
            raise TypeError("approved必须是布尔值")
        async for event in self._astream_run(
            Command(resume=approved),
            thread_id,
            "resume",
            request_id,
            approved=approved,
        ):
            yield event

    async def _astream_run(
        self,
        graph_input,
        thread_id,
        operation,
        request_id=None,
        **log_fields,
    ):
        """运行一次图，把内部事件翻译为稳定且不泄露工具结果的公开协议。"""

        config = self._config(thread_id)
        with request_scope(thread_id, request_id) as context:
            started = time.perf_counter()
            self.logger.log("request_started", operation=operation, **log_fields)
            yield self._stream_event(
                "started",
                request_id=context.request_id,
                thread_id=thread_id,
                operation=operation,
            )
            try:
                async for chunk in self.graph.astream(
                    graph_input,
                    config=config,
                    stream_mode=["messages", "updates"],
                    version="v2",
                ):
                    for event in self._public_events(chunk):
                        yield event
                result = await self._async_result(thread_id, context.request_id)
            except asyncio.CancelledError:
                self.logger.log(
                    "request_cancelled",
                    operation=operation,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                )
                raise
            except Exception as error:
                self.logger.log(
                    "request_failed",
                    operation=operation,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_type=type(error).__name__,
                )
                raise

            log_event = (
                "approval_required"
                if result.status == "awaiting_confirmation"
                else "request_finished"
            )
            self.logger.log(
                log_event,
                operation=operation,
                status=result.status,
                model_calls=result.model_calls,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            yield self._stream_event("result", **self._result_payload(result))
            yield self._stream_event("done", status=result.status)

    async def _async_result(self, thread_id, request_id):
        """从异步Checkpointer读取最终状态，并复用第5课结果转换规则。"""

        snapshot = await self.graph.aget_state(self._config(thread_id))
        state = dict(snapshot.values)
        if snapshot.interrupts:
            state["__interrupt__"] = snapshot.interrupts
        result = self._to_result(thread_id, state)
        result.request_id = request_id
        return result

    @classmethod
    def _public_events(cls, chunk):
        """过滤框架内部结构，只公开Token、节点和工具执行摘要。"""

        if chunk.get("type") == "messages":
            message, metadata = chunk["data"]
            if isinstance(message, (AIMessage, AIMessageChunk)):
                text = message_text(message)
                if text:
                    yield cls._stream_event(
                        "token",
                        text=text,
                        message_id=getattr(message, "id", None),
                        node=metadata.get("langgraph_node"),
                    )
            return

        if chunk.get("type") != "updates":
            return
        updates = chunk.get("data", {})
        interrupts = updates.get("__interrupt__", ())
        for item in interrupts:
            value = item.value
            calls = value.get("tool_calls", []) if isinstance(value, dict) else []
            yield cls._stream_event(
                "approval_required",
                tools=[call.get("name") for call in calls],
                count=len(calls),
            )

        for node, update in updates.items():
            if node == "__interrupt__" or not isinstance(update, dict):
                continue
            yield cls._stream_event("node_finished", node=node)
            if node == "agent":
                messages = update.get("messages", [])
                calls = messages[-1].tool_calls if messages else []
                for call in calls:
                    yield cls._stream_event(
                        "tool_requested",
                        name=call.get("name"),
                        call_id=call.get("id"),
                    )
            for tool_event in update.get("tool_events", []):
                yield cls._stream_event(
                    "tool_finished",
                    name=tool_event.get("name"),
                    call_id=tool_event.get("call_id"),
                    success=bool(tool_event.get("success")),
                    error=tool_event.get("error"),
                )

    @staticmethod
    def _stream_event(event, **data):
        return {"event": event, "data": data}

    @staticmethod
    def _result_payload(result):
        return {
            "request_id": result.request_id,
            "thread_id": result.thread_id,
            "status": result.status,
            "answer": result.answer,
            "pending_action": result.pending_action,
            "model_calls": result.model_calls,
        }
