"""在第5课LangGraph Agent上增加请求、模型和工具可观测性。"""

import time

from lesson_05_langgraph_agent.graph_agent import LangGraphAgent
from lesson_06_agent_quality.observability import request_scope
from lesson_06_agent_quality.reliability import ObservedRetryModel, RetryPolicy


class QualityLangGraphAgent(LangGraphAgent):
    """保留原状态图，只横向加入日志、Token、延迟和模型重试。"""

    def __init__(
        self,
        model,
        tools,
        logger,
        checkpointer=None,
        risky_tools=None,
        retry_policy=None,
        retry_sleep=time.sleep,
    ):
        self.logger = logger
        observed_model = ObservedRetryModel(
            model,
            logger,
            retry_policy or RetryPolicy(),
            retry_sleep,
        )
        super().__init__(observed_model, tools, checkpointer, risky_tools)

    def _execute_tool(self, call):
        """记录工具耗时；写工具不自动重试，避免重复副作用。"""

        started = time.perf_counter()
        name = call.get("name")
        self.logger.log("tool_started", tool_name=name, call_id=call.get("id"))
        tool_message, event = super()._execute_tool(call)
        self.logger.log(
            "tool_finished",
            tool_name=name,
            call_id=call.get("id"),
            success=event["success"],
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            error=event.get("error"),
        )
        return tool_message, event

    def ask(self, question, thread_id="default", max_steps=6, request_id=None):
        with request_scope(thread_id, request_id) as context:
            started = time.perf_counter()
            self.logger.log(
                "request_started",
                operation="ask",
                question_length=len(question) if isinstance(question, str) else None,
            )
            try:
                result = super().ask(question, thread_id, max_steps)
            except Exception as error:
                self.logger.log(
                    "request_failed",
                    operation="ask",
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_type=type(error).__name__,
                )
                raise
            event = (
                "approval_required"
                if result.status == "awaiting_confirmation"
                else "request_finished"
            )
            self.logger.log(
                event,
                operation="ask",
                status=result.status,
                model_calls=result.model_calls,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            result.request_id = context.request_id
            return result

    def resume(self, thread_id, approved, request_id=None):
        with request_scope(thread_id, request_id) as context:
            started = time.perf_counter()
            self.logger.log("request_started", operation="resume", approved=approved)
            try:
                result = super().resume(thread_id, approved)
            except Exception as error:
                self.logger.log(
                    "request_failed",
                    operation="resume",
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_type=type(error).__name__,
                )
                raise
            self.logger.log(
                "request_finished",
                operation="resume",
                status=result.status,
                model_calls=result.model_calls,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            result.request_id = context.request_id
            return result
