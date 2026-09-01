"""模型调用的错误分类、指数退避重试与观测包装器。"""

import time
from dataclasses import dataclass


TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts必须至少为1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("重试等待时间不能为负数")

    def delay(self, failed_attempt):
        return min(
            self.base_delay_seconds * (2 ** (failed_attempt - 1)),
            self.max_delay_seconds,
        )


def is_transient_error(error):
    """只把超时、连接错误、限流和服务端错误判定为可重试。"""

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return status_code in TRANSIENT_STATUS_CODES


def usage_from_message(message):
    """兼容LangChain的usage_metadata与OpenAI响应元数据。"""

    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(message, "response_metadata", None) or {}
        usage = metadata.get("token_usage", {})
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens) or 0
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


class ObservedRetryModel:
    """为任意支持bind_tools/invoke的模型增加日志和有限重试。"""

    def __init__(self, model, logger, retry_policy=None, sleep=time.sleep):
        self.model = model
        self.logger = logger
        self.retry_policy = retry_policy or RetryPolicy()
        self.sleep = sleep

    def bind_tools(self, tools):
        return ObservedRetryModel(
            self.model.bind_tools(tools),
            self.logger,
            self.retry_policy,
            self.sleep,
        )

    def invoke(self, messages):
        last_error = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            started = time.perf_counter()
            self.logger.log(
                "model_started",
                attempt=attempt,
                message_count=len(messages),
            )
            try:
                message = self.model.invoke(messages)
            except Exception as error:
                last_error = error
                retryable = is_transient_error(error)
                self.logger.log(
                    "model_failed",
                    attempt=attempt,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_type=type(error).__name__,
                    retryable=retryable,
                )
                if not retryable or attempt >= self.retry_policy.max_attempts:
                    raise
                delay = self.retry_policy.delay(attempt)
                self.logger.log(
                    "model_retry",
                    next_attempt=attempt + 1,
                    delay_seconds=delay,
                )
                self.sleep(delay)
                continue

            usage = usage_from_message(message)
            self.logger.log(
                "model_finished",
                attempt=attempt,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                tool_calls=len(getattr(message, "tool_calls", []) or []),
                **usage,
            )
            return message
        raise last_error
