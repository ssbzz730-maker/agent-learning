"""结构化JSON日志、请求上下文和敏感字段脱敏。"""

import json
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SENSITIVE_KEYS = {"api_key", "authorization", "password", "secret", "token"}


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    thread_id: str


_REQUEST_CONTEXT = ContextVar("agent_request_context", default=None)


@contextmanager
def request_scope(thread_id, request_id=None):
    """在一次ask或resume期间提供统一request_id和thread_id。"""

    context = RequestContext(request_id or f"req-{uuid.uuid4().hex}", thread_id)
    token = _REQUEST_CONTEXT.set(context)
    try:
        yield context
    finally:
        _REQUEST_CONTEXT.reset(token)


def current_request_context():
    return _REQUEST_CONTEXT.get()


def redact(value):
    """递归隐藏Key、密码和Authorization等敏感字段。"""

    if isinstance(value, dict):
        return {
            key: "***REDACTED***"
            if any(fragment in key.lower() for fragment in SENSITIVE_KEYS)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class EventLogger:
    """把同一事件同时保存在内存，并可选择追加到JSONL文件。"""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.events = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event, **fields):
        context = current_request_context()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **(asdict(context) if context else {}),
            **redact(fields),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.events.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        return record
