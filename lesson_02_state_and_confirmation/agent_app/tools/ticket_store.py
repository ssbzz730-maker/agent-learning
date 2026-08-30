"""模拟外部工单系统，并使用幂等键防止重复创建。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TICKET_DIR = Path(__file__).parents[2] / "tickets"


class TicketStore:
    """把每张模拟工单保存为独立 JSON 文件。"""

    def __init__(self, directory=DEFAULT_TICKET_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ticket_id(idempotency_key):
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        return f"T-{digest[:10].upper()}"

    def create(self, title, description, priority, idempotency_key):
        """相同幂等键永远返回同一张工单，不执行第二次写入。"""

        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("创建工单必须提供 idempotency_key")
        ticket_id = self._ticket_id(idempotency_key)
        path = self.directory / f"{ticket_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            return {**existing, "created_now": False}

        ticket = {
            "ticket_id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "idempotency_key": idempotency_key,
        }
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(ticket, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(path)
        return {**ticket, "created_now": True}

    def list_tickets(self):
        """读取所有模拟工单，主要用于演示和测试。"""

        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.directory.glob("T-*.json"))
        ]
