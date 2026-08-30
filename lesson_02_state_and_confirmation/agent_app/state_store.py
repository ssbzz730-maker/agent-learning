"""使用独立 JSON 文件持久化每个 Agent 会话。"""

import json
import re
from copy import deepcopy
from pathlib import Path


DEFAULT_STATE_DIR = Path(__file__).parents[1] / "agent_states"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class StateStore:
    """按 ``session_id`` 隔离并原子保存会话状态。"""

    def __init__(self, directory=DEFAULT_STATE_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_session_id(session_id):
        """限制会话 ID，避免 ``../`` 等路径穿越。"""

        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(
            session_id
        ):
            raise ValueError(
                "session_id 只能包含字母、数字、下划线和连字符，长度为 1 到 64"
            )
        return session_id

    def _path(self, session_id):
        return self.directory / f"{self.validate_session_id(session_id)}.json"

    def exists(self, session_id):
        """判断指定会话是否已经保存。"""

        return self._path(session_id).exists()

    def load(self, session_id):
        """读取会话并返回副本；不存在时返回 ``None``。"""

        path = self._path(session_id)
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("session_id") != session_id:
            raise ValueError("状态文件中的 session_id 不一致")
        return deepcopy(state)

    def save(self, state):
        """先写临时文件再替换，避免中途退出留下半个 JSON。"""

        if not isinstance(state, dict):
            raise TypeError("state 必须是字典")
        session_id = self.validate_session_id(state.get("session_id"))
        path = self._path(session_id)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return deepcopy(state)
