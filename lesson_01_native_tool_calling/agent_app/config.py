"""集中读取模型配置，避免在代码中写入真实密钥。"""

import os


DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_api_key():
    """读取当前进程继承的 DeepSeek API Key。"""

    return os.getenv("DEEPSEEK_API_KEY")
