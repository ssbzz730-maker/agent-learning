"""原生 Agent 的离线单元测试。

这些测试不会调用 DeepSeek。测试代码使用 ``SimpleNamespace`` 构造假的
SDK 响应，让 Agent 误以为自己收到了真实模型返回的 ``message`` 和
``tool_calls``。这样可以稳定、快速地验证工具安全性和 Agent 控制循环，
同时不会消耗 API 额度。
"""

import json
import unittest
from types import SimpleNamespace

from lesson_01_native_tool_calling.agent_app.agent import NativeAgent
from lesson_01_native_tool_calling.agent_app.tools.calculator import calculate
from lesson_01_native_tool_calling.agent_app.tools.knowledge_base import KnowledgeBase


def tool_call(call_id, name, arguments):
    """创建一个与 OpenAI SDK 属性结构相同的假工具调用。

    Args:
        call_id: 本次工具调用的唯一标识，工具结果要使用同一个 ID。
        name: 模型选择的工具名称，例如 ``calculator``。
        arguments: Python 字典形式的工具参数。

    Returns:
        一个支持 ``call.id``、``call.function.name`` 和
        ``call.function.arguments`` 属性访问的简单对象。

    真实模型返回的 arguments 是 JSON 字符串，因此这里使用
    ``json.dumps`` 把测试中的 Python 字典转换成 JSON，以模拟真实响应。
    """

    return SimpleNamespace(
        id=call_id,
        # function 也是一个带 name 和 arguments 属性的嵌套对象。
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeCompletions:
    """模拟 ``client.chat.completions``，按顺序返回预设模型消息。"""

    def __init__(self, messages):
        """保存预设响应，并准备记录 Agent 发来的所有模型请求。"""

        # 转成迭代器后，每调用一次 create() 就取出下一条响应。
        self.responses = iter(messages)

        # 保存 create() 收到的参数，必要时可检查 messages、tools 等内容。
        self.requests = []

    def create(self, **kwargs):
        """模拟 SDK 的 create() 方法并返回下一条假响应。"""

        # 先记录请求，方便测试确认 Agent 给模型发送了什么。
        self.requests.append(kwargs)

        # 模拟模型按轮次返回消息。预设响应耗尽时 next() 会抛出异常，
        # 提醒我们 Agent 调用模型的次数超过了测试预期。
        message = next(self.responses)

        # NativeAgent 使用 response.choices[0].message 读取模型消息，
        # 所以假响应必须保持相同的嵌套属性结构。
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentTests(unittest.TestCase):
    """验证工具本身以及 Agent 的工具选择、执行和回传流程。"""

    def test_safe_calculator(self):
        """计算器应能完成正常运算，同时拒绝执行任意 Python 代码。"""

        # 正常路径：基础乘法应得到正确结果。
        self.assertEqual(calculate("500 * 3"), 1500)

        # 安全路径：表达式试图导入 os 并运行系统命令。
        # 计算器使用 AST 白名单而不是 eval，因此必须抛出 ValueError。
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('whoami')")

    def test_knowledge_search(self):
        """知识库应把会议室取消规定排在第一名。"""

        # Arrange + Act：加载示例知识库并只取相似度最高的一条结果。
        results = KnowledgeBase().search("取消会议室需要提前多久？", top_k=1)

        # Assert：不仅要命中文档 ID，还要确认内容包含真正的答案。
        # 只检查 ID 可能出现“文件正确但具体内容错误”的漏测。
        self.assertEqual(results[0]["id"], "meeting-room")
        self.assertIn("三十分钟", results[0]["content"])

    def test_agent_executes_tool_and_returns_answer(self):
        """Agent 应执行模型选择的工具，并在第二轮返回最终答案。"""

        # Arrange，第一个模型响应：没有文本答案，而是要求调用计算器。
        # call_1 是工具调用 ID，后面的 tool 消息必须使用相同 ID。
        first = SimpleNamespace(
            content=None,
            tool_calls=[tool_call("call_1", "calculator", {"expression": "125*0.8"})],
        )

        # Arrange，第二个模型响应：模型已经看到计算结果，因此不再调用
        # 工具，而是通过 content 返回最终答案。
        second = SimpleNamespace(content="打八折后是100元。", tool_calls=None)

        # FakeCompletions 会让两次模型调用依次拿到 first 和 second。
        completions = FakeCompletions([first, second])

        # 构造 client.chat.completions.create 的相同属性层级，
        # 使 NativeAgent 无需知道自己连接的是假客户端。
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        # Act：运行 Agent。关闭轨迹输出，避免测试终端出现调试文字。
        result = NativeAgent(client=client).run("125元打八折是多少？", show_trace=False)

        # Assert 1：最终答案应来自第二轮模型响应。
        self.assertEqual(result.answer, "打八折后是100元。")

        # Assert 2：这个任务应调用模型两轮：第一轮选工具，第二轮回答。
        self.assertEqual(result.steps, 2)

        # 从完整消息历史中取出 role=tool 的消息，检查控制器是否真的
        # 执行了计算器，并把结果作为工具消息追加到了 messages。
        tool_messages = [item for item in result.messages if item["role"] == "tool"]

        # tool 消息的 content 是 JSON 字符串，解析后结果应为 100.0。
        self.assertEqual(json.loads(tool_messages[0]["content"])["result"], 100.0)

        # 工具结果必须通过 tool_call_id 与 first 中的 call_1 对应。
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")

    def test_unknown_tool_returns_error_to_model(self):
        """模型请求未注册工具时，程序应拒绝执行并回传错误信息。"""

        # Arrange：模拟模型尝试调用危险且未注册的 delete_files 工具。
        first = SimpleNamespace(
            content=None,
            tool_calls=[tool_call("call_bad", "delete_files", {"path": "C:/"})],
        )

        # 第二轮模拟模型阅读错误结果后，向用户说明工具不被允许。
        second = SimpleNamespace(content="该工具不被允许。", tool_calls=None)
        completions = FakeCompletions([first, second])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        # Act：运行任务。注册表应拦截 delete_files，而不是执行它。
        result = NativeAgent(client=client).run("删除文件", show_trace=False)

        # Assert：找到控制器回传给模型的 tool 消息。
        tool_message = next(
            item for item in result.messages if item["role"] == "tool"
        )

        # 工具异常应该转成可供模型阅读的错误结果，而不是让程序崩溃。
        self.assertIn("不允许调用未知工具", tool_message["content"])


if __name__ == "__main__":
    # 允许直接执行本文件；通常推荐从项目根目录使用 python -m unittest。
    unittest.main()
