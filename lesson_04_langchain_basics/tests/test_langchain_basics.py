"""LangChain Prompt、Runnable 和 Tool 的离线测试。"""

import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lesson_04_langchain_basics.chains import build_answer_chain, normalize_input
from lesson_04_langchain_basics.manual_agent import AgentLoopError, ManualToolAgent
from lesson_04_langchain_basics.prompts import (
    build_conversation_prompt,
    render_basic_messages,
)
from lesson_04_langchain_basics.tools import (
    build_rag_search_tool,
    calculator_tool,
    tool_summary,
)


class FakeRAGEngine:
    """记录 LangChain Tool 传入的参数，并返回固定证据。"""

    def __init__(self):
        self.calls = []

    def search(self, query, source=None, top_k=3):
        self.calls.append({"query": query, "source": source, "top_k": top_k})
        return {
            "success": True,
            "results": [
                {
                    "evidence_id": "E1",
                    "content": "上海住宿标准为每晚500元。",
                    "source": source or "policy.md",
                }
            ],
            "message": None,
        }


class ScriptedToolModel:
    """按顺序返回AIMessage，并记录bind_tools和invoke收到的数据。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = []
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("假模型没有剩余响应")
        return self.responses.pop(0)


class LangChainBasicsTests(unittest.TestCase):
    def test_prompt_formats_system_history_and_user_messages(self):
        """Prompt 应把输入变量转换成有角色的消息对象。"""

        prompt = build_conversation_prompt()
        value = prompt.invoke(
            {
                "history": [
                    HumanMessage(content="上一问"),
                    AIMessage(content="上一答"),
                ],
                "question": "当前问题",
            }
        )
        messages = value.to_messages()

        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[1].content, "上一问")
        self.assertEqual(messages[2].content, "上一答")
        self.assertEqual(messages[3].content, "当前问题")

    def test_render_basic_messages_trims_question(self):
        messages = render_basic_messages("  什么是RAG？  ")
        self.assertEqual(messages[-1].content, "什么是RAG？")

    def test_runnable_chain_invoke_returns_plain_string(self):
        """Prompt输出进入假模型，AIMessage再由Parser转换成字符串。"""

        model = FakeListChatModel(responses=["这是离线假模型答案。"])
        chain = build_answer_chain(model)
        result = chain.invoke({"question": " 测试问题 "})

        self.assertEqual(result, "这是离线假模型答案。")
        self.assertIsInstance(result, str)

    def test_runnable_chain_supports_batch(self):
        model = FakeListChatModel(responses=["答案A", "答案B"])
        chain = build_answer_chain(model)
        results = chain.batch(
            [{"question": "问题A"}, {"question": "问题B"}],
            config={"max_concurrency": 1},
        )
        self.assertEqual(results, ["答案A", "答案B"])

    def test_runnable_chain_supports_stream(self):
        model = FakeListChatModel(responses=["流式答案"])
        chain = build_answer_chain(model)
        chunks = list(chain.stream({"question": "问题"}))

        self.assertEqual("".join(chunks), "流式答案")
        self.assertGreater(len(chunks), 1)

    def test_normalize_input_rejects_invalid_payload(self):
        with self.assertRaises(TypeError):
            normalize_input("不是字典")
        with self.assertRaises(ValueError):
            normalize_input({"question": "   "})

    def test_calculator_tool_has_schema_and_uses_safe_function(self):
        result = calculator_tool.invoke({"expression": "500 * 3"})
        summary = tool_summary(calculator_tool)

        self.assertEqual(result["result"], 1500)
        self.assertEqual(summary["name"], "calculator")
        self.assertIn("expression", summary["args_schema"]["properties"])
        with self.assertRaises(ValueError):
            calculator_tool.invoke(
                {"expression": "__import__('os').system('whoami')"}
            )

    def test_rag_engine_is_adapted_to_structured_tool(self):
        engine = FakeRAGEngine()
        rag_tool = build_rag_search_tool(engine)
        result = rag_tool.invoke(
            {"query": "上海住宿标准？", "source": "travel.md", "top_k": 1}
        )

        self.assertEqual(rag_tool.name, "search_company_documents")
        self.assertEqual(result["results"][0]["evidence_id"], "E1")
        self.assertEqual(
            engine.calls,
            [{"query": "上海住宿标准？", "source": "travel.md", "top_k": 1}],
        )
        schema = rag_tool.args_schema.model_json_schema()
        self.assertEqual(schema["properties"]["top_k"]["maximum"], 5)

    def test_rag_tool_schema_rejects_invalid_top_k(self):
        rag_tool = build_rag_search_tool(FakeRAGEngine())
        with self.assertRaises(Exception):
            rag_tool.invoke({"query": "问题", "top_k": 10})

    def test_manual_agent_binds_executes_and_returns_tool_result(self):
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "500 * 3"},
                            "id": "calc-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="三晚合计1500元。"),
            ]
        )
        agent = ManualToolAgent(model, [calculator_tool])
        result = agent.ask("计算500乘3", show_trace=False)

        self.assertEqual([tool.name for tool in model.bound_tools], ["calculator"])
        self.assertEqual(result.answer, "三晚合计1500元。")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.tool_events[0]["result"]["result"], 1500)
        returned = [item for item in model.calls[1] if isinstance(item, ToolMessage)]
        self.assertEqual(returned[0].tool_call_id, "calc-1")
        self.assertIn("1500", returned[0].content)

    def test_manual_agent_can_use_rag_then_calculator(self):
        rag_tool = build_rag_search_tool(FakeRAGEngine())
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_company_documents",
                            "args": {"query": "上海住宿标准？", "top_k": 1},
                            "id": "rag-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "500 * 3"},
                            "id": "calc-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="根据[E1]，三晚最多1500元。"),
            ]
        )
        result = ManualToolAgent(model, [rag_tool, calculator_tool]).ask(
            "上海出差住三晚最多报销多少？",
            show_trace=False,
        )

        self.assertEqual(result.steps, 3)
        self.assertEqual(
            [event["name"] for event in result.tool_events],
            ["search_company_documents", "calculator"],
        )
        self.assertIn("1500", result.answer)

    def test_manual_agent_returns_unknown_tool_error_to_model(self):
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delete_files",
                            "args": {"path": "C:/"},
                            "id": "bad-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="该工具未注册，无法执行。"),
            ]
        )
        result = ManualToolAgent(model, [calculator_tool]).ask(
            "删除文件", show_trace=False
        )

        self.assertFalse(result.tool_events[0]["success"])
        self.assertIn("未注册工具", result.tool_events[0]["error"])

    def test_manual_agent_stops_repeated_tool_loop(self):
        repeated_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": "1 + 1"},
                    "id": "repeat-1",
                    "type": "tool_call",
                }
            ],
        )
        agent = ManualToolAgent(ScriptedToolModel([repeated_call]), [calculator_tool])
        with self.assertRaises(AgentLoopError):
            agent.ask("计算", max_steps=1, show_trace=False)


if __name__ == "__main__":
    unittest.main()
