"""可观测性、重试和离线评估的确定性测试。"""

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from lesson_04_langchain_basics.tools import calculator_tool
from lesson_06_agent_quality.evaluation import (
    evaluate_case,
    evaluate_dataset,
    run_to_record,
)
from lesson_06_agent_quality.observability import EventLogger, request_scope
from lesson_06_agent_quality.quality_agent import QualityLangGraphAgent
from lesson_06_agent_quality.reliability import ObservedRetryModel, RetryPolicy


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AgentQualityTests(unittest.TestCase):
    def test_logger_adds_context_and_redacts_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = EventLogger(Path(directory) / "events.jsonl")
            with request_scope("thread-a", "req-fixed"):
                record = logger.log("example", api_key="secret", value=3)

            self.assertEqual(record["request_id"], "req-fixed")
            self.assertEqual(record["thread_id"], "thread-a")
            self.assertEqual(record["api_key"], "***REDACTED***")
            self.assertTrue(logger.path.read_text(encoding="utf-8").endswith("\n"))

    def test_transient_model_error_retries_then_succeeds(self):
        model = ScriptedModel(
            [TimeoutError("temporary"), AIMessage(content="成功")]
        )
        logger = EventLogger()
        wrapper = ObservedRetryModel(
            model,
            logger,
            RetryPolicy(max_attempts=2, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        result = wrapper.invoke([])

        self.assertEqual(result.content, "成功")
        self.assertEqual(model.calls, 2)
        self.assertIn("model_retry", [event["event"] for event in logger.events])

    def test_non_transient_model_error_does_not_retry(self):
        model = ScriptedModel([ValueError("bad input")])
        logger = EventLogger()
        wrapper = ObservedRetryModel(
            model,
            logger,
            RetryPolicy(max_attempts=3, base_delay_seconds=0),
        )
        with self.assertRaises(ValueError):
            wrapper.invoke([])
        self.assertEqual(model.calls, 1)

    def test_quality_agent_logs_request_model_and_tool(self):
        tool_call = {
            "name": "calculator",
            "args": {"expression": "20*5"},
            "id": "calc-1",
            "type": "tool_call",
        }
        model = ScriptedModel(
            [
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content="答案是100。"),
            ]
        )
        logger = EventLogger()
        agent = QualityLangGraphAgent(
            model,
            [calculator_tool],
            logger,
            InMemorySaver(),
            retry_policy=RetryPolicy(max_attempts=1),
        )
        result = agent.ask("计算20乘5", "quality", request_id="req-test")

        events = [item["event"] for item in logger.events]
        self.assertEqual(result.request_id, "req-test")
        self.assertEqual(events.count("model_finished"), 2)
        self.assertIn("tool_finished", events)
        self.assertEqual(events[0], "request_started")
        self.assertEqual(events[-1], "request_finished")

    def test_evaluation_detects_invalid_citation(self):
        case = {
            "id": "c1",
            "expected_tools": ["search"],
            "expected_sources": ["policy.pdf"],
            "expected_keywords": ["500"],
            "should_refuse": False,
        }
        run = {
            "id": "c1",
            "actual_tools": ["search"],
            "retrieved_sources": ["policy.pdf"],
            "answer": "标准是500元[E9]",
            "citations": ["E9"],
            "available_evidence_ids": ["E1"],
            "refused": False,
        }
        metrics = evaluate_case(case, run)
        self.assertFalse(metrics.citations_valid)
        self.assertFalse(metrics.passed)

    def test_dataset_summary_calculates_mrr_and_tokens(self):
        cases = [
            {
                "id": "c1",
                "expected_tools": ["search"],
                "expected_sources": ["right.pdf"],
                "expected_keywords": [],
                "should_refuse": False,
            }
        ]
        runs = [
            {
                "id": "c1",
                "actual_tools": ["search"],
                "retrieved_sources": ["wrong.pdf", "right.pdf"],
                "answer": "答案",
                "citations": ["E1"],
                "available_evidence_ids": ["E1"],
                "refused": False,
                "latency_ms": 100,
                "total_tokens": 20,
            }
        ]
        summary = evaluate_dataset(cases, runs)["summary"]
        self.assertEqual(summary["mrr"], 0.5)
        self.assertEqual(summary["total_tokens"], 20)

    def test_real_result_is_converted_to_evaluation_record(self):
        result = type(
            "Result",
            (),
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "calculator",
                                "args": {"expression": "2+2"},
                                "id": "c1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
                "answer": "答案是4",
                "request_id": "req-1",
            },
        )()
        events = [
            {"event": "model_finished", "request_id": "req-1", "total_tokens": 8},
            {"event": "request_finished", "request_id": "req-1", "duration_ms": 12},
        ]
        record = run_to_record({"id": "math"}, result, events)
        self.assertEqual(record["actual_tools"], ["calculator"])
        self.assertEqual(record["total_tokens"], 8)
        self.assertEqual(record["latency_ms"], 12)


if __name__ == "__main__":
    unittest.main()
