"""可复现的离线Agent评估指标与JSON报告。"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CaseMetrics:
    case_id: str
    tool_precision: float
    tool_recall: float
    tool_order_correct: bool
    source_hit: bool | None
    reciprocal_rank: float | None
    keyword_score: float
    refusal_correct: bool
    citations_valid: bool
    passed: bool


def load_records(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}顶层必须是JSON数组")
    return data


def _precision_recall(expected, actual):
    expected_set = set(expected)
    actual_set = set(actual)
    correct = len(expected_set & actual_set)
    precision = correct / len(actual_set) if actual_set else float(not expected_set)
    recall = correct / len(expected_set) if expected_set else float(not actual_set)
    return precision, recall


def evaluate_case(case, run):
    """同时评估工具、来源排名、关键词、拒答和引用真实性。"""

    expected_tools = case.get("expected_tools", [])
    actual_tools = run.get("actual_tools", [])
    tool_precision, tool_recall = _precision_recall(expected_tools, actual_tools)
    tool_order_correct = actual_tools == expected_tools

    expected_sources = case.get("expected_sources", [])
    actual_sources = run.get("retrieved_sources", [])
    if expected_sources:
        ranks = [
            actual_sources.index(source) + 1
            for source in expected_sources
            if source in actual_sources
        ]
        source_hit = bool(ranks)
        reciprocal_rank = 1 / min(ranks) if ranks else 0.0
    else:
        source_hit = None
        reciprocal_rank = None

    answer = str(run.get("answer", "")).lower()
    keywords = [str(item).lower() for item in case.get("expected_keywords", [])]
    keyword_score = (
        sum(keyword in answer for keyword in keywords) / len(keywords)
        if keywords
        else 1.0
    )
    refusal_correct = bool(run.get("refused", False)) == bool(
        case.get("should_refuse", False)
    )
    citations = set(run.get("citations", []))
    evidence_ids = set(run.get("available_evidence_ids", []))
    citations_valid = citations <= evidence_ids and (
        bool(citations) if expected_sources else True
    )

    required_checks = [
        tool_recall == 1.0,
        keyword_score == 1.0,
        refusal_correct,
        citations_valid,
    ]
    if source_hit is not None:
        required_checks.append(source_hit)
    return CaseMetrics(
        case_id=case["id"],
        tool_precision=round(tool_precision, 4),
        tool_recall=round(tool_recall, 4),
        tool_order_correct=tool_order_correct,
        source_hit=source_hit,
        reciprocal_rank=reciprocal_rank,
        keyword_score=round(keyword_score, 4),
        refusal_correct=refusal_correct,
        citations_valid=citations_valid,
        passed=all(required_checks),
    )


def _average(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def evaluate_dataset(cases, runs):
    """按case id对齐数据并生成明细与汇总，拒绝静默遗漏案例。"""

    run_map = {run["id"]: run for run in runs}
    missing = [case["id"] for case in cases if case["id"] not in run_map]
    if missing:
        raise ValueError(f"缺少运行结果：{', '.join(missing)}")
    details = [evaluate_case(case, run_map[case["id"]]) for case in cases]
    summary = {
        "case_count": len(details),
        "pass_rate": _average([float(item.passed) for item in details]),
        "tool_precision": _average([item.tool_precision for item in details]),
        "tool_recall": _average([item.tool_recall for item in details]),
        "tool_order_accuracy": _average(
            [float(item.tool_order_correct) for item in details]
        ),
        "retrieval_hit_rate": _average(
            [
                float(item.source_hit) if item.source_hit is not None else None
                for item in details
            ]
        ),
        "mrr": _average([item.reciprocal_rank for item in details]),
        "keyword_score": _average([item.keyword_score for item in details]),
        "refusal_accuracy": _average([float(item.refusal_correct) for item in details]),
        "citation_validity": _average(
            [float(item.citations_valid) for item in details]
        ),
        "average_latency_ms": _average(
            [run_map[item.case_id].get("latency_ms") for item in details]
        ),
        "total_tokens": sum(
            run_map[item.case_id].get("total_tokens", 0) for item in details
        ),
    }
    return {"summary": summary, "details": [asdict(item) for item in details]}


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_to_record(case, result, events):
    """把一次真实Agent结果转换成可复现的评估运行记录。"""

    actual_tools = []
    retrieved_sources = []
    evidence_ids = []
    for message in result.messages:
        for call in getattr(message, "tool_calls", []) or []:
            actual_tools.append(call["name"])
        if getattr(message, "type", None) != "tool":
            continue
        if getattr(message, "name", None) != "search_company_documents":
            continue
        try:
            payload = json.loads(message.content)
            rag_result = payload.get("result", {})
            for item in rag_result.get("results", []):
                source = item.get("source")
                evidence_id = item.get("evidence_id")
                if source and source not in retrieved_sources:
                    retrieved_sources.append(source)
                if evidence_id and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        except (TypeError, json.JSONDecodeError):
            continue

    answer = result.answer or ""
    citations = re.findall(r"\[([A-Za-z]+\d+)\]", answer)
    refusal_phrases = ("无法确认", "没有相关规定", "没有足够证据", "无法回答")
    request_events = [
        event for event in events if event.get("request_id") == result.request_id
    ]
    total_tokens = sum(
        event.get("total_tokens", 0)
        for event in request_events
        if event["event"] == "model_finished"
    )
    finished = [
        event for event in request_events if event["event"] == "request_finished"
    ]
    return {
        "id": case["id"],
        "actual_tools": actual_tools,
        "retrieved_sources": retrieved_sources,
        "available_evidence_ids": evidence_ids,
        "citations": citations,
        "answer": answer,
        "refused": any(phrase in answer for phrase in refusal_phrases),
        "latency_ms": finished[-1].get("duration_ms") if finished else None,
        "total_tokens": total_tokens,
        "request_id": result.request_id,
    }


def run_live_dataset(agent, cases, logger, thread_prefix="eval"):
    """逐条真实调用Agent；每个案例使用独立thread_id避免历史污染。"""

    runs = []
    for case in cases:
        result = agent.ask(
            case["question"],
            thread_id=f"{thread_prefix}-{case['id']}",
        )
        if result.status != "completed":
            raise RuntimeError(f"评估案例{case['id']}意外进入{result.status}")
        runs.append(run_to_record(case, result, logger.events))
    return runs
