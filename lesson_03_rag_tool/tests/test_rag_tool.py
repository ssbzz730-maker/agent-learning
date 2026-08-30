"""完整 RAG 工具的离线测试，不依赖 Chroma 和模型下载。"""

import tempfile
import unittest
from pathlib import Path

from lesson_03_rag_tool.agent_app.tools.rag_tool import HybridRAGTool
from lesson_03_rag_tool.rag.build_index import collect_records, split_text


class FakeBackend:
    """用固定子块和分数模拟向量库。"""

    def __init__(self):
        self.children = [
            {
                "id": "c-travel",
                "document": "上海住宿标准为每人每晚500元",
                "metadata": {"parent_id": "p-travel", "source": "policy.md"},
            },
            {
                "id": "c-meeting",
                "document": "取消会议室预约应至少提前30分钟",
                "metadata": {"parent_id": "p-meeting", "source": "office.md"},
            },
            {
                "id": "c-error",
                "document": "错误代码ZX-42表示网关连接失败",
                "metadata": {"parent_id": "p-error", "source": "it.md"},
            },
        ]
        self.parents = {
            "p-travel": {
                "id": "p-travel",
                "document": "上海地区出差住宿标准为每人每晚不超过500元。",
                "metadata": {
                    "source": "policy.md",
                    "heading": "差旅报销",
                    "file_type": "md",
                },
            },
            "p-meeting": {
                "id": "p-meeting",
                "document": "取消会议室预约应至少提前30分钟。",
                "metadata": {
                    "source": "office.md",
                    "heading": "会议室管理",
                    "file_type": "md",
                },
            },
            "p-error": {
                "id": "p-error",
                "document": "错误代码ZX-42表示网关连接失败，应联系网络管理员。",
                "metadata": {"source": "it.md", "page": 7, "file_type": "md"},
            },
        }

    def available_sources(self):
        return sorted({item["metadata"]["source"] for item in self.children})

    def list_children(self, source=None):
        return [
            item
            for item in self.children
            if source is None or item["metadata"]["source"] == source
        ]

    def vector_search(self, query, recall_k, source=None):
        if "住宿" in query or "上海" in query:
            scores = {"c-travel": 0.92, "c-meeting": 0.18, "c-error": 0.05}
        elif "会议室" in query or "规定" in query:
            scores = {"c-travel": 0.62, "c-meeting": 0.64, "c-error": 0.08}
        elif "ZX-42" in query:
            scores = {"c-travel": 0.05, "c-meeting": 0.08, "c-error": 0.35}
        else:
            scores = {item["id"]: 0.05 for item in self.children}
        records = [
            {**item, "similarity": scores[item["id"]]}
            for item in self.list_children(source)
        ]
        records.sort(key=lambda item: item["similarity"], reverse=True)
        return records[:recall_k]

    def get_parents(self, parent_ids):
        return {item_id: self.parents[item_id] for item_id in parent_ids}


class FakeReranker:
    """把会议室文本排到其他候选之前。"""

    def score(self, query, documents):
        return [10.0 if "会议室" in document else 1.0 for document in documents]


class HybridRAGToolTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()

    def test_hybrid_search_returns_structured_parent_evidence(self):
        tool = HybridRAGTool(self.backend)
        result = tool.search("上海住宿一晚最多报销多少？", top_k=1)

        self.assertTrue(result["success"])
        self.assertIsNone(result["message"])
        evidence = result["results"][0]
        self.assertEqual(evidence["evidence_id"], "E1")
        self.assertEqual(evidence["source"], "policy.md")
        self.assertIn("500元", evidence["content"])
        self.assertIn("similarity", evidence["scores"])

    def test_exact_keyword_can_survive_low_vector_similarity(self):
        tool = HybridRAGTool(self.backend)
        result = tool.search("ZX-42", top_k=1)

        self.assertEqual(result["results"][0]["source"], "it.md")
        self.assertEqual(result["results"][0]["page"], 7)

    def test_unrelated_query_returns_no_evidence(self):
        tool = HybridRAGTool(self.backend)
        result = tool.search("今天食堂供应什么水果？")

        self.assertTrue(result["success"])
        self.assertEqual(result["results"], [])
        self.assertIn("没有找到足够可靠", result["message"])

    def test_source_filter_and_missing_source(self):
        tool = HybridRAGTool(self.backend)
        filtered = tool.search("取消会议室", source="office.md")
        missing = tool.search("取消会议室", source="missing.pdf")

        self.assertEqual(filtered["results"][0]["source"], "office.md")
        self.assertEqual(missing["results"], [])
        self.assertIn("不存在指定来源", missing["message"])

    def test_reranker_changes_final_order(self):
        tool = HybridRAGTool(self.backend, reranker=FakeReranker())
        result = tool.search("公司的规定是什么？", top_k=2)

        self.assertEqual(result["results"][0]["source"], "office.md")
        self.assertEqual(result["results"][0]["scores"]["reranker"], 10.0)

    def test_markdown_ingestion_creates_parent_and_child_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "policy.md"
            path.write_text("# 规定\n" + "报销标准500元。" * 30, encoding="utf-8")
            parents, children = collect_records(
                root,
                parent_size=80,
                parent_overlap=10,
                child_size=30,
                child_overlap=5,
            )

        self.assertGreater(len(parents), 1)
        self.assertGreater(len(children), len(parents))
        self.assertEqual(parents[0]["metadata"]["source"], "policy.md")
        self.assertIn("parent_id", children[0]["metadata"])

    def test_split_text_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("内容", chunk_size=20, overlap=20)


if __name__ == "__main__":
    unittest.main()
