"""混合检索 RAG 工具：向量 + BM25 + RRF + Reranker + 父块取回。"""

import math
import re
from collections import Counter
from pathlib import Path

from lesson_03_rag_tool.rag.build_index import (
    CHILD_COLLECTION_NAME,
    DEFAULT_DATABASE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_CACHE_DIR,
    PARENT_COLLECTION_NAME,
)


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANKER_CACHE_DIR = Path(__file__).parents[2] / ".reranker-cache"


def tokenize(text):
    """把中文切成双字片段，同时保留英文单词和数字。"""

    tokens = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(
                    part[index : index + 2] for index in range(len(part) - 1)
                )
        else:
            tokens.append(part)
    return tokens


class BM25:
    """用于候选召回的精简 BM25 实现。"""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.document_tokens = [tokenize(document) for document in documents]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.average_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0
        )
        document_frequency = Counter()
        for tokens in self.document_tokens:
            document_frequency.update(set(tokens))
        document_count = len(self.document_tokens)
        self.idf = {
            token: math.log(
                1 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

    def score(self, query, document_index):
        """计算查询对指定文档的 BM25 分数。"""

        frequencies = Counter(self.document_tokens[document_index])
        document_length = self.document_lengths[document_index]
        score = 0.0
        for token in tokenize(query):
            frequency = frequencies[token]
            if frequency == 0:
                continue
            denominator = frequency + self.k1 * (
                1
                - self.b
                + self.b * document_length / max(self.average_length, 1)
            )
            score += self.idf.get(token, 0.0) * (
                frequency * (self.k1 + 1) / denominator
            )
        return score


def keyword_coverage(query, document):
    """返回查询词片段被文档覆盖的比例。"""

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_tokens = set(tokenize(document))
    return len(query_tokens & document_tokens) / len(query_tokens)


class CrossEncoderReranker:
    """延迟加载 CrossEncoder，避免离线测试下载模型。"""

    def __init__(
        self,
        model_name=DEFAULT_RERANKER_MODEL,
        cache_dir=DEFAULT_RERANKER_CACHE_DIR,
        local_files_only=False,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("使用 Reranker 需要安装 requirements-rag.txt") from error
        self.model = CrossEncoder(
            model_name,
            cache_folder=str(cache_dir),
            local_files_only=local_files_only,
            max_length=512,
        )

    def score(self, query, documents):
        """同时阅读查询和候选文本，并返回相关性分数。"""

        if not documents:
            return []
        scores = self.model.predict(
            [[query, document] for document in documents],
            batch_size=16,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


class ChromaParentChildBackend:
    """把 Chroma 父子集合适配成混合检索需要的统一接口。"""

    def __init__(
        self,
        database_dir=DEFAULT_DATABASE_DIR,
        model_cache_dir=DEFAULT_MODEL_CACHE_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        local_files_only=False,
    ):
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("请先安装 requirements-rag.txt") from error

        self.client = chromadb.PersistentClient(
            path=str(database_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.parent_collection = self.client.get_collection(
                PARENT_COLLECTION_NAME
            )
            self.child_collection = self.client.get_collection(
                CHILD_COLLECTION_NAME
            )
        except Exception as error:
            raise RuntimeError("未找到父子索引，请先运行 build_index") from error
        self.model = SentenceTransformer(
            embedding_model,
            cache_folder=str(model_cache_dir),
            local_files_only=local_files_only,
        )

    def available_sources(self):
        """列出索引中全部来源文件。"""

        stored = self.child_collection.get(include=["metadatas"])
        return sorted({item["source"] for item in stored["metadatas"]})

    def list_children(self, source=None):
        """读取用于 BM25 的全部子块，可按来源过滤。"""

        kwargs = {"include": ["documents", "metadatas"]}
        if source:
            kwargs["where"] = {"source": source}
        stored = self.child_collection.get(**kwargs)
        return [
            {"id": item_id, "document": document, "metadata": metadata}
            for item_id, document, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        ]

    def vector_search(self, query, recall_k, source=None):
        """在子块集合中执行向量召回。"""

        records = self.list_children(source)
        if not records:
            return []
        vector = self.model.encode(query, normalize_embeddings=True)
        kwargs = {
            "query_embeddings": [vector.tolist()],
            "n_results": min(recall_k, len(records)),
            "include": ["documents", "metadatas", "distances"],
        }
        if source:
            kwargs["where"] = {"source": source}
        results = self.child_collection.query(**kwargs)
        return [
            {
                "id": item_id,
                "document": document,
                "metadata": metadata,
                "similarity": 1 - distance,
            }
            for item_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_parents(self, parent_ids):
        """批量读取父块，并以 parent_id 为键返回。"""

        if not parent_ids:
            return {}
        stored = self.parent_collection.get(
            ids=parent_ids,
            include=["documents", "metadatas"],
        )
        return {
            item_id: {
                "id": item_id,
                "document": document,
                "metadata": metadata,
            }
            for item_id, document, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        }


class HybridRAGTool:
    """执行完整检索流程，并只向 Agent 返回结构化证据。"""

    def __init__(
        self,
        backend,
        reranker=None,
        recall_k=10,
        rrf_k=60,
        min_vector_similarity=0.4,
        min_keyword_coverage=0.3,
    ):
        self.backend = backend
        self.reranker = reranker
        self.recall_k = recall_k
        self.rrf_k = rrf_k
        self.min_vector_similarity = min_vector_similarity
        self.min_keyword_coverage = min_keyword_coverage

    def available_sources(self):
        return self.backend.available_sources()

    def search(self, query, source=None, top_k=3):
        """混合召回、精排、过滤并返回去重后的父块证据。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if source is not None and (
            not isinstance(source, str) or not source.strip()
        ):
            raise ValueError("source 必须是非空字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
            raise ValueError("top_k 必须是 1 到 5 之间的整数")
        query = query.strip()
        source = source.strip() if source else None

        sources = self.available_sources()
        if source and source not in sources:
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": f"知识库中不存在指定来源：{source}",
                "available_sources": sources,
            }
        records = self.backend.list_children(source)
        if not records:
            return self._empty_result(query, "知识库中没有可检索的文本块")

        recall_k = min(self.recall_k, len(records))
        vector_results = self.backend.vector_search(query, recall_k, source)
        bm25 = BM25([record["document"] for record in records])
        keyword_results = []
        for index, record in enumerate(records):
            keyword_results.append(
                {
                    **record,
                    "bm25_score": bm25.score(query, index),
                    "keyword_coverage": keyword_coverage(
                        query, record["document"]
                    ),
                }
            )
        keyword_results.sort(
            key=lambda item: (item["bm25_score"], item["keyword_coverage"]),
            reverse=True,
        )
        keyword_results = keyword_results[:recall_k]

        fused = {}
        for rank, result in enumerate(vector_results, start=1):
            fused[result["id"]] = {
                **result,
                "vector_rank": rank,
                "bm25_rank": None,
                "bm25_score": 0.0,
                "keyword_coverage": keyword_coverage(
                    query, result["document"]
                ),
                "rrf_score": 1 / (self.rrf_k + rank),
            }
        for rank, result in enumerate(keyword_results, start=1):
            item = fused.setdefault(
                result["id"],
                {
                    **result,
                    "similarity": None,
                    "vector_rank": None,
                    "rrf_score": 0.0,
                },
            )
            item["bm25_rank"] = rank
            item["bm25_score"] = result["bm25_score"]
            item["keyword_coverage"] = result["keyword_coverage"]
            item["rrf_score"] += 1 / (self.rrf_k + rank)

        candidates = sorted(
            fused.values(), key=lambda item: item["rrf_score"], reverse=True
        )
        if self.reranker:
            scores = self.reranker.score(
                query, [item["document"] for item in candidates]
            )
            if len(scores) != len(candidates):
                raise ValueError("Reranker 返回的分数数量与候选数量不一致")
            for item, score in zip(candidates, scores):
                item["reranker_score"] = float(score)
            candidates.sort(
                key=lambda item: item["reranker_score"], reverse=True
            )

        accepted = [
            item
            for item in candidates
            if (
                item.get("similarity") is not None
                and item["similarity"] >= self.min_vector_similarity
            )
            or item.get("keyword_coverage", 0.0) >= self.min_keyword_coverage
        ]
        selected = []
        seen_parent_ids = set()
        for item in accepted:
            parent_id = item["metadata"].get("parent_id", item["id"])
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
            selected.append((parent_id, item))
            if len(selected) >= top_k:
                break
        if not selected:
            return self._empty_result(
                query, "知识库中没有找到足够可靠的相关证据"
            )

        parent_map = self.backend.get_parents(
            [parent_id for parent_id, _ in selected]
        )
        evidence = []
        for parent_id, child in selected:
            parent = parent_map.get(parent_id)
            if not parent:
                continue
            metadata = parent["metadata"]
            item = {
                "evidence_id": f"E{len(evidence) + 1}",
                "content": parent["document"],
                "matched_chunk": child["document"],
                "source": metadata.get("source", "unknown"),
                "scores": {
                    "similarity": child.get("similarity"),
                    "bm25": round(child.get("bm25_score", 0.0), 6),
                    "keyword_coverage": round(
                        child.get("keyword_coverage", 0.0), 6
                    ),
                    "rrf": round(child["rrf_score"], 8),
                    "reranker": child.get("reranker_score"),
                },
            }
            for field in ("page", "heading", "file_type"):
                if field in metadata:
                    item[field] = metadata[field]
            evidence.append(item)
        if not evidence:
            return self._empty_result(query, "命中的子块缺少对应父块")
        return {
            "success": True,
            "query": query,
            "results": evidence,
            "message": None,
        }

    @staticmethod
    def _empty_result(query, message):
        return {"success": True, "query": query, "results": [], "message": message}
