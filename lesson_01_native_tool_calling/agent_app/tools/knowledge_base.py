"""用标准库 TF-IDF 实现的小型本地知识库搜索工具。"""

import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_DATA_PATH = Path(__file__).parents[2] / "data" / "knowledge_base.json"


def tokenize(text):
    """提取英文单词、数字，以及中文单字和双字片段。"""

    text = text.lower()
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in chinese_runs:
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def cosine_similarity(left, right):
    """计算两个稀疏向量的余弦相似度。"""

    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class KnowledgeBase:
    """加载JSON知识库，并通过TF-IDF相似度返回相关条目。"""

    def __init__(self, data_path=DEFAULT_DATA_PATH):
        payload = json.loads(Path(data_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("知识库必须是非空数组")
        self.documents = payload
        token_counts = [Counter(tokenize(item["content"])) for item in payload]
        document_frequency = Counter(
            token for counts in token_counts for token in counts
        )
        total = len(payload)
        self.idf = {
            token: math.log((total + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        self.document_vectors = [self._vector(counts) for counts in token_counts]

    def _vector(self, counts):
        """把词频转换成TF-IDF稀疏向量。"""

        total = sum(counts.values()) or 1
        return {
            token: count / total * self.idf.get(token, 1.0)
            for token, count in counts.items()
        }

    def search(self, query, top_k=3, min_score=0.05):
        """搜索最相关条目；没有结果达到阈值时返回空列表。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
            raise ValueError("top_k 必须是 1 到 5 之间的整数")
        query_vector = self._vector(Counter(tokenize(query)))
        ranked = []
        for document, vector in zip(self.documents, self.document_vectors):
            score = cosine_similarity(query_vector, vector)
            if score >= min_score:
                ranked.append({**document, "score": round(score, 4)})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]
