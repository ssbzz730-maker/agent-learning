"""建立父子分块 Chroma 索引；子块检索，父块提供完整证据。"""

import argparse
import hashlib
from pathlib import Path

from lesson_03_rag_tool.rag.document_loaders import SUPPORTED_SUFFIXES, load_file


BASE_DIR = Path(__file__).parents[1]
DEFAULT_DOCUMENT_DIR = BASE_DIR / "data" / "documents"
DEFAULT_DATABASE_DIR = BASE_DIR / "rag_database"
DEFAULT_MODEL_CACHE_DIR = BASE_DIR / ".model-cache"
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
PARENT_COLLECTION_NAME = "parent_documents"
CHILD_COLLECTION_NAME = "child_documents"


def split_text(text, chunk_size, overlap):
    """按字符滑窗分块；重叠用于保留边界语义。"""

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("必须满足 chunk_size > 0 且 0 <= overlap < chunk_size")
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def stable_id(*parts):
    """根据来源、位置和内容生成稳定 ID。"""

    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def collect_records(
    document_dir,
    parent_size=600,
    parent_overlap=100,
    child_size=160,
    child_overlap=40,
):
    """读取文档并构建父块和子块记录。"""

    document_dir = Path(document_dir)
    parents = []
    children = []
    paths = sorted(
        path
        for path in document_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in paths:
        for section_index, section in enumerate(
            load_file(path, document_dir), start=1
        ):
            parent_chunks = split_text(
                section["text"], parent_size, parent_overlap
            )
            for parent_index, parent_text in enumerate(parent_chunks, start=1):
                source = section["metadata"]["source"]
                parent_id = stable_id(
                    "parent", source, section_index, parent_index, parent_text
                )
                metadata = {
                    **section["metadata"],
                    "section_id": section_index,
                    "parent_index": parent_index,
                }
                parents.append(
                    {"id": parent_id, "document": parent_text, "metadata": metadata}
                )
                for child_index, child_text in enumerate(
                    split_text(parent_text, child_size, child_overlap), start=1
                ):
                    children.append(
                        {
                            "id": stable_id(
                                "child", parent_id, child_index, child_text
                            ),
                            "document": child_text,
                            "metadata": {
                                **metadata,
                                "parent_id": parent_id,
                                "child_index": child_index,
                            },
                        }
                    )
    return parents, children


def replace_collection(client, name):
    """重建指定集合，避免旧记录残留。"""

    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def build_index(
    document_dir=DEFAULT_DOCUMENT_DIR,
    database_dir=DEFAULT_DATABASE_DIR,
    model_cache_dir=DEFAULT_MODEL_CACHE_DIR,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
):
    """编码父子块并写入 Chroma，返回数量摘要。"""

    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("请先安装 requirements-rag.txt") from error

    parents, children = collect_records(document_dir)
    if not parents or not children:
        raise ValueError(f"{document_dir} 中没有可建立索引的内容")

    model = SentenceTransformer(
        embedding_model,
        cache_folder=str(model_cache_dir),
    )
    parent_embeddings = model.encode(
        [item["document"] for item in parents],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    child_embeddings = model.encode(
        [item["document"] for item in children],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(
        path=str(database_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    parent_collection = replace_collection(client, PARENT_COLLECTION_NAME)
    child_collection = replace_collection(client, CHILD_COLLECTION_NAME)
    parent_collection.add(
        ids=[item["id"] for item in parents],
        documents=[item["document"] for item in parents],
        embeddings=parent_embeddings.tolist(),
        metadatas=[item["metadata"] for item in parents],
    )
    child_collection.add(
        ids=[item["id"] for item in children],
        documents=[item["document"] for item in children],
        embeddings=child_embeddings.tolist(),
        metadatas=[item["metadata"] for item in children],
    )
    return {"parents": len(parents), "children": len(children)}


def main():
    parser = argparse.ArgumentParser(description="建立第3课父子分块RAG索引")
    parser.add_argument("--document-dir", type=Path, default=DEFAULT_DOCUMENT_DIR)
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    result = build_index(
        args.document_dir,
        args.database_dir,
        args.model_cache_dir,
        args.embedding_model,
    )
    print(f"索引完成：父块 {result['parents']}，子块 {result['children']}")


if __name__ == "__main__":
    main()
