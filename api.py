"""
api.py：FastAPI服务端
POST /query 接收问题，返回答案+出处+耗时
"""

import os
import time
from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core import Settings, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from Chunker import chunk_by_paragraph, add_file_metadata, chunks_to_nodes
from engine import build_query_engine


# ==========================================
# 配置
# ==========================================
os.environ["DEEPSEEK_API_KEY"] = "sk-47aa291d867345d0991602b8b9b4441d"
Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

DATA_DIR = "data_cleaned"
STORAGE_DIR = "storage"


# ==========================================
# 索引加载（启动时执行一次）
# ==========================================
def storage_is_valid(path):
    return os.path.exists(path) and os.path.exists(os.path.join(path, "docstore.json"))


def load_or_build_index() -> VectorStoreIndex:
    if storage_is_valid(STORAGE_DIR):
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("构建新索引...")
        all_nodes = []
        for filename in os.listdir(DATA_DIR):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunk_by_paragraph(text, max_size=512)
            chunks = add_file_metadata(chunks, filename)
            nodes = chunks_to_nodes(chunks)
            all_nodes.extend(nodes)
        index = VectorStoreIndex(nodes=all_nodes)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        return index


index = load_or_build_index()
query_engine = build_query_engine(index)


# ==========================================
# FastAPI应用
# ==========================================
app = FastAPI(title="RAG问答系统")


# ---------- 数据模型 ----------

class QueryRequest(BaseModel):
    question: str


class SourceNode(BaseModel):
    score: float
    source_file: str
    heading: str
    position: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode]
    time_seconds: float


# ---------- 接口 ----------

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()

    response = query_engine.query(request.question)

    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append(SourceNode(
            score=round(node.score, 4),
            source_file=meta.get("source_file", "未知"),
            heading=meta.get("heading", ""),
            position=meta.get("position", ""),
            text=node.text[:300]
        ))

    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        answer=str(response),
        sources=sources,
        time_seconds=elapsed
    )


@app.get("/health")
def health():
    return {"status": "ok"}