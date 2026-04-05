"""
main.py V6：Multi-Query改写 + Hybrid Search + Rerank
用法：python mainV6.py
变更（相比V5）：
    - 新增 query_rewriter.py，用LLM把一个问题改写成多个角度
    - 检索从单query混合检索 → 多query混合检索
    - 被多个query多路召回的chunk分数叠加，排名更靠前
"""

import os
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# 导入自定义模块
from Chunker import chunk_by_paragraph, chunk_by_heading, add_file_metadata, chunks_to_nodes
from engine import build_components, HybridOnlyRetriever
from router import route_query
from intent_classifier import classify_intent

# ==========================================
# 配置
# ==========================================
os.environ["DEEPSEEK_API_KEY"] = "sk-69ec2fcb0a404e5e9b4a58a2412cb8ea"
Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

DATA_DIR = "data_cleaned"
STORAGE_DIR = "storage"

CHUNK_STRATEGY = chunk_by_paragraph
CHUNK_MAX_SIZE = 512


# ==========================================
# 构建索引
# ==========================================
def build_index_from_chunks() -> VectorStoreIndex:
    all_nodes = []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"切片: {filename} ({len(text):,} 字符)")

        chunks = CHUNK_STRATEGY(text, max_size=CHUNK_MAX_SIZE)
        chunks = add_file_metadata(chunks, filename)
        nodes = chunks_to_nodes(chunks)
        all_nodes.extend(nodes)

        print(f"  → {len(nodes)} 个切片")

    print(f"\n总计: {len(all_nodes)} 个切片，开始构建索引...\n")

    index = VectorStoreIndex(nodes=all_nodes)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index


def storage_is_valid(path):
    return os.path.exists(path) and os.path.exists(os.path.join(path, "docstore.json"))


def load_or_build_index() -> VectorStoreIndex:
    if storage_is_valid(STORAGE_DIR):
        print("加载已有索引...\n")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    else:
        return build_index_from_chunks()


# ==========================================
# 查询引擎配置
# ==========================================
index = load_or_build_index()

query_engine, retriever, reranker = build_components(index)
simple_retriever = HybridOnlyRetriever(index, top_k=10) 

# ==========================================
# 交互循环
# ==========================================
try:
    while True:
        question = input("\n问点什么（输入q退出）：")
        if question == "q":
            break
        if not question.strip():
            continue

        intent = classify_intent(question)
        result = route_query(question, intent, index, query_engine, simple_retriever)
        print(f"回答：{result['answer']}")
        for i, s in enumerate(result["sources"]):
            print(f"\n片段{i+1}（相关度：{s['score']}）")
            print(f"  来源: {s['source_file']}" + 
                (f" | 章节: {s['heading']}" if s['heading'] else "") + 
                (f" | 位置: {s['position']}" if s['position'] else ""))
            print(f"  内容: {s['text'][:200]}")

except KeyboardInterrupt:
    print("\n已退出")