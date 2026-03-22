"""
main.py V5：Hybrid Search（向量 + BM25 + RRF融合）+ Rerank + 元数据溯源
用法：python mainV5.py
变更（相比V4）：
    - 纯向量检索 → 混合检索（向量 + BM25）
    - 新增 retriever.py 负责检索逻辑
    - 用 RetrieverQueryEngine 替代 index.as_query_engine()
"""

import os
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# 导入自定义模块
from Chunker import chunk_by_paragraph, chunk_by_heading, add_file_metadata, chunks_to_nodes
from retriever import create_retrievers, hybrid_retrieve

# ==========================================
# 配置
# ==========================================
os.environ["DEEPSEEK_API_KEY"] = "sk-47aa291d867345d0991602b8b9b4441d"
Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

DATA_DIR = "data_cleaned"
STORAGE_DIR = "storage"

# 切片策略：改这里就能切换
# 可选：chunk_by_paragraph（推荐）、chunk_by_heading（有章节结构的书）
CHUNK_STRATEGY = chunk_by_paragraph
CHUNK_MAX_SIZE = 512


# ==========================================
# 构建索引：用自定义切片替代默认切片
# ==========================================
def build_index_from_chunks() -> VectorStoreIndex:
    """
    自定义切片流程：
        1. 读取 data_cleaned/ 下的 txt 文件
        2. 用 chunker 切片（不再依赖 LlamaIndex 的 SimpleDirectoryReader + 默认 splitter）
        3. 切片转为 TextNode
        4. 直接从 nodes 构建索引
    """
    all_nodes = []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"切片: {filename} ({len(text):,} 字符)")

        # 切片
        chunks = CHUNK_STRATEGY(text, max_size=CHUNK_MAX_SIZE)
        chunks = add_file_metadata(chunks, filename)

        # 转为 LlamaIndex 节点
        nodes = chunks_to_nodes(chunks)
        # 把一个列表的所有元素追加到另一个列表末尾,append会把整个对象作为一个元素塞进去 例：a = [1, 2, [3, 4]]
        all_nodes.extend(nodes)

        print(f"  → {len(nodes)} 个切片")

    print(f"\n总计: {len(all_nodes)} 个切片，开始构建索引...\n")

    # 从 nodes 直接构建索引（绕过 SimpleDirectoryReader）
    # 第一行：把所有切好的node传进去，计算embedding，建立向量索引。
    # 第二行：把索引保存到磁盘（storage文件夹），下次启动直接加载，不用重新算embedding。这就是解决"每次重启重新embedding"的问题。
    index = VectorStoreIndex(nodes=all_nodes)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

def storage_is_valid(path):
    return os.path.exists(path) and os.path.exists(os.path.join(path, "docstore.json"))

def load_or_build_index() -> VectorStoreIndex:
    """有 storage 就加载，没有就新建"""
    if storage_is_valid(STORAGE_DIR):
        print("加载已有索引...\n")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    else:
        return build_index_from_chunks()

# ==========================================
# 自定义混合检索器（包装成LlamaIndex能用的格式）
# ==========================================
# 定义一个叫HybridRetriever的类，继承自BaseRetriever。继承意味着HybridRetriever自动拥有BaseRetriever的所有方法和属性
# 只需要重写_retrieve方法，把自己的混合检索逻辑填进去。括号里的就是父类。
class HybridRetriever(BaseRetriever):
    """
    混合检索器：向量 + BM25 + RRF融合
    包装成 BaseRetriever 子类，这样可以直接传给 RetrieverQueryEngine
    """
    def __init__(self, index, top_k=10):
        self.vector_retriever, self.bm25_retriever = create_retrievers(index, top_k)
        super().__init__()
 
    def _retrieve(self, query_bundle):
        """LlamaIndex 调用检索时会走这个方法"""
        query = query_bundle.query_str
        return hybrid_retrieve(query, self.vector_retriever, self.bm25_retriever)

# ==========================================
# 查询引擎配置
# ==========================================
index = load_or_build_index()

# V4: query_engine = index.as_query_engine(similarity_top_k=10, ...)
# V5: 自定义混合检索器 → RetrieverQueryEngine
hybrid_retriever = HybridRetriever(index, top_k=10)
reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=3)
 
qa_prompt = PromptTemplate(
    "以下是相关的参考资料：\n-----\n{context_str}\n-----\n"
    "请基于以上参考资料回答问题。可以基于资料内容进行合理推理，但不要编造资料中没有的信息。"
    "如果资料中完全没有相关信息，请说'文档中没有相关内容'。\n"
    "问题：{query_str}\n回答："
)
 
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    response_mode="compact",
    text_qa_template=qa_prompt
)
 


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

        response = query_engine.query(question)

        print(f"\n{'='*50}")
        print(f"回答：{response}")
        print(f"\n--- 检索到的原文片段 ---")
        for i, node in enumerate(response.source_nodes):
            score = node.score
            meta = node.metadata
            source = meta.get("source_file", "未知")
            heading = meta.get("heading", "")
            position = meta.get("position", "")

            print(f"\n片段{i+1}（相关度：{score:.3f}）")
            print(f"  来源: {source}" + (f" | 章节: {heading}" if heading else "") + (f" | 位置: {position}" if position else ""))
            print(f"  内容: {node.text[:200]}")
        print(f"{'='*50}")

except KeyboardInterrupt:
    print("\n已退出")