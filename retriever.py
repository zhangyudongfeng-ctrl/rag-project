"""
retriever.py：混合检索模块
职责：向量检索 + BM25检索 + RRF融合
"""

from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever


def create_retrievers(index, top_k=10):
    """
    创建两个检索器（启动时调用一次）
    - 向量检索器：用embedding相似度检索
    - BM25检索器：用关键词词频检索
    两者用同一批node，只是检索方式不同
    """
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=list(index.docstore.docs.values()),  # 从index中取出所有node
        similarity_top_k=top_k
    )
    return vector_retriever, bm25_retriever


def hybrid_retrieve(query, vector_retriever, bm25_retriever, k=60):
    """
    混合检索完整流程：
        1. 向量检索器和BM25检索器分别检索，各返回top_k个结果
        2. 用RRF（Reciprocal Rank Fusion）融合两路结果
        3. 返回融合排序后的结果列表

    参数：
        query: 用户问题
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器
        k: RRF平滑系数，默认60，防止排名靠前的结果分数差距过大

    返回：
        List[NodeWithScore]：融合后按分数降序排列，已去重
    """
    # 第一步：两路分别检索，返回的列表已按分数从高到低排序
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    # 第二步：RRF融合
    # 不看原始分数（两套评分体系不同），只看排名位置
    # 公式：score = 1/(k + rank + 1)，排名越靠前分数越高
    # 同一个chunk被两路都召回，分数会叠加，排名自然靠前
    scores = {}    # node_id -> 融合分数
    node_map = {}  # node_id -> node对象（用于去重，同一node只保留一份）

    for rank, node_with_score in enumerate(vector_results):
        node_id = node_with_score.node.node_id
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank + 1)
        node_map[node_id] = node_with_score.node

    for rank, node_with_score in enumerate(bm25_results):
        node_id = node_with_score.node.node_id
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank + 1)
        node_map[node_id] = node_with_score.node

    # 第三步：按融合分数降序排列，包装成NodeWithScore返回
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [NodeWithScore(node=node_map[nid], score=scores[nid]) for nid in sorted_ids]