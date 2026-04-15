'''
 * @Author       : MatthewZhang
 * @Date         : 2026-03-22 14:02:26
 * @Description  : 
'''
"""
retriever.py：混合检索模块
职责：向量检索 + BM25检索 + RRF融合 + Multi-Query支持
B向量检索靠语义相似度，但有时候用户问的就是一个精确关键词，语义相近的chunk反而不是用户要的。比如用户问"八正道"，向量检索可能把讲"修行方法"的chunk排很高（语义相近），但漏掉了明确包含"八正道"三个字的那个chunk。
BM25纯看词频——文档里"八正道"出现越多、越集中，分数越高。不理解语义，但精确匹配不会漏。
"""

import jieba
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever

def chinese_tokenizer(text):
    return list(jieba.cut_for_search(text))

def create_retrievers(index, top_k=10, use_jieba=False):
    """
    创建两个检索器（启动时调用一次）
    - 向量检索器：用embedding相似度检索
    - BM25检索器：用关键词词频检索
    - 是否使用jieba分词，默认关闭
    两者用同一批node，只是检索方式不同
    """
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    kwargs = {
        "nodes": list(index.docstore.docs.values()),
        "similarity_top_k": top_k
    }
    if use_jieba:
        kwargs["tokenizer"] = chinese_tokenizer

    bm25_retriever = BM25Retriever.from_defaults(**kwargs)
    return vector_retriever, bm25_retriever


def rrf_fusion(results_list, k=60):
    """
    RRF融合（通用版）：把多路检索结果按排名合并

    输入：results_list — 多个 List[NodeWithScore]，每个是一路检索的结果
    输出：融合排序后的 List[NodeWithScore]，已去重

    公式：score = sum( 1/(k + rank + 1) )，同一个chunk被多路召回分数叠加
    """
    scores = {}    # node_id -> 融合分数
    node_map = {}  # node_id -> node对象

    # 遍历并给每一路打分 
    total_results = sum(len(results) for results in results_list)
    print(f"去重前总节点数: {total_results}")  
    for results in results_list:
        for rank, node_with_score in enumerate(results):
            node_id = node_with_score.node.node_id
            scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank + 1)
            node_map[node_id] = node_with_score.node

    print(f"去重后的候选节点数: {len(scores)}") 
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [NodeWithScore(node=node_map[nid], score=scores[nid]) for nid in sorted_ids]


def hybrid_retrieve(query, vector_retriever, bm25_retriever, k=60):
    """
    单query混合检索：向量 + BM25 + RRF融合

    输入：一个用户问题 + 两个检索器
    输出：融合排序后的 List[NodeWithScore]
    """
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)
    return rrf_fusion([vector_results, bm25_results], k)


def multi_query_hybrid_retrieve(queries, vector_retriever, bm25_retriever, k=60):
    """
    Multi-Query混合检索：多个query分别走混合检索，最后统一RRF融合

    输入：
        queries — 改写后的多个问题列表（如 [原始问题, 改写1, 改写2, 改写3]）
        vector_retriever — 向量检索器
        bm25_retriever — BM25检索器
        k — RRF平滑系数

    输出：所有query的检索结果统一融合后的 List[NodeWithScore]

    流程：
        1. 每个query分别用向量检索器和BM25检索器检索（共 4个query × 2路 = 8次检索）
        2. 所有结果统一RRF融合（被多个query多路召回的chunk分数叠加，排名靠前）
    """
    all_results = []
    for query in queries:
        vector_results = vector_retriever.retrieve(query)
        bm25_results = bm25_retriever.retrieve(query)
        all_results.append(vector_results)
        all_results.append(bm25_results)

    return rrf_fusion(all_results, k)