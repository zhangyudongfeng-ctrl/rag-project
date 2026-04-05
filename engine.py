"""
engine.py：共享查询引擎配置
main和run_eval都从这里获取query_engine，改一处全生效
"""

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from retriever import create_retrievers, multi_query_hybrid_retrieve
from query_rewriter import multi_query_rewrite


# ==========================================
# Prompt配置
# ==========================================
qa_prompt = PromptTemplate(
    "你是一个严谨的文献问答助手，基于提供的参考资料回答问题。\n\n"
    
    "【参考资料】\n{context_str}\n\n"
    
    "【回答规则】\n"
    "1. 仔细阅读所有参考资料，提取与问题相关的关键信息\n"
    "2. 如果问题涉及多个方面，分点回答，确保覆盖完整\n"
    "3. 如果问题有歧义，列出所有可能的理解并分别回答\n"
    "4. 可以基于资料进行合理推理，但明确标注哪些是推理\n"
    "5. 如果资料中完全没有相关信息，直接说'文档中没有相关内容'\n"
    "6. 回答中尽量使用资料中的原始术语和关键词\n\n"
    
    "问题：{query_str}\n回答："
)


# ==========================================
# 自定义混合检索器
# ==========================================
class MultiQueryHybridRetriever(BaseRetriever):
    """Multi-Query + 混合检索器"""
    def __init__(self, index, llm, top_k=10, use_jieba=False):
        self.vector_retriever, self.bm25_retriever = create_retrievers(index, top_k, use_jieba)
        self.llm = llm
        super().__init__()

    def _retrieve(self, query_bundle):
        query = query_bundle.query_str
        queries = multi_query_rewrite(query, self.llm)
        return multi_query_hybrid_retrieve(
            queries, self.vector_retriever, self.bm25_retriever
        )

# ==========================================
# 纯Hybrid的检索器
# ==========================================
class HybridOnlyRetriever(BaseRetriever):
    """V5: 向量+BM25，不做Query改写"""
    def __init__(self, index, top_k=10, use_jieba=False):
        self.vector_retriever, self.bm25_retriever = create_retrievers(index, top_k, use_jieba)
        super().__init__()

    def _retrieve(self, query_bundle):
        from retriever import hybrid_retrieve
        return hybrid_retrieve(query_bundle.query_str, self.vector_retriever, self.bm25_retriever)

# ==========================================
# 构建查询引擎（唯一入口）
# ==========================================
'''
 * @description: 向外界暴漏查询引擎/检索器/重排器
 * @param {*} index
 * @return {*} query_engine, retriever, reranker
'''
def build_components(
    index,
    mode="multi_query",
    similarity_top_k=10,
    reranker_top_n=3,
    reranker_model="BAAI/bge-reranker-v2-m3",
):
    """
    mode: 
        "vector_only" — 纯向量检索（V4）
        "hybrid" — 向量+BM25（V5）
        "hybrid_jieba" — 向量+BM25+jieba（V5变体）
        "multi_query" — Multi-Query+混合检索（V6） -- 默认
    """
    if mode == "vector_only":
        from llama_index.core.retrievers import VectorIndexRetriever
        retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    elif mode == "hybrid":
        retriever = HybridOnlyRetriever(index, top_k=similarity_top_k)
    elif mode == "hybrid_jieba":
        retriever = HybridOnlyRetriever(index, top_k=similarity_top_k, use_jieba=True)
    elif mode == "multi_query":
        retriever = MultiQueryHybridRetriever(index, llm=Settings.llm, top_k=similarity_top_k)
    elif mode == "multi_query_jieba":
        retriever = MultiQueryHybridRetriever(index, llm=Settings.llm, top_k=similarity_top_k, use_jieba=True)
    else:
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    reranker = FlagEmbeddingReranker(model=reranker_model, top_n=reranker_top_n)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_mode="compact",
        text_qa_template=qa_prompt
    )
    return query_engine, retriever, reranker
