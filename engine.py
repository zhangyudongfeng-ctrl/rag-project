"""
engine.py：共享查询引擎配置
main和run_eval都从这里获取query_engine，改一处全生效
"""

from collections import Counter
import logging
from typing import List, Optional

from attr import dataclass
from llama_index.core import QueryBundle, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from pydantic import PrivateAttr, model_validator

from config import RagConfig
from retriever import create_retrievers, multi_query_hybrid_retrieve
from query_rewriter import multi_query_rewrite


logger = logging.getLogger(__name__)


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


'''
 * @description: 重写BaseNodePostprocessor中的_postprocess_nodes方法,用于降低chunk中的content_type类型为Auxiliary时的权重->也就是分数 * 0.5
'''
class Auxiliary_downweight_postprocessor(BaseNodePostprocessor):
    weight: float = 0.5
    
    # 输入: 只需要nodes->List[NodeWithScore], 不需要query_bundle
    # 输出: 新的处理过的List[NodeWithScore]
    # 过程: 把List[NodeWithScore]中的score分数降权
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        for n in nodes:
            if n.node.metadata.get("content_type") == "auxiliary":
                n.score = (n.score or 0) * self.weight
        nodes.sort(key=lambda x: (x.score or 0), reverse=True)
        return nodes


'''
 * @description: 仅返回前top_n个节点，不进行任何rerank, 用于reranker_strategy=none的情况
'''
class NoOpReranker(BaseNodePostprocessor):
    reranker_top_n: int = 3

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        return nodes[: self.reranker_top_n]

# 类内部的字段值兜底,但这个字段值在实例化时被 RagConfig 覆盖了
'''
 * @description: 先取前candidate_pool_size个节点进行rerank，再返回前top_n个节点
'''
class CappedReranker(BaseNodePostprocessor):
    reranker_top_n: int = 3
    candidate_pool_size: int = 14
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    _reranker: FlagEmbeddingReranker = PrivateAttr(default=None)

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = FlagEmbeddingReranker(
                model=self.reranker_model,
                top_n=self.reranker_top_n,
            )
        return self._reranker

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        candidates = nodes[: self.candidate_pool_size]
        reranked = self._get_reranker().postprocess_nodes(
            candidates, query_bundle=query_bundle
        )
        logger.debug(
            "CappedReranker applied rerank: original=%s candidate_pool=%s returned=%s",
            len(nodes),
            len(candidates),
            len(reranked),
        )
        return reranked

''' SelectiveReranker调用逻辑: ---> 废弃
接收 retriever + RRF 之后的候选节点列表
            ↓
先截断成前 candidate_pool_size 个候选
            ↓
调用 _should_skip_rerank() 判断要不要跳过 rerank
            ↓
如果跳过：直接返回这批候选里的前 top_n
            ↓
如果不跳过：用底层 FlagEmbeddingReranker 对这批候选重排
            ↓
返回 rerank 后的前 top_n
'''

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
        results = multi_query_hybrid_retrieve(
        queries, self.vector_retriever, self.bm25_retriever)
        return results

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


'''
 * @description: 用于映射retriever模式
 * @param {*} index
 * @param {RagConfig} config
 * @return {*} retriever
'''
def build_retriever(index, config: RagConfig):
    """
    mode: 
        "vector_only" — 纯向量检索（V4）
        "hybrid" — 向量+BM25（V5）
        "hybrid_jieba" — 向量+BM25+jieba（V5变体）
        "multi_query" — Multi-Query+混合检索（V6） -- 默认
    """
    # 根据map里的config.default_mode使用不同的retriever,避免if elif的堆叠
    retriever_builders = {
        "vector_only": lambda: VectorIndexRetriever(
            index=index,
            similarity_top_k=config.similarity_top_k,
        ),
        "hybrid": lambda: HybridOnlyRetriever(
            index=index,
            top_k=config.similarity_top_k,
        ),
        "hybrid_jieba": lambda: HybridOnlyRetriever(
            index=index,
            top_k=config.similarity_top_k,
            use_jieba=True,
        ),
        "multi_query": lambda: MultiQueryHybridRetriever(
            index=index,
            llm=Settings.llm,
            top_k=config.similarity_top_k,
        ),
        "multi_query_jieba": lambda: MultiQueryHybridRetriever(
            index=index,
            llm=Settings.llm,
            top_k=config.similarity_top_k,
            use_jieba=True,
        ),
    }

    builder = retriever_builders.get(config.default_mode)
    # TODO 测试代码
    print(f"Building retriever with mode: {config.default_mode}")
    if builder is None:
        raise ValueError(f"Unsupported retrieval mode: {config.default_mode}")

    return builder()
'''
 * @description: 用于映射reranker模式
 * @param {RagConfig} config
 * @return {*} reranker
'''
def build_reranker(config: RagConfig):
    reranker_builders = {
        "none": lambda: NoOpReranker(reranker_top_n=config.reranker_top_n),
        "capped": lambda: CappedReranker(
            reranker_model=config.reranker_model,
            reranker_top_n=config.reranker_top_n,
            candidate_pool_size=config.reranker_candidate_pool_size,),
    }          
    
    builder = reranker_builders.get(config.reranker_strategy)
    # TODO 测试代码
    print(f"Building reranker with strategy: {config.reranker_strategy}")
    if builder is None:
        raise ValueError(f"Unsupported reranker strategy: {config.reranker_strategy}")
    return builder()

# ==========================================
# 构建查询引擎（唯一入口）
# ==========================================
'''
 * @description: 向外界暴漏查询引擎/检索器/重排器
 * @param {*} index, RagConfig
 * @return {*} query_engine, retriever, reranker
'''
def build_components(
    index,
    config: RagConfig,
):
    
    retriever = build_retriever(index, config)
    reranker = build_reranker(config)
    # reranker 先用 cross-encoder 给出真实相关性分数,
    # 然后 post_processor 对 auxiliary 类型的 chunk 在真实分数上降权,
    # 最终影响 top_n 截断的取舍
    post_processor = Auxiliary_downweight_postprocessor(weight=config.auxiliary_weight) # 在实例化时传入字段值，而字段值由 Pydantic 初始化机制接收和绑定
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker, post_processor],
        #node_postprocessors=[post_processor],
        response_mode='compact',
        text_qa_template=qa_prompt
    )
    return query_engine, retriever, reranker
