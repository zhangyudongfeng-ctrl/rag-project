# 目的：写一个retriever，用于替换query_engine里的retriever参数
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from retriever import create_retrievers, multi_query_hybrid_retrieve

# 自定义类，继承自BaseRetriever
class MultiQueryHybridRetriever(BaseRetriever):
    # 需要调用create()
    def __init__(self, index, top_k=10, llm):
        self.vector_retriever, self.bm25_retriever = create_retrievers(index, top_k)
        self.llm = llm
        super().__init__()
    
    # 需要返回RRF融合查询后的结果
    def _retrieve(self, query_bundle):
        query = query_bundle.query_str
        res = multi_query_hybrid_retrieve(query, vector_res, bm25_res)
        return res