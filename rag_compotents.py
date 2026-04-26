'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-16 15:34:49
 * @Description  : 
'''
from dataclasses import dataclass
from typing import Any
from engine import CappedReranker, HybridOnlyRetriever, NoOpReranker


@dataclass
class RagComponents:
    index: Any
    query_engine: Any
    streaming_query_engine: Any
    simple_retriever: HybridOnlyRetriever
    reranker: NoOpReranker | CappedReranker
    local_llm: Any