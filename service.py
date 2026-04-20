'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:12
 * @Description  : 
'''
import os
from dataclasses import dataclass

from llama_index.llms.openai_like import OpenAILike
from config import RagConfig
from engine import CappedReranker, HybridOnlyRetriever, NoOpReranker, build_components
from rag_compotents import RagComponents
from router import route_query
from index_factory import load_or_build_index, rebuild_index_from_LlamaIndex
from intent_classifier import classify_intent

class RagService:
    def __init__(self, config: RagConfig):
        self.config = config
        self.local_llm = OpenAILike(
            api_base=self.config.local_llm_base_url,
            api_key="not-needed",
            model=self.config.local_llm_model,
            is_chat_model=True,  # 让传输格式匹配模型的训练格式, False发纯字符串, True发messages数组
        )
        self.components = self._build_components()

    '''
     * @description: 从配置中获取各个参数, 构建索引、查询引擎、检索器和重排器等组件
     * @param {*} self
     * @return {RagComponents} 
    '''    
    def _build_components(self) -> RagComponents:
        # TODO 在这里切换切片方案
        # 使用框架内部的文本切分方案
        # index = rebuild_index_from_LlamaIndex(self.config)

        # 自定义文本切分方案
        index = load_or_build_index(self.config)
        query_engine, _, reranker = build_components(
            index,
            config=self.config,
        )
        simple_retriever = HybridOnlyRetriever(
            index=index,
            top_k=self.config.similarity_top_k,
        )
        return RagComponents(
            index=index,
            query_engine=query_engine,
            simple_retriever=simple_retriever,
            reranker=reranker,
            local_llm = self.local_llm
        )

    def reload(self) -> None:
        self.components = self._build_components()

    
    # 有了路由机制后,所有问题都会经过这个函数，先进行意图分类，再路由到不同的处理逻辑
    def query(self, question: str) -> dict:
        intent = classify_intent(question, self.local_llm)
        return route_query(question, intent, self.components)

    
    # 目前主要就是给 api.py 的 /upload 接口用的，而这个接口又是给前端网页上传文件走的
    def upload_text(self, filename: str, content: str) -> dict:
        from Chunker import add_file_metadata, chunk_by_paragraph, chunks_to_nodes

        save_path = os.path.join(self.config.data_dir, filename)
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(content)

        chunks = chunk_by_paragraph(content, max_size=self.config.chunk_size)
        chunks = add_file_metadata(chunks, filename)
        new_nodes = chunks_to_nodes(chunks)

        for node in new_nodes:
            self.components.index.insert_nodes([node])

        self.components.index.storage_context.persist(persist_dir=self.config.storage_dir)
        self.reload()

        return {"filename": filename, "chunks_added": len(new_nodes)}
