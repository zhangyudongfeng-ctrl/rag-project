import os
from dataclasses import dataclass

from config import RagConfig
from engine import HybridOnlyRetriever, build_components
from index_factory import load_or_build_index
from router import route_query
from intent_classifier import classify_intent


@dataclass
class RagComponents:
    index: object
    query_engine: object
    simple_retriever: object
    reranker: object


class RagService:
    def __init__(self, config: RagConfig):
        self.config = config
        self.components = self._build_components()

    '''
     * @description: 从配置中获取各个参数, 构建索引、查询引擎、检索器和重排器等组件
     * @param {*} self
     * @return {RagComponents} 
    '''    
    def _build_components(self) -> RagComponents:
        index = load_or_build_index(self.config)
        query_engine, _, reranker = build_components(
            index,
            mode=self.config.default_mode,
            similarity_top_k=self.config.similarity_top_k,
            reranker_top_n=self.config.reranker_top_n,
            reranker_model=self.config.reranker_model,
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
        )

    def reload(self) -> None:
        self.components = self._build_components()

    
    # 有了路由机制后,所有问题都会经过这个函数，先进行意图分类，再路由到不同的处理逻辑
    def query(self, question: str) -> dict:
        intent = classify_intent(question)
        return route_query(
            question=question,
            intent=intent,
            index=self.components.index,
            query_engine=self.components.query_engine,
            retriever=self.components.simple_retriever,
        )

    
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
