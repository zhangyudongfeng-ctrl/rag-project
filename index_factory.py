'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:12
 * @Description  : 把“检查 storage 是否可用”和“从 data_cleaned 构建索引”的逻辑抽出来
'''

import os

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from Chunker import add_file_metadata, chunk_by_paragraph, chunks_to_nodes
from config import RagConfig


def storage_is_valid(path: str) -> bool:
    return os.path.exists(path) and os.path.exists(os.path.join(path, "docstore.json"))


def build_index_from_data(config: RagConfig) -> VectorStoreIndex:
    all_nodes = []
    for filename in os.listdir(config.data_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(config.data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_by_paragraph(text, max_size=config.chunk_size)
        chunks = add_file_metadata(chunks, filename)
        all_nodes.extend(chunks_to_nodes(chunks))

    index = VectorStoreIndex(nodes=all_nodes)
    index.storage_context.persist(persist_dir=config.storage_dir)
    return index


def load_or_build_index(config: RagConfig) -> VectorStoreIndex:
    if storage_is_valid(config.storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=config.storage_dir)
        return load_index_from_storage(storage_context)
    return build_index_from_data(config)
