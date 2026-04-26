'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:12
 * @Description  : 
'''
import os
from dataclasses import dataclass

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek


@dataclass(frozen=True)
class RagConfig:
    deepseek_api_key: str
    llm_model: str = "deepseek-v4-flash"
    # 本地模型配置（值，不是对象）
    local_llm_base_url: str = "http://198.18.0.1:1234/v1"
    local_llm_model: str = "qwen/qwen3.5-9b"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_strategy: str = "capped"    # 目前仅支持2种策略: capped(一定 rerank，但只 rerank 前 N 个候选) / none(完全不走 reranker)
    data_dir: str = "data_cleaned"
    storage_dir: str = "storage"
    chunk_size: int = 512
    chunk_overlap: int = 50
    default_mode: str = "multi_query"
    similarity_top_k: int = 10
    reranker_top_n: int = 3                     # reranker内部使用
    reranker_candidate_pool_size: int = 14      # CappedReranker内部使用
    auxiliary_weight: float = 0.5               # Auxiliary_downweight_postprocessor内部使用


def load_config() -> RagConfig:
    from dotenv import load_dotenv
    load_dotenv()  # 读取 .env 文件到环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: DEEPSEEK_API_KEY")
    return RagConfig(deepseek_api_key=api_key)

def configure_settings(config: RagConfig) -> None:  
    os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api_key
    Settings.llm = DeepSeek(model=config.llm_model)
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embedding_model)
    # 这是默认兜底 splitter，不是主索引链路当前使用的切分器
    Settings.text_splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separator="\n\n",
    )
