import os
from dataclasses import dataclass

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek


@dataclass(frozen=True)
class RagConfig:
    deepseek_api_key: str
    llm_model: str = "deepseek-chat"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    data_dir: str = "data_cleaned"
    storage_dir: str = "storage"
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 10
    reranker_top_n: int = 3
    default_mode: str = "multi_query"


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
    Settings.text_splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separator="\n\n",
    )