import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# 配置
os.environ["DEEPSEEK_API_KEY"] = "sk-47aa291d867345d0991602b8b9b4441d"

# 选择大语言模型和embedding模型
Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1. 这里的配置，决定了你的 AI 的“视野宽度”和“专注度”
# chunk_size=512: 适合中文 Embedding 模型 (bge-small 建议 300-512)
# chunk_overlap=50: 预留 10% 的重叠，防止语义割裂
Settings.text_splitter = SentenceSplitter(
    chunk_size=512, 
    chunk_overlap=50,
    separator="\n\n" # 优先按段落切分，这非常重要！
)

# 函数作用：根据文件路径自动提取书名 -> 目的：准备在读取文件时把书名标签贴到每一个Node上
def get_book_metadata(file_path):
    # 比如：file_path 是 "data_cleaned/道德经.txt"
    file_name = os.path.basename(file_path)
    book_name = os.path.splitext(file_name)[0]
    return {"book_name": book_name}

# 如果已有索引就直接加载，没有就新建
if os.path.exists("storage"):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    print("已加载现有索引")
else:
    # 在读取的时候，把这个“元数据提取器”挂上去
    documents = SimpleDirectoryReader(
        "./data_cleaned", 
        file_metadata=get_book_metadata
    ).load_data()   # 这一步之后，所有的 documents 对象里，都已经自带了 book_name 这个元数据
    index = VectorStoreIndex.from_documents(documents)  # from_documents内部先把Document按默认策略（SentenceSplitter, chunk_size=1024）切成Node，再算embedding建索引。V4就是把这个切片步骤拿出来自己控制。
    index.storage_context.persist(persist_dir="storage")
    print("索引已创建并保存")

# 在system prompt里加约束，让LLM只基于检索内容回答
qa_prompt = PromptTemplate(
    "以下是相关的参考资料：\n"
    "-----\n"
    "{context_str}\n"
    "-----\n"
    "请基于以上参考资料回答问题。可以基于资料内容进行合理推理，但不要编造资料中没有的信息。"
    "如果资料中完全没有相关信息，请说'文档中没有相关内容'。\n"
    "问题：{query_str}\n"
    "回答："
)

# Rerank：向量检索先捞回一批候选片段，然后用一个cross-encoder模型对每个片段和问题重新打分，把真正相关的排到前面
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-v2-m3", 
    top_n=3     
)

query_engine = index.as_query_engine(
    similarity_top_k=10,    # 向量检索先粗捞10个片段，宁可多捞，不怕有不相关的
    node_postprocessors=[reranker],     # Rerank对这10个片段和问题逐一精算相关度，只留最相关的3个喂给LLM
    response_mode="compact",
    text_qa_template=qa_prompt
)

try:
    while True:
        question = input("\n问点什么（输入q退出）：")
        if question == "q":
            break
        if not question.strip():
            continue
        response = query_engine.query(question)
        print(f"\n{'='*50}")
        print(f"回答：{response}")
        print(f"\n--- 检索到的原文片段 ---")
        for i, node in enumerate(response.source_nodes):
            print(f"\n片段{i+1}（相似度：{node.score:.3f}）：")
            print(node.text[:200])
            meta = node.metadata
            # 尝试提取你关心的核心元数据，如果不存在就返回 'N/A'
            file_name = meta.get("file_name", meta.get("source", "未知文件"))
            print(f"\n 来源: {file_name}")
            print(f" 完整元数据: {meta}") # 调试时可以保留，等结构稳定了再精简
        print(f"{'='*50}")
except KeyboardInterrupt:
    print("\n已退出")