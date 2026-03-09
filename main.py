import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 配置
os.environ["DEEPSEEK_API_KEY"] = "sk-47aa291d867345d0991602b8b9b4441d"

# 选择大语言模型和embedding模型
Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 如果已有索引就直接加载，没有就新建
if os.path.exists("storage"):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    print("已加载现有索引")
else:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="storage")
    print("索引已创建并保存")

query_engine = index.as_query_engine()

while True:
    question = input("\n问点什么（输入q退出）：")
    if question == "q":
        break
    response = query_engine.query(question)
    print(f"\n{'='*50}")
    print(f"回答：{response}")
    print(f"\n--- 检索到的原文片段 ---")
    for i, node in enumerate(response.source_nodes):
        print(f"\n片段{i+1}（相似度：{node.score:.3f}）：")
        print(node.text[:200])
    print(f"{'='*50}")