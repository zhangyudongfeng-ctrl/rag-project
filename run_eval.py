"""
评估运行脚本：一键跑评估
用法：python run_eval.py [run_name]
示例：
    python run_eval.py v3_baseline      # 给当前版本跑基准线
    python run_eval.py v4_paragraph     # 改了切片策略后再跑
    
然后对比两次结果看效果变化。
"""

import sys
import os
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.prompts import PromptTemplate

from evaluator import run_evaluation, print_report, save_report

# ==========================================
# 配置（和 main.py 保持一致）
# ==========================================
os.environ["DEEPSEEK_API_KEY"] = "sk-47aa291d867345d0991602b8b9b4441d"
Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

STORAGE_DIR = "storage"


def main():
    # 运行名称：用于区分不同版本的评估结果
    run_name = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

    # 加载索引
    if not os.path.exists(STORAGE_DIR):
        print(f"错误：找不到 {STORAGE_DIR}/ 目录，请先运行 main.py 构建索引")
        return

    print("加载索引...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    from llama_index.core import load_index_from_storage
    index = load_index_from_storage(storage_context)

    # 配置查询引擎（和 main.py 保持一致）
    reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=3)

    qa_prompt = PromptTemplate(
        "以下是相关的参考资料：\n-----\n{context_str}\n-----\n"
        "请基于以上参考资料回答问题。可以基于资料内容进行合理推理，但不要编造资料中没有的信息。"
        "如果资料中完全没有相关信息，请说'文档中没有相关内容'。\n"
        "问题：{query_str}\n回答："
    )

    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker],
        response_mode="compact",
        text_qa_template=qa_prompt
    )

    # 运行评估
    print(f"\n开始评估 [{run_name}]...\n")
    results = run_evaluation(
        query_engine=query_engine,
        llm=Settings.llm,
        use_llm_judge=True,  # 改成 False 可省 API 费用，只跑关键词指标
    )

    # 输出报告
    print_report(results, run_name=run_name)
    save_report(results, run_name=run_name)


if __name__ == "__main__":
    main()