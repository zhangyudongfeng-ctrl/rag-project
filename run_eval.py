'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:38
 * @Description  : 
'''
import os
import sys
import logging

from llama_index.core import Settings, StorageContext, load_index_from_storage

from config import configure_settings, load_config
from engine import HybridOnlyRetriever, build_components
from evaluator import print_report, run_evaluation, save_report

logger = logging.getLogger(__name__)

def main():
    config = load_config()
    configure_settings(config)
    run_name = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

    if not os.path.exists(config.storage_dir):
        logger.info(f"Missing storage directory: {config.storage_dir}")
        return

    logger.info("Loading index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=config.storage_dir)
    index = load_index_from_storage(storage_context)

    simple_retriever = HybridOnlyRetriever(index, top_k=config.similarity_top_k)
    query_engine, _, _ = build_components(
        index,
        mode=config.default_mode,
        similarity_top_k=config.similarity_top_k,
        reranker_top_n=config.reranker_top_n,
        reranker_model=config.reranker_model,
    )

    from evaluator import load_golden_dataset

    cases = load_golden_dataset()

    # 评测控制
    EVAL_MODE = "full"   # "full" / "single" / "indices"
    EVAL_CASE_INDEX = 30     # single 模式用 
    EVAL_CASE_INDICES = [
        14,  # 道德经核心观点
        15,  # 土方修佛(已知架构天花板)
        27,  # 金刚经全称(合并后,验证是否修复)
        29,  # 庄子生死
        30,  # 什么是逍遥游
        32,  # 道德经和庄子无为异同
        34,  # 庄子开头第一篇(content_type 的重点验证)
    ]
    # 内部转成 0-indexed
    selected_cases = [cases[i - 1] for i in EVAL_CASE_INDICES]

    if EVAL_MODE == "single":
        if EVAL_CASE_INDEX < 0 or EVAL_CASE_INDEX >= len(cases):
            raise IndexError(f"EVAL_CASE_INDEX 越界: {EVAL_CASE_INDEX}, 当前 case 总数: {len(cases)}")
        selected_cases = [cases[EVAL_CASE_INDEX]]
    elif EVAL_MODE == "indices":
        for i in EVAL_CASE_INDICES:
            if i < 0 or i >= len(cases):
                raise IndexError(f"索引越界: {i}, 当前 case 总数: {len(cases)}")
        selected_cases = [cases[i] for i in EVAL_CASE_INDICES]
    else:
        selected_cases = cases

    logger.info(f"\nRunning evaluation [{run_name}]...\n")
    results = run_evaluation(
        query_engine=query_engine,
        llm=Settings.llm,
        cases=selected_cases,
        use_llm_judge=True, # 改成 False 可省 API 费用，只跑关键词指标
        index=index,
        simple_retriever=simple_retriever,
    )

    print_report(results, run_name=run_name)
    save_report(results, run_name=run_name)


if __name__ == "__main__":
    main()
