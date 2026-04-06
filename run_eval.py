'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-04 10:53:38
 * @Description  : 
'''
import os
import sys

from llama_index.core import Settings, StorageContext, load_index_from_storage

from config import configure_settings, load_config
from engine import HybridOnlyRetriever, build_components
from evaluator import print_report, run_evaluation, save_report


def main():
    config = load_config()
    configure_settings(config)
    run_name = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

    if not os.path.exists(config.storage_dir):
        print(f"Missing storage directory: {config.storage_dir}")
        return

    print("Loading index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=config.storage_dir)
    index = load_index_from_storage(storage_context)

    simple_retriever = HybridOnlyRetriever(index, top_k=config.similarity_top_k)
    query_engine, _, reranker = build_components(
        index,
        mode=config.default_mode,
        similarity_top_k=config.similarity_top_k,
        reranker_top_n=config.reranker_top_n,
        reranker_model=config.reranker_model,
    )

    from evaluator import load_golden_dataset

    cases = load_golden_dataset()

    # 评测控制
    EVAL_MODE = "full"      # 手动控制: "full" 或 "single"
    EVAL_CASE_INDEX = 5     # 仅在 single 模式下生效

    if EVAL_MODE == "single":
        if EVAL_CASE_INDEX < 0 or EVAL_CASE_INDEX >= len(cases):
            raise IndexError(f"EVAL_CASE_INDEX 越界: {EVAL_CASE_INDEX}, 当前 case 总数: {len(cases)}")
        selected_cases = [cases[EVAL_CASE_INDEX]]
    else:
        selected_cases = cases

    print(f"\nRunning evaluation [{run_name}]...\n")
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
