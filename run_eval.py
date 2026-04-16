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
from service import RagService

logger = logging.getLogger(__name__)

def main():
    config = load_config()
    configure_settings(config)
    rag_service = RagService(config)
    run_name = sys.argv[1] if len(sys.argv) > 1 else "unnamed"

    if not os.path.exists(config.storage_dir):
        logger.info(f"Missing storage directory: {config.storage_dir}")
        return

    logger.info("Loading index from storage...")

    components = rag_service.components

    from evaluator import load_golden_dataset

    cases = load_golden_dataset()

    # 评测控制
    EVAL_MODE = "single"   # "full" / "single" / "indices"
    EVAL_CASE_INDEX = 30     # single 模式用 
    EVAL_CASE_INDICES = [
        1, # 道德经中关于水的论述
        2, # 老子认为理想的统治者是什么样的？
        6, # 尼采说的超人是什么概念？
        8, # 帕斯卡尔怎么看人的处境？
        11, # 新选组的局中法度是什么？
        30, # 什么是逍遥游?
    ]   # 快速小批量测试normal性能的问题, 下标=索引-1
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
        components=components,
        llm=Settings.llm,
        cases=selected_cases,
        use_llm_judge=True, # 改成 False 可省 API 费用，只跑关键词指标
    )

    print_report(results, run_name=run_name)
    save_report(results, run_name=run_name)


if __name__ == "__main__":
    main()
