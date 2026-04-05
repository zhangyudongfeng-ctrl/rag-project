"""
query_rewriter.py：Query改写模块
职责：把用户的一个问题改写成多个不同角度的问题，提升检索覆盖面
"""
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate

REWRITE_PROMPT = """你是一个搜索查询改写助手。请将用户的问题改写为3个不同角度的搜索查询，用于在文档中检索相关内容。

要求：
1. 每个改写要用不同的关键词和表述方式
2. 覆盖问题可能涉及的不同方面
3. 只输出改写后的3个问题，每行一个，不要编号，不要其他内容

用户问题：{query}
"""


def multi_query_rewrite(query: str, llm: LLM) -> list[str]:
    """
    Multi-Query改写：把一个问题变成多个不同角度的问题

    输入：用户原始问题 + LLM实例
    输出：改写后的问题列表（包含原始问题，共4个）

    流程：
        1. 用LLM把原始问题改写成3个不同角度的查询
        2. 加上原始问题，共4个query
        3. 后续每个query分别检索，合并结果
    """
    # 调用LLM改写
    prompt = REWRITE_PROMPT.format(query=query)
    response = llm.complete(prompt)

    # 解析LLM返回的多行文本为列表
    rewritten = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line:
            rewritten.append(line)

    # 原始query + 改写query，确保原始问题不丢
    all_queries = [query] + rewritten
    return all_queries
