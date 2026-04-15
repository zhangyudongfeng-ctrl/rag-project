"""
router.py：意图分类 + 路由分发 + 三个handler
整个项目的数据格式就三种：chunk（字典）→ TextNode（加了id）→ NodeWithScore（加了分数）
"""

from typing import Tuple, Optional, Any
from llama_index.core import Settings
from engine import HybridOnlyRetriever
from llama_index.core.schema import NodeWithScore, TextNode
import logging
import time

logger = logging.getLogger(__name__)

# 一个工具函数，统一把NodeWithScore或TextNode格式的检索结果转换成前端需要的字典格式
# nodes: List[NodeWithScore] 或 List[TextNode]
# 注意区别, 如果是Nodewithscore类型,但是里面score是None, 处理成0.0, 如果是TextNode类型, 那么score直接设为None
def format_nodes_to_sources(nodes: list[NodeWithScore | TextNode] | None) -> list:
    if not nodes:
        return []
    # 把类型判断放到函数内部, 调用方不需要关心传入的nodes是带分数的NodeWithScore还是普通的TextNode, 函数内部会自动适配
    sources = []
    for n in nodes:
        if hasattr(n, "node") and hasattr(n, "score"):  # NodeWithScore
            node = n.node
            score = round(n.score, 4) if n.score is not None else 0.0
        else:
            node = n 
            score = None  

        sources.append({
            "score": score,
            "source_file": node.metadata.get("source_file", "未知来源"),
            "heading": node.metadata.get("heading", ""),
            "position": node.metadata.get("position", "未知位置"),
            "text": node.text[:300] + ("..." if len(node.text) > 300 else ""),
        })

    return sources

'''
 * @description: # 处理normal问题, 以字典格式返回，因为api端需要统一接收一种格式，保证3种路由返回json格式一致
 * @param {*} question
 * @param {*} query_engine
 * @return {*}
'''
def handle_normal(question: str, query_engine: Any) -> dict:
    response = query_engine.query(question)
    return {
            "answer": response.response,
            "sources": format_nodes_to_sources(response.source_nodes)
        }

'''
 * @description: 只返回结果位置的关键词
 * @param {*} raw
 * @param {*} question
 * @return {*}
'''
def normalize_position(raw, question):
    raw = (raw or "").strip()
    for key in ("开头", "结尾"):
        if key in raw:
            return key
    if question:
        if any(k in question for k in ["开头", "第一句", "最开始", "起头"]):
            return "开头"
        if any(k in question for k in ["结尾", "最后一句", "末尾", "最后"]):
            return "结尾"
    return raw


'''
 * @description: 进行soure_file和position的联合提取，提升过滤的准确率
 * @param {*} question
 * @return {*}
'''
def extract_position_and_source(question: str) -> Tuple[Optional[str], Optional[str]]:
    prompt = ("从问题中提取两个信息：\n"
              "1. 位置信息（开头/中间/结尾）\n"
              "2. 提到的书名或文档名关键词，没有则输出：无\n"
              "只输出两行，不要解释。\n"
              f"问题：{question}\n")

    try:
        response = Settings.llm.complete(prompt)
    except (TimeoutError, ConnectionError, ValueError) as e:
        logger.warning(f"extract_position_and_source LLM调用失败: {e}")
        return None, None
    
    # 清洗文本，防止 LLM 多输出乱码
    text = response.text.strip() 
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # 在取之前需要进行输出检验,避免因LLM输出不规范导致的报错
    if not lines:
        return None, None
    position = lines[0]

    # 提取 source，如果为空或"无"则设为 None
    source_raw = lines[1] if len(lines) > 1 else ""
    source = None if (source_raw == "无" or not source_raw) else source_raw

    return position, source

'''
 * @description: # 处理position问题,过滤时应该同时匹配position和source_file
 * @param {str} question
 * @param {Any} index
 * @param {Any} query_engine
 * @return {*}
'''
def handle_position(question: str, index : Any, query_engine : Any) -> dict:
        # 1. 提取目标位置
        position, source = extract_position_and_source(question)  # 数据此时是(position, source)
        target = normalize_position(position, question=question)
        logger.debug(f"提取到的位置: {target}, source hint: {source}")

        # 加上防御,如果LLM没有正确提取到位置，或者提取到了但不是开头或结尾，就当做普通问题处理
        if target not in ("开头", "结尾"):
            return handle_normal(question, query_engine)
        
        # 2. 过滤节点 (注意：这里直接操作 index.docstore.docs 比较重，但现阶段先跑通)
        all_nodes = index.docstore.docs.values()
        # 需要拿到source_file,这个信息在metadata里,但metadata又是存在TextNode里的,所以只能先拿到TextNode再比对metadata
        matched = [n for n in all_nodes if n.metadata.get("position") == target and (source in n.metadata.get("source_file", "") if source else True)]
        matched = [n for n in matched if n.metadata.get("content_type") == "main"]      # 过滤前言之类的无效内容

        # index.docstore.docs.values() 的顺序取决于插入顺序，可能不是严格按文档位置排的。加一行排序更稳
        matched.sort(key=lambda n: n.metadata.get("chunk_index", 0))
        # 3. 拼接上下文,根据不同的target位置取不同位置的切片,位置类问题一般选取5个切片就够了，太多反而可能干扰LLM判断，开头和结尾的切片往往比较短，信息量有限。
        if target == "结尾":
            matched = matched[-5:]
        elif target == "开头":
            matched = matched[:5]

        if not matched:
            return {
                "answer": "未找到符合位置条件的内容。",
                "sources": [],
            }
        
        context = "\n\n".join([n.text for n in matched])
        prompt = f"根据以下参考资料回答问题。\n\n参考资料：\n{context}\n\n问题：{question}\n回答："
        
        # 4. LLM 生成
        try:
            response_text = Settings.llm.complete(prompt).text
        except Exception:
            response_text = "已定位到相关片段，但生成答案时失败。"
        
        # 5. 返回统一格式 (调用刚才定义的工具方法)
        return {
            "answer": response_text,
            "sources": format_nodes_to_sources(matched)
        }

'''
 * @description: 处理跨文档检索问题
 * @param {str} question
 * @param {HybridOnlyRetriever} retriever
 * @param {FlagEmbeddingReranker} reranker
 * @return {dict} 自定义的json格式
'''
def handle_multi_doc(question: str, retriever: HybridOnlyRetriever)-> dict:
    # 把question拆分为多个子问题分别检索，最后合并
    prompt = ("把该问题拆分为2-3个独立的子问题，每行一个，不要编号，不要多余内容。\n"
          f"问题：{question}\n"
          "子问题：\n") # 即使这样写，LLM偶尔还是会输出编号、空行、解释性文字。
    try:
        response = Settings.llm.complete(prompt)
    except (TimeoutError, ConnectionError, ValueError) as e:
        logger.warning(f"handle_multi_doc-response LLM调用失败: {e}")
        return {"answer": "抱歉，服务暂时不可用，请稍后重试。", "sources": []}
    
    sub_questions = [line.strip() for line in response.text.strip().split(sep="\n") if line.strip()]    # 简单清洗
    # 每个子问题走现有检索管线（Hybrid Search + Reranker，不需要LLM）
    all_nodes = []
    for i, sub_question in enumerate(sub_questions):
        # ← 只检索，返回NodeWithScore列表
        nodes = retriever.retrieve(sub_question)
        # TODO 目前不用reranker方案,耗时太大 -> 子问题已经很精确了，hybrid检索的RRF融合排序就够用，直接取前3个
        # reranked = reranker.postprocess_nodes(nodes, query_str=sub_question)
        all_nodes.extend(nodes[:3]) # extend展平，不是append
    context = "\n\n".join([n.node.text for n in all_nodes])
    prompt = f"根据以下参考资料回答问题。\n\n参考资料：\n{context}\n\n问题：{question}\n回答："

    try:
        response_text = Settings.llm.complete(prompt).text
    except (TimeoutError, ConnectionError, ValueError) as e:
        logger.warning(f"handle_multi_doc-response_text LLM调用失败: {e}")
        return {"answer": "抱歉，服务暂时不可用，请稍后重试。", "sources": []}

    return {
        "answer": response_text,
        "sources": format_nodes_to_sources(nodes=all_nodes)
    }


# 输入：question, intent, index
# 输出：调用方要拿到回答展示给用户
# 过程：根据intent调用不同的处理函数
def route_query(question : str, intent : str, index : Any, query_engine : Any, retriever : Any) -> dict:
    print(f"  → intent: {intent}, question: {question[:30]}")
    # 根据map里的intent使用不同的路由,避免if elif的堆叠
    handler_map = {
        "position": lambda q: handle_position(q, index, query_engine),
        "multi_doc": lambda q: handle_multi_doc(q, retriever),
        "normal": lambda q: handle_normal(q, query_engine),
    }
    handler = handler_map.get(intent, handler_map["normal"])
    return handler(question)


'''
 * @description: 调试代码->1.看llm返回的response是什么格式,这决定了应该用什么方式分割它们->2.看检索后的nodes是什么格式，这决定了怎么从中提取文本和元信息
 * @return {*}
'''
if __name__ == "__main__":
    from llama_index.core.schema import TextNode

    # index.docstore.docs.values() 返回的是TextNode，不是NodeWithScore
    fake_nodes = [
        TextNode(text="土方岁三是新选组副长", metadata={"source_file": "shinsengumi.txt", "position": "结尾"}),
        TextNode(text="土方以铁律著称", metadata={"source_file": "shinsengumi.txt", "position": "开头"}),
        TextNode(text="四圣谛是佛教根本", metadata={"source_file": "buddhism.txt", "position": "中间"}),
        TextNode(text="天之道，利而不害；人之道，为而不争", metadata={"source_file": "道德经.txt", "position": "结尾"}),
    ]

    # 模拟extract_position_and_source的返回值
    target = "结尾"
    source_hint = "道德经"

    # 测过滤逻辑
    matched = [n for n in fake_nodes if n.metadata.get("position") == target]
    print(f"位置过滤后: {len(matched)}个")  # 期望2个（shinsengumi和道德经的结尾）

    if source_hint:
        filtered = [n for n in matched if source_hint in n.metadata.get("source_file", "")]
        if filtered:
            matched = filtered
    print(f"文档过滤后: {len(matched)}个")  # 期望1个（只有道德经）
    print(matched[0].text)  # 期望: 天之道...