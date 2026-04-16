"""
RAG 评估模块
用法：放到 D:\rag-project下，运行 python evaluator.py

功能：
    1. 管理 golden dataset（测试用例集）
    2. 评估检索质量：命中率、MRR
    3. 评估生成质量：忠实度、相关性（LLM-as-Judge）
    4. 输出对比报告，量化每次改动的效果

核心原则：先有评估再优化，否则一切改动都是盲人摸象。
"""

import os
import json
import csv
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple
from datetime import datetime
from router import  route_query
from intent_classifier import classify_intent
from rag_compotents import RagComponents


# ==========================================
# 数据结构：一条测试用例长什么样
# ==========================================
@dataclass
class TestCase:
    """
    一条评估用例。
    
    为什么需要这些字段：
        - question: 用户问题
        - expected_answer: 期望的正确答案（人工标注）
        - expected_keywords: 答案中应该出现的关键词（用于自动判断命中）
        - expected_source: 期望检索命中的文件名（用于评估检索质量）
        - category: 问题类别，方便按类型分析弱项
        - difficulty: 难度标记
    """
    question: str
    expected_answer: str                # 人工校准
    expected_keywords: List[str]        # 答案中应包含的关键词
    expected_source: str = ""           # 期望的来源文件
    category: str = ""                  # 问题类型：事实/推理/对比/位置/开放
    difficulty: str = "normal"          # easy / normal / hard


@dataclass
class EvalResult:
    """一条用例的评估结果"""
    question: str
    expected_answer: str
    actual_answer: str
    
    # 检索指标
    retrieval_hit: bool             # 期望来源是否在检索结果中
    top1_source: str                # 排名第一的来源文件
    top1_score: float               # 排名第一的相关度分数
    retrieved_sources: List[str]    # 所有检索到的来源
    
    # 生成指标
    keyword_recall: Optional[float] # 关键词命中率（0-1）
    faithfulness: float             # 忠实度评分（LLM打分，1-5）
    relevancy: float                # 相关性评分（LLM打分，1-5）
    
    # 元信息
    category: str = ""                                              # 所属类别
    latency_seconds: float = 0.0                                    # 回答耗时
    retrieved_chunks: List[str] = field(default_factory=list)       # 检索到了什么chunk
    prompt_tokens: int = 0                                          # prompt大小
    completion_tokens: int = 0                                      # 回答消耗的tokens
    context_length: int = 0         # retrieval->context->generation中的context分析，在出现问题时有更完全的排查过程


# ==========================================
# Golden Dataset 管理
# ==========================================
GOLDEN_FILE = "golden_dataset.json"


def create_golden_dataset() -> List[TestCase]:
    """
    初始 golden dataset。
    初始兜底数据集。仅在 golden_dataset.json 不存在时使用。
    真源是 golden_dataset.json，请直接编辑 json 文件。
    
    构建原则：
        1. 覆盖不同问题类型（事实/推理/对比/开放）
        2. 覆盖不同文档来源
        3. 包含已知的难case（从 test_cases.tsv 迁移）
        4. 每条都有人工标注的期望答案和关键词
        5. 目标：50-100条，先从20条开始
    
    关键词选取原则：
        - 选答案中最核心的实体/概念词
        - 不选虚词和通用词
        - 3-5个关键词为宜
    """
    cases = [
        # === 道德经 ===
        TestCase(
            question="道可道非常道是什么意思？",
            expected_answer="可以用言语表达的道，就不是永恒不变的道。可以用名称称呼的名，就不是永恒不变的名。",
            expected_keywords=["言语", "表达", "永恒", "道", "名"],
            expected_source="道德经",
            category="事实",
        ),
        TestCase(
            question="道德经中关于水的论述",
            expected_answer="上善若水。水善利万物而不争，处众人之所恶，故几于道。",
            expected_keywords=["上善若水", "不争", "万物"],
            expected_source="道德经",
            category="事实",
        ),
        TestCase(
            question="老子认为理想的统治者是什么样的？",
            expected_answer="太上，不知有之。最好的统治者，人民不知道他的存在。其次亲之誉之，再次畏之侮之。",
            expected_keywords=["不知有之", "统治", "太上"],
            expected_source="道德经",
            category="推理",
        ),

        # === 佛经 ===
        TestCase(
            question="什么是四圣谛？",
            expected_answer="苦谛、集谛、灭谛、道谛。苦是生命的本质，集是苦的原因，灭是苦的终止，道是灭苦的方法。",
            expected_keywords=["苦", "集", "灭", "道", "四谛"],
            expected_source="佛教十三经",
            category="事实",
        ),
        TestCase(
            question="色即是空是什么意思？",
            expected_answer="一切物质现象（色）本质上是空的，没有独立不变的自性。空不是没有，而是指事物的无自性。",
            expected_keywords=["色", "空", "自性", "物质"],
            expected_source="佛教十三经",
            category="事实",
        ),
        TestCase(
            question="佛教中有漏和无漏是什么意思？",
            expected_answer="有漏指有烦恼、有缺陷的状态，漏是烦恼的别名。无漏指断除烦恼、清净无染的境界。",
            expected_keywords=["烦恼", "漏", "有漏", "无漏"],
            expected_source="佛教十三经",
            category="事实",
        ),

        # === 查拉图斯特拉如是说（尼采）===
        TestCase(
            question="尼采说的超人是什么概念？",
            expected_answer="超人是人类应当超越自身的目标。人是猿猴到超人之间的一根绳索。超人是大地的意义。",
            expected_keywords=["超人", "超越", "绳索", "大地"],
            expected_source="查拉图斯特拉如是说",
            category="事实",
        ),
        TestCase(
            question="永恒回归是什么意思？",
            expected_answer="一切事物将以完全相同的方式无限次重复发生。这是对生命最大的肯定——你是否愿意让此刻永远重复。",
            expected_keywords=["永恒", "重复", "回归", "肯定"],
            expected_source="查拉图斯特拉如是说",
            category="推理",
        ),

        # === 思想录（帕斯卡尔）===
        TestCase(
            question="帕斯卡尔怎么看人的处境？",
            expected_answer="人是一根会思想的芦苇，是自然界最脆弱的东西。但人的全部尊严就在于思想。",
            expected_keywords=["芦苇", "思想", "脆弱", "尊严"],
            expected_source="思想录",
            category="事实",
        ),
        TestCase(
            question="帕斯卡尔的赌注论证是什么？",
            expected_answer="如果信上帝，上帝存在则获得无限幸福；不存在也没什么损失。如果不信，上帝存在则失去一切。理性选择是信。",
            expected_keywords=["上帝", "赌注", "信", "理性"],
            expected_source="思想录",
            category="推理",
        ),

        # === 燃烧吧！剑（司马辽太郎）===
        TestCase(
            question="土方岁三是谁？",
            expected_answer="新选组副长，幕末武士。出身武州多摩，性格刚烈，以铁律维持新选组纪律。",
            expected_keywords=["新选组", "副长", "土方"],
            expected_source="燃烧吧！剑",
            category="事实",
        ),
        TestCase(
            question="新选组的局中法度是什么？",
            expected_answer="新选组内部的严格规章制度，违反者切腹。包括不得私自脱离、不得私自筹款等条目。",
            expected_keywords=["法度", "切腹", "规章", "新选组"],
            expected_source="燃烧吧！剑",
            category="事实",
        ),

        # === 跨文档对比（难题）===
        TestCase(
            question="道德经的'无为'和佛教的'空'有什么关系？",
            expected_answer="道德经的无为是不妄为、顺应自然；佛教的空是指事物无自性。两者都指向放下执着，但出发点不同。",
            expected_keywords=["无为", "空", "自然", "执着"],
            expected_source="",  # 跨文档，不指定单一来源
            category="对比",
            difficulty="hard",
        ),
        TestCase(
            question="尼采和帕斯卡尔对信仰的态度有何不同？",
            expected_answer="帕斯卡尔认为理性指向信仰上帝，尼采宣告上帝已死，人应自我超越。",
            expected_keywords=["上帝", "信仰", "超越"],
            expected_source="",
            category="对比",
            difficulty="hard",
        ),

        # === 已知难case（从建立的EXCEL中迁移）===
        TestCase(
            question="道德经的核心观点是什么？",
            expected_answer="道法自然、无为而治、柔弱胜刚强。道是万物本源，人应顺应自然之道。",
            expected_keywords=["道法自然", "无为", "柔弱"],
            expected_source="道德经",
            category="开放",
            difficulty="hard",
        ),
        TestCase(
            question="道德经的主旨是什么？",
            expected_answer="与'核心观点'同义，测试同义问题检索稳定性。",
            expected_keywords=["道法自然", "无为", "柔弱"],
            expected_source="道德经",
            category="开放",
            difficulty="hard",
        ),

        # === 跨文档推理 ===
        TestCase(
            question="土方岁三是修佛的吗？",
            expected_answer="文档中没有相关内容。土方岁三是武士，文档中未提及他修佛。",
            expected_keywords=["没有", "武士"],
            expected_source="燃烧吧！剑",
            category="跨文档",
            difficulty="hard",
        ),
        TestCase(
            question="土方岁三的人生在佛学中可以认为他修行的是顶级的有为法吗？虽然高级，但终究不是无为法",
            expected_answer="文档中没有将土方岁三与佛学联系的内容。这是跨文档推理，需要分别从两本书检索再综合。",
            expected_keywords=["有为法", "无为法"],
            expected_source="",
            category="跨文档",
            difficulty="hard",
        ),

        # === 精确定位 ===
        TestCase(
            question="一切有为法，如梦幻泡影。后一句是什么？",
            expected_answer="如露亦如电，应作如是观。",
            expected_keywords=["如露", "如电", "如是观"],
            expected_source="佛教十三经",
            category="精确定位",
        ),
        TestCase(
            question="诸幻尽灭，不入断灭。出自哪里？",
            expected_answer="出自《圆觉经》。",
            expected_keywords=["圆觉经"],
            expected_source="佛教十三经",
            category="精确定位",
        ),
        TestCase(
            question="道德经最后一句话是什么？",
            expected_answer="圣人之道，为而不争。",
            expected_keywords=["圣人之道", "为而不争"],
            expected_source="道德经",
            category="精确定位",
            difficulty="hard",
        ),

        # === 进一步推理 ===
        TestCase(
            question="为何土方要杀六车宗伯？",
            expected_answer="需要从小说具体情节中检索，涉及人物冲突和剧情推理。",
            expected_keywords=["土方", "六车"],
            expected_source="燃烧吧！剑",
            category="推理",
        ),

        # === 幻觉测试（期望系统诚实说不知道）===
        TestCase(
            question="言语道断心行处灭出自哪里？",
            expected_answer="出自《维摩诘所说经》或佛经。系统应基于文档回答，如果文档中没有则说不知道，不要编造。",
            expected_keywords=["言语道断"],
            expected_source="佛教十三经",
            category="幻觉测试",
        ),
        TestCase(
            question="文档中对量子力学有什么看法？",
            expected_answer="文档中没有关于量子力学的内容。系统应诚实回答'没有相关内容'。",
            expected_keywords=["没有"],
            expected_source="",
            category="幻觉测试",
        ),
        TestCase(
            question="真空生妙有，妙有归真空，出自文档中的哪部经书？",
            expected_answer="系统应基于文档检索回答，若文档中没有原文则诚实说明。",
            expected_keywords=["真空", "妙有"],
            expected_source="佛教十三经",
            category="幻觉测试",
        ),

        # === 同一问题不同问法 ===
        TestCase(
            question="土方岁三是燃烧吧！剑的主角吗？",
            expected_answer="是的，土方岁三是《燃烧吧！剑》的主角。",
            expected_keywords=["土方", "主角"],
            expected_source="燃烧吧！剑",
            category="同义问法",
        ),
        TestCase(
            question="燃烧吧！创这本小说的主角是谁？",
            expected_answer="土方岁三。测试同一问题不同问法的检索稳定性。",
            expected_keywords=["土方"],
            expected_source="燃烧吧！剑",
            category="同义问法",
        ),

        # === 模糊问题 ===
        TestCase(
            question="金刚经的主旨是什么？核心内容是什么？",
            expected_answer="金刚经的核心是破除一切执着，包括对法的执着。凡所有相皆是虚妄，应无所住而生其心。",
            expected_keywords=["执着", "虚妄", "无所住"],
            expected_source="佛教十三经",
            category="开放",
            difficulty="hard",
        ),

        # === 歧义理解 ===
        TestCase(
            question="金刚经是谁在说经，全名叫什么？",
            expected_answer="释迦牟尼佛（佛陀）在说经，全名为《金刚般若波罗蜜经》。问题有歧义：'全名'可能指说经人或经书。",
            expected_keywords=["释迦", "金刚般若波罗蜜"],
            expected_source="佛教十三经",
            category="歧义",
        ),
        TestCase(
            question="金刚经是谁在说经，说经人的全名叫什么？",
            expected_answer="释迦牟尼佛。消除歧义后的版本，对比上一条看检索差异。",
            expected_keywords=["释迦牟尼"],
            expected_source="佛教十三经",
            category="歧义",
        ),
    ]
    return cases

# 存储测试用例
def save_golden_dataset(cases: List[TestCase], filepath: str = GOLDEN_FILE):
    """保存 golden dataset 到 JSON"""
    data = [asdict(c) for c in cases]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存 {len(cases)} 条测试用例到 {filepath}")

# 读取测试用例
def load_golden_dataset(filepath: str = GOLDEN_FILE) -> List[TestCase]:
    """加载 golden dataset"""
    if not os.path.exists(filepath):
        print(f"未找到 {filepath}，创建初始数据集...")
        cases = create_golden_dataset()
        save_golden_dataset(cases)
        return cases

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [TestCase(**d) for d in data]    # 把字典拆成参数传进去 -- 字典解包


# ==========================================
# 检索评估指标
# ==========================================
def eval_retrieval_hit(expected_source: str, retrieved_sources: List[str]) -> bool:
    """
    检索命中：期望的来源文件是否在检索结果中。
    
    用模糊匹配：expected_source 是子串即算命中。
    比如 expected_source="道德经"，retrieved_source="道德经（张景、张松辉）.txt" → 命中
    """
    if not expected_source:
        return True  # 跨文档题不评检索来源
    
    for source in retrieved_sources:
        if expected_source in source:
            return True
    return False

# 跳过开放问题的召回率计算
# 返回值可能是 float，也可能是 None, 后续需要处理为None的情况
def eval_keyword_recall(case: TestCase, actual_answer: str) -> Optional[float]:
    """
    关键词召回率：期望关键词在实际答案中出现了多少。
    
    为什么用这个：
        - 简单、快速、不依赖 LLM
        - 作为生成质量的粗筛指标
        - 配合 LLM-as-Judge 使用，互相校验
    """
    # 开放题跳过关键词评估
    if case.category == "开放":
        return None  # 或 "N/A"
    # TODO如果没有关键词，默认满分 --- 有隐患
    if not case.expected_keywords:
        return 1.0
    
    hits = sum(1 for kw in case.expected_keywords if kw in actual_answer)
    return hits / len(case.expected_keywords)


# ==========================================
# 生成评估：LLM-as-Judge
# ==========================================
def eval_faithfulness(question: str, answer: str, context: str, llm) -> float:
    """
    忠实度评分：答案是否基于检索到的上下文，有没有编造。
    
    用 LLM 给 LLM 打分（LLM-as-Judge）。
    返回 1-5 分。
    
    面试考点：为什么用 LLM 评估而不是人工？
        - 人工标注成本高，无法每次迭代都做
        - LLM-as-Judge 与人类评分相关性在0.8以上（论文验证）
        - 适合快速迭代阶段，上线前再做人工抽检
    """
    prompt = f"""你是一个严格的评估专家。请评估以下回答的忠实度。

忠实度定义：回答是否严格基于给定的参考资料，没有编造资料中不存在的信息。

参考资料：
{context[:2000]}

问题：{question}
回答：{answer}

评分标准：
1分：完全编造，答案与资料无关
2分：大部分内容编造，仅少量与资料相关
3分：有部分内容基于资料，但也有明显编造
4分：基本忠实于资料，仅有细微推理超出
5分：完全忠实于资料，所有信息都有据可查

请只返回一个数字（1-5），不要解释。
评分："""

    try:
        response = llm.complete(prompt)
        score = float(response.text.strip()[:1])
        return min(max(score, 1), 5)
    except Exception:
        return 0


def eval_relevancy(question: str, answer: str, llm) -> float:
    """
    相关性评分：答案是否回答了用户的问题。
    
    与忠实度的区别：
        - 忠实度：答案 vs 检索到的资料（有没有编）
        - 相关性：答案 vs 用户问题（有没有答到点上）
        
    一个忠实但不相关的例子：
        问"老子怎么看水"，检索到道德经第一章，
        回答了"道可道非常道"——忠实于资料但没回答问题。
    """
    prompt = f"""你是一个严格的评估专家。请评估以下回答与问题的相关性。

相关性定义：回答是否直接回答了用户的问题，信息是否对用户有帮助。

问题：{question}
回答：{answer}

评分标准：
1分：完全无关，答非所问
2分：擦边相关，没有直接回答
3分：部分回答了问题，但不完整
4分：基本回答了问题，较为完整
5分：精准回答问题，内容完整且有帮助

请只返回一个数字（1-5），不要解释。
评分："""

    try:
        response = llm.complete(prompt)
        score = float(response.text.strip()[:1])
        return min(max(score, 1), 5)
    except Exception:
        return 0


@dataclass
class RAGContext:
    """存放从 Response 解析出来的所有关键信息的容器"""
    actual_answer: str
    source_nodes: list
    retrieved_chunks: List[str]
    retrieved_sources: List[str]
    top1_source: str
    top1_score: float
    context: str
    prompt_tokens: int
    completion_tokens: int

# 处理Response对象，输出一个规范化的RAGContext
def parse_response(response) -> RAGContext:
    nodes = response.source_nodes or []
    
    # 提取信息
    chunks = [n.text for n in nodes]
    # 把一堆 Node 对象转化成一堆字符串（文件名）
    sources = [n.metadata.get("source_file", "") for n in nodes]

    # 确保 metadata 存在metadata，or的作用是处理属性存在但值为 None
    meta = getattr(response, "metadata", {}) or {}
    
    # 构造并返回结构化对象
    return RAGContext(
        actual_answer=str(response),
        source_nodes=nodes,
        retrieved_chunks=chunks,
        retrieved_sources=sources,
        top1_source=sources[0] if sources else "",
        top1_score=nodes[0].score if nodes else 0.0,
        context="\n".join(chunks),
        # token 统计：从 response 的元数据中提取
        # 安全获取 Token 信息，处理可能为空的情况--- LlamaIndex 的 Response 对象在 metadata 中可能携带 token 信息，所以有个兜底 or 0
        prompt_tokens = meta.get('prompt_tokens') or 0,
        completion_tokens = meta.get('completion_tokens') or 0,
    )

# 检索评估和生成评估
# 输入：case, RAGContext，llm, 可选参数(是否使用LLM评估)use_llm_judge
# 输出：检索是否命中，关键词召回率，忠实度评分，相关性评分
def search_generate_evaluation(case, ctx, llm, use_llm_judge) -> Tuple[bool, Optional[float], float, float]:
    # 检索评估
    hit = eval_retrieval_hit(case.expected_source, ctx.retrieved_sources)
    kw_recall = eval_keyword_recall(case, ctx.actual_answer)

    # 生成评估（可选）
    faithfulness = 0.0
    relevancy = 0.0
    if use_llm_judge:
        faithfulness = eval_faithfulness(case.question, ctx.actual_answer, ctx.context, llm)
        relevancy = eval_relevancy(case.question, ctx.actual_answer, llm)
    return hit, kw_recall, faithfulness, relevancy

# ==========================================
# 核心：运行评估 -- 主要做了3件事：提取检索信息、统计tokens、检索评估和生成评估
# ==========================================
def run_evaluation(components: "RagComponents", llm, cases: List[TestCase] = None, 
                   use_llm_judge: bool = True) -> List[EvalResult]:
    """
    对 golden dataset 跑完整评估。
    
    参数：
        llm: 用于 LLM-as-Judge 的模型
        cases: 测试用例，默认从 golden_dataset.json 加载
        use_llm_judge: 是否启用 LLM 打分（关闭可省 API 费用，只跑关键词指标）
        components: RagComponents
    """
    if cases is None:
        cases = load_golden_dataset()

    results = []
    total = len(cases)

    for i, case in enumerate(cases):
        print(f"[{i+1}/{total}] {case.question[:30]}...", end=" ", flush=True)

        start_time = time.time()
            
        intent = classify_intent(case.question)
        result = route_query(case.question, intent, components)
        latency = time.time() - start_time
    
        # 从字典构造RAGContext，绕过parse_response
        sources = result["sources"]
        ctx = RAGContext(
            actual_answer=result["answer"],
            source_nodes=[],
            retrieved_chunks=[s["text"] for s in sources],
            retrieved_sources=[s["source_file"] for s in sources],
            top1_source=sources[0]["source_file"] if sources else "",
            top1_score=sources[0]["score"] if sources else 0.0,
            context="\n".join(s["text"] for s in sources),
            prompt_tokens=0,
            completion_tokens=0,
        )

        # 检索和生成评估
        hit, kw_recall, faithfulness, relevancy = search_generate_evaluation(case, ctx, llm, use_llm_judge)

        result = EvalResult(
            question=case.question,
            expected_answer=case.expected_answer,
            actual_answer=ctx.actual_answer,
            retrieval_hit=hit,
            top1_source=ctx.top1_source,
            top1_score=ctx.top1_score,
            retrieved_sources=ctx.retrieved_sources,
            keyword_recall=kw_recall,
            faithfulness=faithfulness,
            relevancy=relevancy,
            category=case.category,
            latency_seconds=latency,
            retrieved_chunks=ctx.retrieved_chunks,
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=ctx.completion_tokens,
            context_length=len(ctx.context),
        )
        results.append(result)

        if kw_recall is not None:
            status = "✅" if hit and kw_recall > 0.5 else "❌"
            print(f"{status} 命中:{hit} 关键词:{kw_recall:.0%} 忠实:{faithfulness:.0f} 相关:{relevancy:.0f} ({latency:.1f}s)")
        else:
        # 开放题只看忠实度和相关性
            status = "✅" if hit and faithfulness >= 4 else "❌"
            print(f"{status} 命中:{hit} 关键词:N/A 忠实:{faithfulness:.0f} 相关:{relevancy:.0f} ({latency:.1f}s)")

    return results


def is_failure(r):
    if not r.retrieval_hit:
        return True
    if r.keyword_recall is None:  # 开放题
        return r.faithfulness < 4
    return r.keyword_recall < 0.3

# ==========================================
# 报告生成
# 先看总体平均指标->再看tokens->难易度不同的类别分析->看失败案例
# ==========================================
def print_report(results: List[EvalResult], run_name: str = ""):
    """打印评估报告"""
    total = len(results)
    if total == 0:
        print("无评估结果")
        return

    # 总体指标
    if total > 0:
        hit_rate = sum(1 for r in results if r.retrieval_hit) / total
        # 过滤召回率为None的情况
        recalls = [r.keyword_recall for r in results if r.keyword_recall is not None]
        avg_kw = sum(recalls) / len(recalls) if recalls else 0.0
        avg_faith = sum(r.faithfulness for r in results) / total
        avg_rel = sum(r.relevancy for r in results) / total
        avg_latency = sum(r.latency_seconds for r in results) / total
    else:
        hit_rate = 0.0
        avg_kw = 0.0
        avg_faith = 0.0
        avg_rel = 0.0
        avg_latency = 0.0

    print(f"\n{'='*60}")
    print(f"评估报告" + (f"  [{run_name}]" if run_name else ""))
    print(f"{'='*60}")
    print(f"总用例数:     {total}")
    print(f"检索命中率:   {hit_rate:.1%}")
    print(f"关键词召回:   {avg_kw:.1%}")
    print(f"忠实度均分:   {avg_faith:.1f}/5")
    print(f"相关性均分:   {avg_rel:.1f}/5")
    print(f"平均延迟:     {avg_latency:.1f}s")
    # token 和 context 统计
    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_completion_tokens = sum(r.completion_tokens for r in results)
    if total > 0:
        avg_context_len = sum(r.context_length for r in results) // total
    else:
        avg_context_len = 0
    print(f"Prompt tokens: {total_prompt_tokens:,} (总计)")
    print(f"输出 tokens:   {total_completion_tokens:,} (总计)")
    print(f"平均context:   {avg_context_len:,} 字符")
    # 按类别分析
    categories = set(r.category for r in results if r.category)
    if categories:
        print(f"\n--- 按类别分析 ---")
        for cat in sorted(categories):
            cat_results = [r for r in results if r.category == cat]
            cat_hit = sum(1 for r in cat_results if r.retrieval_hit) / len(cat_results)
            # 过滤召回率为None的情况
            cat_recalls = [r.keyword_recall for r in cat_results if r.keyword_recall is not None]
            cat_kw = sum(cat_recalls) / len(cat_recalls) if cat_recalls else 0.0
            print(f"  {cat:<6} ({len(cat_results)}条): 命中={cat_hit:.0%}  关键词={cat_kw:.0%}")

    # 失败案例
    failures = [r for r in results if is_failure(r)]
    if failures:
        print(f"\n--- 失败案例 ({len(failures)}条) ---")
        for r in failures:
            print(f"  ❌ {r.question[:40]}")
            kw_str = "N/A" if r.keyword_recall is None else f"{r.keyword_recall:.0%}"
            print(f"     命中:{r.retrieval_hit} 关键词:{kw_str} 来源:{r.top1_source}")
            print(f"     context:{r.context_length}字符 | tokens: prompt={r.prompt_tokens} completion={r.completion_tokens}")
            # 显示第一个chunk的前100字符，快速判断检索对不对
            if r.retrieved_chunks:
                print(f"     top1_chunk: {r.retrieved_chunks[0][:100]}...")

    print(f"{'='*60}\n")


def save_report(results: List[EvalResult], run_name: str = "mainV3"):
    """保存评估结果到 CSV，方便跨版本对比"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{run_name}_{timestamp}.csv" if run_name else f"eval_{timestamp}.csv"

    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "category", "retrieval_hit", "top1_source", "top1_score",
            "keyword_recall", "faithfulness", "relevancy", "latency",
            "prompt_tokens", "completion_tokens", "context_length",
            "expected_answer", "actual_answer", "retrieved_chunks"
        ])
        for r in results:
            writer.writerow([
                r.question, r.category, r.retrieval_hit, r.top1_source,
            f"{r.top1_score:.3f}" if r.top1_score is not None else "N/A",
            f"{r.keyword_recall:.2f}" if r.keyword_recall is not None else "N/A",
            f"{r.faithfulness:.0f}", f"{r.relevancy:.0f}",
            f"{r.latency_seconds:.1f}",
            r.prompt_tokens, r.completion_tokens, r.context_length,
            r.expected_answer, r.actual_answer,
            "|||".join(c[:200] for c in r.retrieved_chunks)  # chunk用|||分隔，每个截200字符
            ])

    print(f"评估结果已保存: {filename}")
    return filename


# ==========================================
# 对比工具：两次评估结果放一起看
# ==========================================
def compare_runs(results_a: List[EvalResult], results_b: List[EvalResult],
                 name_a: str = "A", name_b: str = "B"):
    """对比两次评估，看哪些指标变好了哪些变差了"""
    def avg(lst, attr):
        vals = [getattr(r, attr) for r in lst]
        vals = [v for v in vals if v is not None]  # 过滤 None
        return sum(vals) / len(vals) if vals else 0

    metrics = ["keyword_recall", "faithfulness", "relevancy"]
    
    print(f"\n{'指标':<16} {name_a:>10} {name_b:>10} {'变化':>10}")
    print("-" * 50)
    
    # 命中率
    hit_a = sum(1 for r in results_a if r.retrieval_hit) / len(results_a)
    hit_b = sum(1 for r in results_b if r.retrieval_hit) / len(results_b)
    delta = hit_b - hit_a
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"{'检索命中率':<12} {hit_a:>10.1%} {hit_b:>10.1%} {arrow}{abs(delta):>8.1%}")

    for m in metrics:
        va = avg(results_a, m)
        vb = avg(results_b, m)
        delta = vb - va
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        label = {"keyword_recall": "关键词召回", "faithfulness": "忠实度", "relevancy": "相关性"}[m]
        fmt = ".1%" if m == "keyword_recall" else ".1f"
        print(f"{label:<12} {va:>10{fmt}} {vb:>10{fmt}} {arrow}{abs(delta):>8{fmt}}")


# ==========================================
# 主入口：单独运行时初始化 golden dataset
# ==========================================
if __name__ == "__main__":
    print("=== RAG 评估模块 ===\n")

    # 初始化/加载 golden dataset
    cases = load_golden_dataset()
    print(f"\n加载 {len(cases)} 条测试用例")

    # 按类别统计
    categories = {}
    for c in cases:
        cat = c.category or "未分类"
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n用例分布:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}条")

    print(f"\n如需运行评估，在 main.py 中调用:")
    print(f"  from evaluator import run_evaluation, print_report, save_report")
    print(f"  results = run_evaluation(query_engine, Settings.llm)")
    print(f"  print_report(results, run_name='v3_baseline')")
    print(f"  save_report(results, run_name='v3_baseline')")