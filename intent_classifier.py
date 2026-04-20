'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-05 15:57:23
 * @Description  : 
'''
from llama_index.core.prompts import PromptTemplate
from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

classify_prompt = PromptTemplate(
   "你是问题分类助手。先分析问题涉及哪些知识领域，再给出分类。\n\n"
    
    "【分类标准】\n"
    " position：问题包含明确位置词（最后一句、第一句、开头、结尾）\n"
    " multi_doc：回答该问题需要两个或以上不同领域的知识\n"
    " normal：其余所有问题\n\n"
    
    "【示例】\n"
    "问题：道德经最后一句是什么\n"
    "分析：包含位置词'最后一句' → position\n\n"
    
    "问题：佛陀对苦的根源怎么看\n"
    "分析：只涉及佛教一个领域 → normal\n\n"
    
    "问题：土方岁三是修佛的吗\n"
    "分析：土方岁三属于日本历史/小说，修佛属于佛教，需要两个领域 → multi_doc\n\n"
    
    "问题：尼采和帕斯卡尔对信仰的态度有何不同\n"
    "分析：尼采和帕斯卡尔属于不同哲学家，需要两个领域 → multi_doc\n\n"
    
    "问题：道德经中关于水的论述\n"
    "分析：只涉及道德经一个领域 → normal\n\n"
    
    '问题：{question}\n'
    '分析：'
)

# 输入：question
# 输出：问题的标签
# 过程：通过LLM对问题进行分类, 规则兜底, 主要有三类：正常问题normal、位置类问题position、跨文档类问题multi_doc. 只要分类失败,统一降级到normal, 避免路由失败导致整个查询失败
def classify_intent(question, llm) -> str:
    # 规则优先：position的关键词是有限的
    position_keywords = ["最后一句", "第一句", "开头", "结尾", "末尾"]
    if any(kw in question for kw in position_keywords):
        return "position"

    # 规则无法判断的，再交给LLM
    prompt = classify_prompt.format(question=question)
    try:
        response = llm.complete(prompt)
    except (TimeoutError, ConnectionError, ValueError) as e:
        logger.warning(f"classify_intent LLM调用失败: {e}")
        return "normal"   # 分类失败时降级到兜底路由
    text = response.text.strip()
    last_line = text.splitlines()[-1].strip().lower() if text else ""

    if "multi_doc" in last_line:
        return "multi_doc"
    if "position" in last_line:
        return "position"
    return "normal"