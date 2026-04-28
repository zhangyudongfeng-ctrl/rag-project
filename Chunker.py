"""
切片模块：多策略切片 + 元数据保留 + 策略对比
用法：放到 D:\rag-project 下
    - 单独运行 python chunker.py 可预览切片效果
    - 在 main.py 中 import 使用

支持三种策略：
    1. fixed     — 固定字符数切片（基线）
    2. paragraph — 按段落边界切片，尊重文本自然结构
    3. heading   — 按标题切片

核心思路：
    清洗后的txt → 按策略切成chunks → 每个chunk附带元数据 → 喂给LlamaIndex建索引

# 已知边界(content_type 状态机):
# 1. AUXILIARY_MARKERS 不可能穷尽,标题用罕见词的辅助章节会被错标
#    例:庄子的"庄子的艺术特色"导读
# 2. 状态机只看 chunk 前5行, heading如果出现在大于5行的位置, 这种情况会被忽略
#    例:庄子题解 chunk 第一行是日期"二〇一〇年一月",第6行才是 ### 【题解】
# 应对:这两类罕见 case 通过 golden_dataset 的 category 标注绕过,不打补丁


"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

# LlamaIndex 相关（在 main.py 中集成时使用）
from llama_index.core.schema import TextNode
import logging
logger = logging.getLogger(__name__)

AUXILIARY_MARKERS = ["前言", "序言", "序", "导论", "题解", "译注", "注释", "后记", "附录", "目录", "凡例"]

# 状态机扫描 chunk 前 N 行寻找 heading,N 设置为 5 是因为:
# - 1 行不够(日期/空行可能占据第一行)
# - 10 行太宽(可能误识别正文中的子标题)
# - 5 是真实文本里"前导内容 + heading"的常见上限
# 如果将来遇到反例,优先改这个常量,不要重写状态机逻辑
HEADING_SCAN_LINES = 5

# ==========================================
# 数据结构：一个chunk长什么样
# ==========================================
@dataclass
class Chunk:
    """一个切片单元"""
    text: str                          # 切片内容
    metadata: dict = field(default_factory=dict)  # 元数据，保证每次创建新Chunk时，不共用一个字典对象

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        preview = self.text[:50].replace('\n', '\\n')
        return f"Chunk({len(self.text)}字符, '{preview}...')"


'''
 * @description: 识别chunk中的标题类内容,提取并添加进元数据. 只要不在AUXILIARY_MARKERS里的就算是正文->目前只能这样取舍了,过滤掉一些chunk,但不可能完全过滤
 * @param {Chunk} chunks
 * @return {Chunk} chunks
'''
def propagate_content_type(chunks: List[Chunk]) -> List[Chunk]:
    current = "main"  # 默认正文
    
    for chunk in chunks:
        # 旧方案->只看第一行 -> 需要修改,最多看前五行
        # first_line = head.split(sep="\n")[0].strip()
        # if first_line.startswith("#") or first_line.startswith("【"):
        #     if any(m in first_line for m in AUXILIARY_MARKERS):
        #         current = "auxiliary"
        #     else:
        #         current = "main"

        # 新方案
        lines = chunk.text.split("\n")[:HEADING_SCAN_LINES]
        heading_line = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("【"):
                heading_line = line
                break
        
        if heading_line:
            if any(m in heading_line for m in AUXILIARY_MARKERS):
                current = "auxiliary"
            else:
                current = "main"
        chunk.metadata["content_type"] = current
    
    return chunks

# ==========================================
# 策略一：固定大小切片（基线）
# 从头到尾滑动窗口，每次走chunk_size步，退回overlap步，尽量在句号处断开。
# ==========================================
def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 100) -> List[Chunk]:
    """
    按固定字符数切片，带重叠。
    
    这是最简单的策略，也是对比基线。
    .
    问题：会在句子中间断开，破坏语义完整性。
    
    参数：

        chunk_size: 每个切片的目标字符数
        overlap:    相邻切片的重叠字符数，防止边界信息丢失
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # 如果不是最后一片，尝试在句号/换行处断开，避免切断句子
        if end < len(text):
            # 从 end 往前找最近的句子边界
            boundary = _find_sentence_boundary(text, end, lookback=100)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"chunk_index": len(chunks), "start_char": start, "end_char": end}
            ))

        # 下一片的起点 = 当前终点 - 重叠
        start = end - overlap
        # 防止死循环：如果没有前进，强制推进
        if start <= chunks[-1].metadata["start_char"] if chunks else True:
            start = end

    return chunks


# ==========================================
# 策略二：段落感知切片（推荐）
# ==========================================
def chunk_by_paragraph(text: str, max_size: int = 512, min_size: int = 50) -> List[Chunk]:
    """
    按段落边界切片，尊重文本的自然结构。
    
    逻辑：
        1. 先按双换行（段落边界）拆成段落列表
        2. 贪心合并：把短段落合并到一起，直到接近 max_size
        3. 超长段落：按句号二次切分
    
    为什么推荐：
        - 中文书籍段落通常是完整的语义单元
        - 不会在句子中间断开
        - 保留了作者的分段意图
    
    参数：
        max_size: 每个切片的最大字符数（软上限，不会在句子中间切）
        min_size: 低于此长度的切片会被合并到相邻切片
    """
    # 第一步：按段落拆分
    paragraphs = _split_paragraphs(text)

    # 第二步：贪心合并
    chunks = []
    buffer = ""

    # 因为bge-small-zh的max_sequence_length是512 tokens，超过的部分直接截断，等于白存
    # 所以把一个超长的段落按句子切
    for para in paragraphs:
        # 如果当前段落本身就超长，先单独处理
        if len(para) > max_size:
            # 先把 buffer 里积累的内容输出
            if buffer.strip():
                chunks.append(Chunk(
                    text=buffer.strip(),
                    metadata={"chunk_index": len(chunks)}
                ))
                buffer = ""
            # 超长段落按句子切
            sub_chunks = _split_long_paragraph(para, max_size)
            for sc in sub_chunks:
                chunks.append(Chunk(
                    text=sc.strip(),
                    metadata={"chunk_index": len(chunks)}
                ))
            continue

        # 正常情况：尝试合并到 buffer
        if len(buffer) + len(para) + 1 <= max_size:
            buffer = buffer + "\n" + para if buffer else para
        else:
            # buffer 满了，输出
            if buffer.strip():
                chunks.append(Chunk(
                    text=buffer.strip(),
                    metadata={"chunk_index": len(chunks)}
                ))
            buffer = para

    # 最后一个 buffer
    if buffer.strip():
        # 如果太短且前面有chunk，合并到上一个
        if len(buffer.strip()) < min_size and chunks:
            prev = chunks[-1]
            prev.text = prev.text + "\n" + buffer.strip()
        else:
            chunks.append(Chunk(
                text=buffer.strip(),
                metadata={"chunk_index": len(chunks)}
            ))

    return chunks


# ==========================================
# 策略三：标题感知切片（适合有章节结构的书）
# ==========================================
def chunk_by_heading(text: str, max_size: int = 512, min_size: int = 50) -> List[Chunk]:
    """
    按标题（章/节）边界切片，保留层级结构。
    
    逻辑：
        1. 用正则识别章节标题行（如 "第一章"、"第X节"、markdown # 标题）
        2. 按标题切成 section
        3. 每个 section 如果超长，内部再按段落切
        4. 标题写入 metadata，检索时可以展示来源章节
    
    适用：佛教十三经、道德经这类有明确章节的文档
    """
    # 标题模式：覆盖中文常见格式
    heading_pattern = re.compile(
        r'^('
        r'第[一二三四五六七八九十百千\d]+[章节篇卷品部]'  # 第X章/节/篇/卷/品
        r'|#{1,4}\s+.+'                                     # markdown标题
        r'|【.+?】'                                          # 【标题】
        r'|◎.+'                                              # ◎标题
        r'|○.+'                                              # ○标题
        r')',
        re.MULTILINE
    )

    # 把找到的标题转为列表，为什么要转列表？-> 迭代器只是存放第一个内容，后续所有内容必须要通过迭代器位移来获取
    # list()会一次性把所有结果算出来存进内存，之后才能随意索引、反复访问
    matches = list(heading_pattern.finditer(text))

    # 抽象一下：--内容1-- [start1]标题1[end1] --内容2-- [start2]标题2[end2] --
    if not matches:
        # 没找到标题结构，退回段落策略
        return chunk_by_paragraph(text, max_size, min_size)

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group().strip()
        body = text[start:end].strip()
        sections.append((heading, body))

    # 处理标题之前的内容（序言等）
    if matches[0].start() > min_size:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.insert(0, ("序", preamble))

    # 将 section 转成 chunks
    chunks = []
    for heading, body in sections:
        if len(body) <= max_size:
            chunks.append(Chunk(
                text=body,
                metadata={"chunk_index": len(chunks), "heading": heading}
            ))
        else:
            # section 超长，内部按段落切
            sub_chunks = chunk_by_paragraph(body, max_size, min_size)
            for sc in sub_chunks:
                sc.metadata["heading"] = heading
                sc.metadata["chunk_index"] = len(chunks)
                chunks.append(sc)

    return chunks


# ==========================================
# 辅助函数：策略一和策略二中需要用上的函数
# ==========================================
def _split_paragraphs(text: str) -> List[str]:
    """按双换行拆段落，过滤空段"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_paragraph(text: str, max_size: int) -> List[str]:
    """
    超长段落按句子边界切分。
    中文句子结束符：。！？；（全角）
    """
    sentences = re.split(r'(?<=[。！？；\n])', text)
    
    result = []
    buffer = ""
    
    for sent in sentences:
        if not sent.strip():
            continue
        if len(buffer) + len(sent) <= max_size:
            buffer += sent
        else:
            if buffer.strip():
                result.append(buffer.strip())
            buffer = sent
    
    if buffer.strip():
        result.append(buffer.strip())
    
    return result


def _find_sentence_boundary(text: str, pos: int, lookback: int = 100) -> int:
    """
    从 pos 往前找最近的句子结束位置。
    优先找：。！？；\n
    找不到就返回 pos 本身（硬切）。
    """
    search_start = max(0, pos - lookback)
    region = text[search_start:pos]

    # 从后往前找句子边界
    for marker in ['。', '！', '？', '；', '\n']:
        idx = region.rfind(marker)
        if idx != -1:
            return search_start + idx + 1  # +1 包含标点本身

    return pos


# ==========================================
# 元数据注入：给每个chunk加上来源信息
# ==========================================
def add_file_metadata(chunks: List[Chunk], filename: str, total_chunks: int = None) -> List[Chunk]:
    """
    给切片批量注入文件级元数据。
    
    面试考点：元数据在RAG中的作用
        1. 检索时可按来源过滤
        2. 答案溯源：告诉用户"出自《道德经》第三章"
        3. 位置信息：chunk在文档中的相对位置（前/中/后）
    """
    total = total_chunks or len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source_file": filename,
            "chunk_index": i,
            "total_chunks": total,
            "position": _get_position_label(i, total),
        })
    return chunks


def _get_position_label(index: int, total: int) -> str:
    """文档位置标签：前/中/后"""
    if total <= 3:
        return "全文"
    ratio = index / total
    if ratio < 0.2:
        return "开头"
    elif ratio > 0.8:
        return "结尾"
    return "中间"


# ==========================================
# 转换为 LlamaIndex 节点
# 框架流程：SimpleDirectoryReader → 自动切片 → 生成TextNode → 建索引
# ==========================================
def chunks_to_nodes(chunks: List[Chunk]) -> List[TextNode]:
    """
    将自定义 Chunk 转为 LlamaIndex 的 TextNode。
    
    为什么不直接用 LlamaIndex 的 splitter：
        1. 自定义切片逻辑更灵活（段落感知、标题感知）
        2. 元数据完全可控
        3. 方便对比不同策略，不被框架绑定
        4. 面试能说清楚原理，而不是"我调了个API"
    """
    nodes = []
    for chunk in chunks:
        node = TextNode(
            text=chunk.text,
            metadata=chunk.metadata,
            # excluded_llm_metadata_keys 防止元数据被塞进prompt浪费token
            excluded_llm_metadata_keys=["chunk_index", "total_chunks", "start_char", "end_char"],
            # excluded_embed_metadata_keys 防止元数据干扰embedding
            excluded_embed_metadata_keys=["chunk_index", "total_chunks", "start_char", "end_char", "position"],
        )
        nodes.append(node)
    return nodes


# ==========================================
# 策略对比工具
# ==========================================
def compare_strategies(text: str, filename: str = "test") -> dict:
    """
    对同一段文本跑所有策略，输出统计信息用于对比。
    
    用法：
        stats = compare_strategies(open("data_cleaned/道德经.txt").read(), "道德经")
        # 然后对比哪种策略的 chunk 数量、平均长度、最短/最长更合理
    """
    strategies = {
        "fixed_512": lambda t: chunk_fixed(t, chunk_size=512, overlap=100),
        "fixed_256": lambda t: chunk_fixed(t, chunk_size=256, overlap=50),
        "paragraph_512": lambda t: chunk_by_paragraph(t, max_size=512),
        "paragraph_256": lambda t: chunk_by_paragraph(t, max_size=256),
        "heading_512": lambda t: chunk_by_heading(t, max_size=512),
    }

    results = {}
    for name, strategy in strategies.items():
        chunks = strategy(text)
        chunks = add_file_metadata(chunks, filename)
        lengths = [len(c) for c in chunks]

        results[name] = {
            "chunk_count": len(chunks),
            "avg_length": sum(lengths) // len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "chunks": chunks,  # 保留原始chunks，方便后续检查
        }

    return results


def print_comparison(results: dict):
    """打印对比表格"""
    logger.info(f"\n{'策略':<20} {'切片数':>6} {'平均长度':>8} {'最短':>6} {'最长':>6}")
    logger.info("-" * 52)
    for name, stats in results.items():
        logger.info(f"{name:<20} {stats['chunk_count']:>6} {stats['avg_length']:>8} {stats['min_length']:>6} {stats['max_length']:>6}")


# ==========================================
# 预览工具：单独运行时查看切片效果
# ==========================================
def preview_chunks(chunks: List[Chunk], n: int = 5):
    """预览前 n 个切片"""
    logger.info(f"\n共 {len(chunks)} 个切片，预览前 {min(n, len(chunks))} 个：\n")
    for i, chunk in enumerate(chunks[:n]):
        logger.info(f"--- 切片 {i+1} ({len(chunk)} 字符) ---")
        logger.info(f"元数据: {chunk.metadata}")
        logger.info(chunk.text[:200])


# ==========================================
# 主入口：单独运行可预览效果
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "data_cleaned"

    if not os.path.exists(DATA_DIR):
        logger.info(f"错误：找不到 {DATA_DIR} 目录，请先运行 clean_data.py")
        exit(1)

    # 遍历所有清洗后的文件
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(f"\n{'='*60}")
        logger.info(f"文件: {filename} ({len(text):,} 字符)")
        logger.info(f"{'='*60}")

        # 对比所有策略
        results = compare_strategies(text, filename)
        print(results)

        # 预览推荐策略的切片
        logger.info(f"\n>>> paragraph_512 策略预览:")
        preview_chunks(results["paragraph_512"]["chunks"], n=3)