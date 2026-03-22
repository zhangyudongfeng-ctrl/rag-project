

# 1、按双换行拆成段落列表
# 2、遍历段落，短的往buffer里攒，攒到接近max_size就输出一个chunk
# 3、遇到超长段落，先把buffer输出，再把长段落按句号切，攒buffer，满了就输出
# 4、返回所有chunks
from dataclasses import dataclass, field
from typing import List, Optional
import re

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

# 输入:text
# 输出:各个chunk
# 先判断特殊情况，再走正常流程
def chunk_by_paragraph(text: str, max_size: int = 512, min_size: int = 50) -> List[Chunk]:
    chunks = []
    # 按双换行拆成列表
    paragraphs = text.split('\n\n')

    # buffer在外面，每次循环可能会有残留
    buffer = ""
    for p in paragraphs:
        # 先处理超长段落
        if len(p) > max_size:
            if buffer.strip():
                chunks.append(Chunk(text=buffer.strip(), metadata={"chunk_index": len(chunks)}))
                buffer = ""
            sentences = p.split('。')   # 长段落按句号切割
            for s in sentences:
                if len(buffer) + len(s) <= max_size:
                    buffer += s + '。'
                else:
                    chunks.append(Chunk(text=buffer.strip(), metadata={"chunk_index": len(chunks)}))
                    buffer = s + '。'
        # buffer加上当前段落后会不会超过max_size
        elif len(buffer.strip()) + len(p) <= max_size:
                buffer = buffer + "\n" + p if buffer else p
        # 进入else分支说明buffer大于max_size了，此时先把buffer里的内容输出到一个chunk中,再把当前的p放入buffer
        else:
            chunks.append(Chunk(text=buffer.strip(), metadata={"chunk_index": len(chunks)}))
            buffer = p

    # 如果循环结束后buffer里还有残留，把它给加到chunker里
    if buffer.strip():
        if len(buffer.strip()) < min_size and chunks:
            chunks[-1].text += "\n" + buffer.strip()
        else:
            chunks.append(Chunk(text=buffer.strip(), metadata={"chunk_index": len(chunks)}))
    return chunks
 
# 输入：text
# 输出：各个chunk
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
    )   # re.MULTILINE — 改变^和$的行为。默认^只匹配整个字符串的开头，加了这个flag后，^匹配每一行的开头（即每个\n之后的位置）

    # 把找到的标题转为列表，为什么要转列表？-> 迭代器只是存放第一个内容，后续所有内容必须要通过迭代器位移来获取
    # list()会一次性把所有结果算出来存进内存，之后才能随意索引、反复访问
    matches = list(heading_pattern.finditer(text))

    sections = []
    chunks = []
    if not matches:
        return chunk_by_paragraph(text, max_size, min_size)
    # 抽象一下：--内容1-- [start1]标题1[end1] --内容2-- [start2]标题2[end2] --
    # 第一个标题前如果有内容，存入sections中
    first_end = matches[0].start()
    if first_end > 0:
        sections.append(('序', text[:first_end]))
    
    # 按标题切为section，写入chunk -> 需要怎么做？找到标题内容的开头位置，作为分割点，下一个分割点是下个标题的开头，写入chunk即可
    # 把标题当做heading存入元数据
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group().strip()
        body = text[start:end].strip()
        # 怎么找到start到end的内容？-> match.group()返回的是text[match.start() : match.end()]的内容
        sections.append((heading, body))

    # 统一处理sections，把内容存入chunks中
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

if __name__ == "__main__":
    # 造一段假数据测
    test_text = """第一章 道可道非常道。名可名非常名。

第二章 天下皆知美之为美，斯恶已。

这是一个超长段落。""" + "测试内容。" * 200 + """

短。

第三章 上善若水。"""

    chunks = chunk_by_paragraph(test_text, max_size=100, min_size=20)
    for i, c in enumerate(chunks):
        print(f"chunk{i}: ({len(c.text)}字符) {c.text[:80]}")


def  hybrid_retrieve(query, vector_retriever, bm25_retriever, k=60):
    """
    混合检索完整流程：
        1. 向量检索器和BM25检索器分别检索，各返回top_k个结果
        2. 用RRF（Reciprocal Rank Fusion）融合两路结果
        3. 返回融合排序后的结果列表

    参数：
        query: 用户问题
        vector_retriever: 向量检索器
        bm25_retriever: BM25检索器
        k: RRF平滑系数，默认60，防止排名靠前的结果分数差距过大
    """
    scores = {}
    node_map = {}

    for rk, node in scores:
        # 思路：遍历两种检索结果，拿到每个node的排名位置，套用RRF公式计算分数，最后同一node的分数叠加
        node_id = node.node_id
        scores[node]