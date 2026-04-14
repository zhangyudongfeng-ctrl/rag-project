'''
 * @Author       : MatthewZhang
 * @Date         : 2026-04-07 19:45:38
 * @Description  : 
'''
# PDF文件
# → 提取文本
# → 变成 str
# → 进入现有 clean/chunk/metadata/index 管线

# !注:PDF格式主要有两种, 文本型PDF和扫描型PDF(目前只处理文本型PDF)
# 文本型PDF:文字以字符形式存储在文件里，每个字都有 Unicode 编码。比如Word/LaTeX/网页导出的 PDF 
# 扫描型PDF:文字以图片形式存储在文件里，无法直接提取文本,必须用 OCR（光学字符识别）先把图片转成文字。比如扫描的纸质书籍生成的 PDF

import os
import fitz
import logging
import re 

logger = logging.getLogger(__name__)



# 把读取的pdf文件中的多余字符给干掉,也就是数据清理,输出为txt格式导入data_clean文件,输入:pdf文件,输出:txt文件,并导入到data_cleaned文件夹中, 把两种 PDF 都归一化成"段落之间 \n\n"的 txt 格式，这样下游 chunk_by_paragraph 完全不用改。清晰且正确。
# 1.markdown的pdf文件简单,边界信号本来就在——空行（\n \n）就是段落分隔，只是多了个空格。删空格就行
# 2.论文pdf文件复杂,边界信号缺失，段落全被单 \n 分开。要人为识别边界再插入 \n\n。标题是最容易识别的边界信号

# TODO: arXiv清洗
# - 截断 References 之后的内容（省token）
# - 段落内启发式切分（如果整节太长）

'''
 * @description: 处理 Markdown 风格和 arxiv 风格的 PDF，返回清洗后的文本
 * @param {*} file_path文件名
 * @return {*} 处理完毕后的新内容
'''
def parse_pdf(file_path: str) -> str:
    """提取 PDF 文本并做轻量清洗，下游交给 clean_data.py"""
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # 1. 去行尾空格（Markdown PDF 特有）
    text = re.sub(r" +\n", "\n", text)

    # 2. 识别 arXiv 风格标题并加段落分隔
    text = re.sub(
        r"\n(\d+(?:\.\d+)*)\n([A-Z][^\n]+)\n",
        r"\n\n\1 \2\n\n",
        text
    )

    return text


'''
 * @description: 针对filename判断是否是扫描型PDF -> 如何判断是不是扫描型PDF? -> 如果第一页图片数量 >= 1，且文本为空 -> 报错, 无返回值
 * @param {str} filename
 * @return {bool} 图片数量 >= 1 and 文本为空 
'''
def is_scanned_pdf(filename: str) -> bool:
    doc = fitz.open(filename)
    page = doc[0]
    text_empty = not page.get_text().strip()
    has_images = len(page.get_images()) >= 1
    doc.close()
    return text_empty and has_images


'''
 * @description: 单个 PDF → 文本
 * @param {str} filepath
 * @return {str} text字符串
'''
def extract_text_from_pdf(filepath: str) -> str:
    doc = fitz.open(filepath)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

'''
 * @description: 目录 → {文件名: 文本} 只处理 .pdf 文件, 跳过其他文件
 * @param {str} dir_path
 * @return {dict} {filename: text}
'''
def extract_texts_from_dir(dir_path: str) -> dict:
    results = {}
    for filename in os.listdir(dir_path):
        # 跳过非pdf文件
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(dir_path, filename)
        # 跳过扫描型PDF
        if is_scanned_pdf(filepath):
            logger.warning(f"跳过扫描型PDF: {filename}")
            continue
        results[filename] = parse_pdf(filepath)

    return results

if __name__ == "__main__":
    texts = extract_texts_from_dir(dir_path="test_data")
    for name, text in texts.items():
        print(f"=== {name} ===")
        print(repr(text))  # 只打印前200字符看看
    # doc = fitz.open("test_data/西方马克思主义概论 (衣俊卿) .pdf")
    # page = doc[0]
    # print("文本:", repr(page.get_text()))
    # print("图片数量:", len(page.get_images()))  # 扫描件每页都有图