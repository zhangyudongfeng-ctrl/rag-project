# 第一阶段：目录遍历与文件读取（“大扫除前的工作”）
# 第二阶段：清洗逻辑的“内化”（“真正的清洁工作”）
# 第三阶段：文件格式的分流处理（“因材施教”）
# 第四阶段：工程化封装（“把活儿干漂亮”）

import re
import os 
from ebooklib import epub
import ebooklib
from html2text import HTML2Text


INPUT_DIR = "data"
OUTPUT_DIR = "data_cleaned"

# 从epub中提取纯文本
# 逻辑思路：
# [路径] 
#   ↓ (read_epub)
# [书本对象] 
#   ↓ (get_items + if filter)
# [HTML 零件] 
#   ↓ (decode + html2text)
# [干净文字段落] 
#   ↓ (join)
# [一整本大 TXT] 
#   ↓
# [交给你的清洗流水线]
def extract_text_from_epub(filepath):
    """专门从 EPUB 中提取纯文本的工具"""
    try:
        book = epub.read_epub(filepath, options={'ignore_ncx': True})
        h = HTML2Text() # 不论原始数据是 TXT 还是 EPUB，统一转为带有 Markdown 标记（如 #, **, []()）的纯文本字符串 --- 为什么先转为这个？因为直接剥离 HTML 标签会丢失段落、换行等结构。HTML2Text 能把 <h1> 变成 #，把 <p> 变成换行，从而保留文档的逻辑结构。
        h.ignore_links = False  # 先保留，交给流水线处理
        h.ignore_images = True
        h.ignore_emphasis = False   # 保留加粗/斜体的文字内容
        h.body_width = 0 # 必要。默认值是78，意思是超过78个字符就自动插换行符。这会把一个完整句子从中间断开，后面按换行切片时一个语义完整的句子会被切成两个chunk。设0就是不自动换行，保留原始段落结构。
        h.unicode_snob = True       # 用unicode替代ascii,这个不开的话可能丢失或替换中文标点，影响后续embedding的语义准确性。
        h.skip_internal_links = True # 目录跳转，在ignore_links=True的前提下不会触发

        content_parts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # 提取 HTML 并转为 Markdown 格式的文字
                html_content = item.get_content().decode('utf-8', errors='ignore')
                # 跳过目录/导航页（通常内容很短或含大量链接）
                if len(html_content) < 200:
                    continue
                text = h.handle(html_content)
                # 跳过清洗后内容太少的内容（封面、版权页等，比如“本文由XXX赞助”）
                if len(text.strip()) < 20:
                    continue
                content_parts.append(text)
        return "\n\n".join(content_parts)
    except Exception as e:
        print(f"❌ 解析 EPUB 出错: {filepath} | 错误: {e}")
        return ""

# ==========================================
# 第一层：工作函数 (真正的干活工人)
# 函数作用：把文本text按pattern正则处理返回
# ==========================================
def re_cleaner_executor(text, pattern, replacement=""):
    """真正执行正则替换的函数"""
    if not text:
        return ""
    # flags=re.MULTILINE 允许 ^ 和 $ 匹配每一行的开头和结尾
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)

# ==========================================
# 第二层：封装函数 (工厂/封装器)
# ==========================================
def create_cleaner(pattern, replacement="", name=""):
    """
    闭包工厂：传入正则和替换逻辑，返回标准化的 C 函数。
    """
    def cleaner_interface(text):
        return re_cleaner_executor(text, pattern, replacement)
    
    # 在 Python 中，每个函数都有一个内置属性叫 __name__。如果你手动设置它，那么流水线里所有的函数都会叫同一个名字：cleaner_interface
    cleaner_interface.__name__ = name or f"cleaner_{pattern[:10]}"
    return cleaner_interface

# ==========================================
# 第三层：定义“滤芯” 
# ==========================================

# 1. 图片清洗：完全删掉
clean_images = create_cleaner(r'!\[.*?\]\(.*?\)', "", "1_images")

# 2. 脚注与引用：删掉类似 [[8]](link) 或 [8]
clean_footnotes = create_cleaner(r'\[+\d+\]+\]\(.*?\)|\[(\d+)\]\(.*?\)|(?<!\[)\[(\d+)\](?!\()', "", "2_footnotes")

# 3. 链接处理：保留[文字]，删掉(url)  <-- 注意这里的 \1，代表保留第一个括号里的内容
preserve_link_text = create_cleaner(r'\[([^\]]+)\]\([^\)]+\)', r'\1', "3_link_text")

# 4. HTML 标签与实体：统统杀掉
clean_html = create_cleaner(r'<[^>]+>|&[a-zA-Z]+;|&#\d+;', "", "4_html")

# 5. EPUB 噪音：删掉文件名残留（如 part001.xhtml）
clean_epub_noise = create_cleaner(r'part\d+\.xhtml[#\w]*|[\w]+\.xhtml', "", "5_epub")

# 6. Markdown 装饰符：删掉空标题行、分割线 *** ---
clean_markdown_symbols = create_cleaner(r'^#{1,6}\s*$|\*{3,}|_{3,}|-{3,}', "", "6_md_symbols")

# 7. 空格压缩：多个空格/制表符变一个空格
compress_spaces = create_cleaner(r'[ \t]+', ' ', "7_spaces")

# 8. 换行压缩：3个以上换行变2个，清理纯空白行
# 重点：在 RAG 中，经常会用“换行符”来作为切片的依据。
compress_newlines = create_cleaner(r'\n{3,}', '\n\n', "8_newlines")
clean_blank_lines = create_cleaner(r'^\s+$', '', "9_blank_lines")

# ==========================================
# 第四层：组装流水线 (按顺序净化)
# 定义流水线清单，按顺序执行
# python的函数名代表函数对象的内存地址，函数也可以
# ==========================================
PIPELINE = [
    clean_images,
    clean_footnotes,
    preserve_link_text,      # 先保留文字
    clean_html,
    clean_epub_noise,
    clean_markdown_symbols,
    compress_spaces,         # 最后处理格式
    compress_newlines,
    clean_blank_lines
]


# 核心运行函数，数据流过每一个管道
def run_pipeline(text):
    for cleaner in PIPELINE:
        text = cleaner(text)
    return text

# 读取和写入文件
# 思路：拼接完整路径(获取文件夹下所有文件名，用join拼接)->拼接的同时读取文件(跳过目录，防止报错)
def read_and_clean_files(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"创建目录: {output_path}")

    # 获取文件夹下的所有文件名
    file_list = os.listdir(input_path)

    # 循环遍历每一个文件
    for filename in file_list:
        # 拼接成完整路径
        file_path = os.path.join(input_path, filename)

        if not os.path.isfile(file_path): 
            continue

        # 1. 获取文件后缀名（转小写方便判断） --- 转成小写 lower()，就是为了代码在 Windows、Linux、macOS 上都能表现一致，不会因为文件名的大小写差异而漏掉重要文档
        # splitext函数作用：将文件名从“最后一个点”的地方切成两半，例子：test_data.epub -> ('test_data', '.epub')索引分别是0和1
        suffix = os.path.splitext(filename)[1].lower()
        
        # 2. 格式分流处理
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
        elif suffix == '.epub':
            print(f"📦 正在解析 EPUB: {filename}...")
            raw_content = extract_text_from_epub(file_path)
        else:
            print(f"⚠️ 跳过不支持的文件: {filename}")
            continue

        # 3. 执行清洗流水线
        clean_content = run_pipeline(raw_content)

        # 4. 【统一接口】将文件名后缀改为 .txt
        new_filename = os.path.splitext(filename)[0] + ".txt"
        file_out = os.path.join(output_path, new_filename)
        
        # 5. 写入新文件
        with open(file_out, 'w', encoding='utf-8') as f_out:
            f_out.write(clean_content)
        
        print(f"✅ 处理完成: {new_filename:25} | 原始长度: {len(raw_content):7} | 清洗后: {len(clean_content):7}")
        # 测试代码
        if "。" not in clean_content:
            print("⚠️ 警告：检测到文档中没有中文句号！切片逻辑可能会崩！")
        

# ==========================================
# 程序的正式入口
# ==========================================
if __name__ == "__main__":
    print("🚀 开始数据清洗流水线...")
    read_and_clean_files(INPUT_DIR, OUTPUT_DIR)
    print("\n✨ 全部清洗任务已完成！请检查 data_cleaned 文件夹。")



