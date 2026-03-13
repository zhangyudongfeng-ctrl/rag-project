# 场景练习：一批用户评论数据，要依次经过以下处理步骤，最终输出清洗后的文本列表
import re 
raw_data = [
    "  这个产品太好了！！！推荐给大家!!!  ",
    "  垃圾，退款退款退款   ",
    "  还行吧...价格有点贵，质量一般般   ",
    "  AMAZING product! 五星好评☆☆☆☆☆  ",
]

# 1、去除左右括号
def clean_space(raw_data):
    return [s.strip() for s in raw_data]

# 2、将连续重复的标点压缩到N个
def compressSymbol_N(n):
    def transform(raw_data):
        return [re.sub(r'([^\w\s])\1+', lambda m: m.group(1) * min(n, len(m.group())), text) for text in raw_data]
    return transform
        

# 3、过滤长度小于M的文本
def filter_Mtext(m):
    def transform(raw_data):
        return [s for s in raw_data if len(s) >= m]
    return transform

# 4、给每条文本加上统一前缀标签，比如[已清洗]（标签内容可配置）
def add_prefix(prefix):
    def transform(raw_data):
        return [prefix + s for s in raw_data]
    return transform

PIPELINE = [
    clean_space,
    compressSymbol_N(1),
    filter_Mtext(5),
    add_prefix("(已清洗)")
]

def run_pipeline(raw_data):
    # 调用上一步处理过的数据
    # 上一步处理过的数据是：step(data)再赋值给data
    data = raw_data
    for step in PIPELINE:
        data = step(data)
    return data

res = run_pipeline(raw_data=raw_data)
print(f"cleaned_data={res}")