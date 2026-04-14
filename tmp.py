'''
 * @Author       : MatthewZhang
 * @Date         : 2026-03-16 16:14:45
 * @Description  : 
'''

from service import RagService
from config import load_config, configure_settings  # ← 补上

config = load_config()
configure_settings(config)  # ← 先配置全局 Settings
service = RagService(config)
index = service.components.index

for node in index.docstore.docs.values():
    if "庄子" not in node.metadata.get("source_file", ""):
        continue
    ct = node.metadata.get("content_type", "未标")
    idx = node.metadata.get("chunk_index", -1)
    preview = node.text[:40].replace("\n", " ")
    print(f"[{idx:3}] [{ct}] {preview}")