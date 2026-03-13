import zipfile

# 替换成你的 epub 路径
epub_path = r"D:\rag-project\data\1.epub" 


try:
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        # 打印出压缩包里所有的文件名
        print("这个 EPUB 里面包含的文件有：")
        for file in zip_ref.namelist()[:50]: # 只看前15个，防止刷屏
            print(f"  - {file}")

        # 1. 找出一个以 .xhtml 或 .html 结尾的文件（通常是正文）
        content_files = [f for f in zip_ref.namelist() if f.endswith(('xhtml', 'html'))]
        
        if content_files:
            target_file = content_files[2]  # 选第三个文件，通常避开了封面和版权页
            with zip_ref.open(target_file) as f:
                raw_content = f.read().decode('utf-8', errors='ignore')
                
                print(f"--- 正在偷看文件: {target_file} ---")
                print("下面是前 500 个字符的原始 HTML（你会看到很多标签）：")
                print("-" * 50)
                print(raw_content[:500]) 
                print("-" * 50)
except Exception as e:
    print(f"连 Python 也打不开它，错误原因: {e}")