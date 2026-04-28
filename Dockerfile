# 基础镜像：Python 3.12
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件，利用Docker缓存（依赖不变就不重装）
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 10 \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 10 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录（如果不存在）
RUN mkdir -p data_cleaned storage

# 默认暴露端口
EXPOSE 8000 8501
