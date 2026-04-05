<!--
 * @Author       : MatthewZhang
 * @Date         : 2026-04-01 08:58:45
 * @Description  :
-->

问题进来 → api.py → router.py → handler → 返回

api.py 启动时构建：
query_engine (multi_query retriever + reranker + LLM)
simple_retriever (hybrid only, 给multi_doc用)
reranker (共享)

route_query(question, intent, index, query_engine, simple_retriever, reranker)
├→ normal → handle_normal(question, query_engine)
├→ position → handle_position(question, index)
└→ multi_doc → handle_multi_doc(question, simple_retriever, reranker)

统一返回: {"answer": str, "sources": [...]}

# 项目架构

## 1. 项目概览

这是一个中文 RAG 问答系统，当前主要支持三类问题：

- 普通问答 `normal`
- 位置问答 `position`
  例如“这本书的开头讲了什么”“结尾写了什么”
- 多文档问答 `multi_doc`
  例如“比较 A 和 B 的观点”“综合多本书回答”

系统当前提供三种使用方式：

- 命令行交互 `main.py`
- FastAPI 后端接口 `api.py`
- Streamlit 前端页面 `app.py`

同时还提供离线评测能力：

- `run_eval.py`
- `evaluator.py`
- `golden_dataset.json`

整体流程如下：

1. 加载配置
2. 初始化全局模型设置
3. 加载或构建索引
4. 构建检索器和问答引擎
5. 对用户问题做意图分类
6. 路由到对应处理逻辑
7. 返回答案和来源片段

---

## 2. 模块划分

### `config.py`

统一配置模块。

主要职责：

- 从环境变量读取 `DEEPSEEK_API_KEY`
- 定义模型名、路径、检索参数
- 初始化 `llama_index.Settings`

核心内容：

- `RagConfig`
- `load_config()`
- `configure_settings()`

---

### `index_factory.py`

索引构建与加载模块。

主要职责：

- 检查 `storage/` 是否可用
- 从持久化目录加载索引
- 当索引不存在时，从 `data_cleaned/` 构建新索引

---

### `service.py`

服务层模块。

主要职责：

- 持有系统运行所需的核心组件
- 对外提供统一的查询和上传接口
- 屏蔽 CLI / API 层对底层装配细节的感知

核心类：

- `RagService`

核心方法：

- `query(question)`
- `upload_text(filename, content)`
- `reload()`

---

### `engine.py`

检索与问答引擎装配模块。

主要职责：

- 构建不同模式下的 retriever
- 构建 reranker
- 构建 `RetrieverQueryEngine`

核心内容：

- `MultiQueryHybridRetriever`
- `HybridOnlyRetriever`
- `build_components()`

支持的模式包括：

- `vector_only`
- `hybrid`
- `hybrid_jieba`
- `multi_query`
- `multi_query_jieba`

---

### `retriever.py`

底层检索逻辑模块。

主要职责：

- 创建向量检索器
- 创建 BM25 检索器
- 通过 RRF 融合检索结果
- 支持多查询混合检索

核心函数：

- `create_retrievers()`
- `rrf_fusion()`
- `hybrid_retrieve()`
- `multi_query_hybrid_retrieve()`

---

### `query_rewriter.py`

查询改写模块。

主要职责：

- 通过 LLM 将一个问题扩展为多个更适合检索的子查询
- 提高召回覆盖率

核心函数：

- `multi_query_rewrite()`

---

### `intent_classifier.py`

意图分类模块。

主要职责：

- 将用户问题分类为以下三种之一：
  - `normal`
  - `position`
  - `multi_doc`

核心函数：

- `classify_intent()`

---

### `router.py`

问题路由与处理模块。

主要职责：

- 统一格式化来源片段
- 处理位置问题的归一化与参数抽取
- 执行不同类型问题的处理逻辑
- 根据意图分发到对应 handler

核心函数：

- `format_nodes_to_sources()`
- `normalize_position()`
- `extract_position_and_source()`
- `handle_normal()`
- `handle_position()`
- `handle_multi_doc()`
- `route_query()`

---

## 3. 入口文件

### `main.py`

命令行入口。

执行流程：

1. 加载配置
2. 初始化全局 Settings
3. 创建 `RagService`
4. 循环接收用户输入
5. 输出答案和来源

适合：

- 本地快速调试
- 手工测试问答效果

---

### `api.py`

FastAPI 后端入口。

提供接口：

- `POST /query`
- `POST /upload`
- `GET /health`

执行流程：

1. 启动时初始化 `RagService`
2. 接收 HTTP 请求
3. 调用服务层处理
4. 返回结构化 JSON 结果

适合：

- 给前端页面调用
- 后续对接其他客户端

---

### `app.py`

Streamlit 前端入口。

主要职责：

- 提供可视化问答界面
- 调用后端 `/query`
- 调用后端 `/upload`
- 展示答案和来源片段

适合：

- 演示
- 人工体验验证

---

### `run_eval.py`

离线评测入口。

主要职责：

- 加载持久化索引
- 构建检索和问答组件
- 加载 golden dataset
- 执行评测并保存结果

适合：

- 比较不同版本效果
- 回归测试

---

## 4. 查询流程

### 普通问答流程 `normal`

1. 用户输入问题
2. 意图分类为 `normal`
3. 调用 `query_engine.query(question)`
4. retriever 返回带分数的检索结果
5. reranker 对结果进一步排序
6. LLM 基于上下文生成答案
7. 返回答案和来源片段

特点：

- 来源通常带 `score`
- 属于标准 RAG 路径

---

### 位置问答流程 `position`

1. 用户输入问题
2. 意图分类为 `position`
3. 提取目标位置信息，例如“开头”或“结尾”
4. 如有需要，提取来源文件提示
5. 从 `index.docstore.docs` 中按 metadata 过滤节点
6. 选取匹配片段作为上下文
7. LLM 基于这些片段生成答案
8. 返回答案和来源片段

特点：

- 不走 RRF
- 不走标准 retriever 打分链路
- 返回的通常是 `TextNode`
- 来源一般没有真实检索分数

---

### 多文档问答流程 `multi_doc`

1. 用户输入问题
2. 意图分类为 `multi_doc`
3. 使用 LLM 将问题拆成多个子问题
4. 对每个子问题执行混合检索
5. 聚合多个子问题的结果
6. 拼接上下文
7. LLM 生成最终答案
8. 返回答案和来源片段

特点：

- 更适合比较、总结、综合类问题
- 来源通常带检索分数

---

## 5. 数据与存储结构

### `data/`

原始文件目录。

通常存放：

- epub
- txt
- 其他原始输入材料

---

### `data_cleaned/`

清洗后的文本目录。

作用：

- 作为索引构建的直接输入
- 是当前系统最主要的知识源目录

---

### `storage/`

LlamaIndex 持久化索引目录。

通常包含：

- `docstore.json`
- `index_store.json`
- vector store 文件

作用：

- 避免每次启动都重新构建索引
- 支持增量更新后的持久化

---

### `golden_dataset.json`

评测数据集。

作用：

- 存放测试问题、期望答案、关键词、来源等信息
- 用于离线评估检索和生成效果

---

### `eval_*.csv`

评测输出结果。

作用：

- 保存每轮评测明细
- 便于版本比较和分析问题样本

---

## 6. 当前关键设计说明

### 6.1 全局 Settings 设计

当前项目通过 `config.py` 统一初始化 `llama_index.Settings`。

优点：

- 启动方式简单
- CLI、API、评测可共用一套配置

缺点：

- 依赖全局状态
- 单元测试和模块隔离测试不够方便

---

### 6.2 位置问题不参与标准检索打分

位置类问题当前不是通过 retriever 返回 `NodeWithScore`，而是直接通过 metadata 过滤 `TextNode`。

这意味着：

- 没有真实检索分数
- 不应该把 `0.0` 误当成低分
- 更合理的语义是“无分数”

---

### 6.3 来源片段存在两种类型

系统当前返回的来源片段，实际上有两种来源：

1. 检索结果  
   通常来自 `NodeWithScore`  
   有真实 `score`

2. 位置过滤结果  
   通常来自 `TextNode`  
   没有真实 `score`

因此展示层应区分：

- 有分数的来源
- 无分数的来源

而不是把“无分数”伪装成 `0.0`

---

## 7. 当前已知问题

- `router.py` 仍然承担了较多职责
- `position` 路径直接扫描 `docstore`，后续扩展性一般
- 一些 prompt、注释和文档存在编码乱码问题
- 多文档路径中的 reranker 语义需要和实际实现保持一致
- 一些历史版本文件和实验脚本仍未清理
- 项目中仍存在实验阶段遗留结构

---

## 8. 下一步建议的演进方向

1. 继续收敛 `router.py` 的职责，但不必过度拆分
2. 增强 `position` 路径的健壮性和空结果处理
3. 统一来源结构，对有分数和无分数节点做清晰建模
4. 将常用 prompt 逐步集中管理
5. 视数据规模决定是否为位置查询建立专门索引
6. 清理旧版本文件、缓存文件和乱码文档
