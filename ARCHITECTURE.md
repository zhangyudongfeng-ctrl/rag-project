# RAG Project Architecture

## 1. 项目定位

这是一个中文 RAG 问答系统，围绕本地文档构建索引，并根据用户问题的意图选择不同的问答路径。

当前支持三类问题：

- `normal`：普通 RAG 问答。
- `position`：位置类问答，例如“这本书开头讲了什么”“结尾写了什么”。
- `multi_doc`：多文档综合问答，例如“比较 A 和 B 的观点”。

当前提供四类使用入口：

- `main.py`：命令行交互入口。
- `api.py`：FastAPI 后端入口。
- `app.py`：Streamlit 前端页面。
- `run_eval.py` / `evaluator.py`：离线评测入口。

## 2. 总体流程

### 索引构建流程

```text
原始文档
  -> 清洗与切分
  -> TextNode
  -> embedding
  -> LlamaIndex 索引
  -> storage/ 持久化
```

### 查询流程

```text
用户问题
  -> RagService
  -> intent_classifier.classify_intent()
  -> router
  -> handler
  -> answer / sources
```

系统中同时存在非流式查询和流式查询两套入口：

- 非流式入口返回完整 JSON：`answer + sources`。
- 流式入口面向前端展示，统一返回文本流；其中 `normal` 是真正 token 级流式，`position` 和 `multi_doc` 当前是生成完整答案后一次性 yield 文本。

## 3. 核心模块

### `config.py`

统一配置模块。

主要职责：

- 读取 `DEEPSEEK_API_KEY`。
- 定义 LLM、embedding、reranker、路径和检索参数。
- 初始化 `llama_index.Settings`。

核心对象：

- `RagConfig`
- `load_config()`
- `configure_settings()`

### `index_factory.py`

索引加载与构建模块。

主要职责：

- 检查 `storage/` 是否已有可用索引。
- 从持久化目录加载索引。
- 在索引不存在时，从 `data_cleaned/` 构建新索引。

### `engine.py`

检索器、reranker、query engine 装配模块。

主要职责：

- 构建 retriever。
- 构建 reranker。
- 构建普通 query engine。
- 构建 streaming query engine。

核心内容：

- `MultiQueryHybridRetriever`
- `HybridOnlyRetriever`
- `Auxiliary_downweight_postprocessor`
- `NoOpReranker`
- `CappedReranker`
- `build_retriever()`
- `build_reranker()`
- `build_synthesizer(streaming: bool)`
- `build_components()`

当前 `build_components()` 返回：

```text
query_engine
streaming_query_engine
retriever
reranker
```

其中：

- `query_engine` 使用 `streaming=False` 的 response synthesizer。
- `streaming_query_engine` 使用 `streaming=True` 的 response synthesizer。

### `rag_compotents.py`

运行期组件容器。

主要职责：

- 用 dataclass 聚合服务运行所需的组件。
- 避免在 handler 之间传递过长参数列表。

当前包含：

- `index`
- `query_engine`
- `streaming_query_engine`
- `simple_retriever`
- `reranker`
- `local_llm`

### `retriever.py`

底层检索逻辑模块。

主要职责：

- 创建向量检索器。
- 创建 BM25 检索器。
- 通过 RRF 融合检索结果。
- 支持单 query 和 multi-query 混合检索。

核心函数：

- `create_retrievers()`
- `rrf_fusion()`
- `hybrid_retrieve()`
- `multi_query_hybrid_retrieve()`

### `query_rewriter.py`

Query 改写模块。

主要职责：

- 使用 LLM 将原始问题扩展为多个更适合检索的查询。
- 服务于 `multi_query` 和 `multi_query_jieba` 检索模式。

核心函数：

- `multi_query_rewrite()`

### `intent_classifier.py`

意图分类模块。

主要职责：

- 将问题分类为 `normal`、`position`、`multi_doc`。
- 分类失败时降级到 `normal`，避免路由失败导致整个查询失败。

核心函数：

- `classify_intent()`

### `router.py`

路由与 handler 模块。

主要职责：

- 根据 intent 分发到不同 handler。
- 统一格式化 sources。
- 处理 `normal`、`position`、`multi_doc` 三类问题。
- 提供非流式路由和流式路由。

核心函数：

- `format_nodes_to_sources()`
- `handle_normal()`
- `handle_normal_stream()`
- `normalize_position()`
- `extract_position_and_source()`
- `handle_position()`
- `handle_multi_doc()`
- `route_query()`
- `route_query_stream()`

当前路由边界：

```text
route_query()
  -> 返回 dict
  -> 用于 /query、main.py、run_eval.py

route_query_stream()
  -> 返回 Iterator[str]
  -> 用于 /query_stream 和 Streamlit 前端
```

`HANDLER_MAP` 只服务非流式 `route_query()`，所有 handler 都必须返回 dict：

```text
normal    -> handle_normal()
position  -> handle_position()
multi_doc -> handle_multi_doc()
```

流式路由单独处理：

```text
if intent == "normal":
    yield from handle_normal_stream()
else:
    result = route_query()
    yield result["answer"]
```

因此不要把 `HANDLER_MAP["normal"]` 替换成 `handle_normal_stream()`，否则会破坏 `/query`、CLI 和评测链路。

### `service.py`

服务层模块。

主要职责：

- 初始化并持有 `RagComponents`。
- 对外提供非流式查询、流式查询、上传文档能力。
- 隔离 API / CLI / 前端与底层组件装配细节。

核心方法：

- `query(question)`：非流式查询，返回 dict。
- `query_stream(question)`：流式查询，返回 `Iterator[str]`。
- `upload_text(filename, content)`：上传文本并增量写入索引。
- `reload()`：重建运行期组件。

当前查询语义：

```text
query()
  -> classify_intent()
  -> route_query()
  -> {"answer": str, "sources": list}

query_stream()
  -> classify_intent()
  -> route_query_stream()
  -> Iterator[str]
```

## 4. 入口文件

### `api.py`

FastAPI 后端入口。

当前接口：

- `POST /query`
- `POST /query_stream`
- `POST /upload`
- `GET /health`

接口语义：

```text
POST /query
  -> rag_service.query()
  -> JSON: answer + sources + time_seconds

POST /query_stream
  -> rag_service.query_stream()
  -> text/plain streaming response
```

`/query` 适合需要 sources、评估、结构化消费的场景。

`/query_stream` 适合前端即时展示 answer 的场景。当前它只输出文本 answer，不输出 sources、done 或 error 事件。

### `app.py`

Streamlit 前端入口。

当前职责：

- 提供可视化问答界面。
- 通过 `/query_stream` 获取流式答案。
- 通过 `/upload` 上传文本文件。
- 使用 `st.form` 包装输入框和提交按钮，因此输入框内按回车可以触发检索。
- 使用 `requests.post(..., stream=True)` 和 `iter_content()` 逐步读取 answer。

当前限制：

- 前端只展示 answer。
- 前端不展示 sources，因为 `/query_stream` 当前只返回纯文本。
- `normal` 问题是 token 级流式。
- `position` 和 `multi_doc` 问题会复用原 handler，生成完成后一次性显示 answer。

### `main.py`

命令行入口。

当前职责：

- 加载配置。
- 初始化 `RagService`。
- 循环读取用户输入。
- 调用 `rag_service.query()`。
- 输出 answer 和 sources。

因为它走的是 `query()`，所以支持完整三类路由和 sources 展示。

### `run_eval.py` / `evaluator.py`

离线评测入口。

当前职责：

- 加载评测集。
- 调用服务或底层组件执行查询。
- 记录 answer、retrieved chunks、sources、score 等信息。
- 输出 CSV 评测结果。

评测链路应优先使用非流式 `query()`，因为评测通常需要 sources 和检索细节。

## 5. 三类问题的处理路径

### `normal`

非流式路径：

```text
question
  -> classify_intent() = normal
  -> route_query()
  -> handle_normal()
  -> query_engine.query()
  -> multi-query rewrite
  -> hybrid retrieval
  -> reranker / postprocessor
  -> LLM synthesis
  -> answer + sources
```

流式路径：

```text
question
  -> classify_intent() = normal
  -> route_query_stream()
  -> handle_normal_stream()
  -> streaming_query_engine.query()
  -> response.response_gen
  -> yield token
```

注意：

- `handle_normal()` 返回 dict，包含 sources。
- `handle_normal_stream()` 返回 token 文本流，不包含 sources。

### `position`

处理路径：

```text
question
  -> classify_intent() = position
  -> handle_position()
  -> extract_position_and_source()
  -> normalize_position()
  -> scan index.docstore.docs
  -> metadata filter
  -> select matched TextNode
  -> LLM synthesis
  -> answer + sources
```

特点：

- 不走标准 retriever。
- 不走 RRF。
- 主要依赖 metadata 中的 `position`、`source_file`、`content_type`。
- sources 多数来自 `TextNode`，没有真实检索分数。

在 `/query_stream` 下：

```text
route_query_stream()
  -> route_query()
  -> handle_position()
  -> yield result["answer"]
```

因此它在前端是一次性输出，不是 token 级流式。

### `multi_doc`

处理路径：

```text
question
  -> classify_intent() = multi_doc
  -> handle_multi_doc()
  -> LLM split into sub questions
  -> simple_retriever.retrieve() for each sub question
  -> merge top nodes
  -> build context
  -> LLM synthesis
  -> answer + sources
```

特点：

- 适合比较、综合、跨文档问题。
- 当前使用 `simple_retriever`，即 hybrid-only 检索路径。
- 子问题检索当前通过线程池并发执行。

在 `/query_stream` 下：

```text
route_query_stream()
  -> route_query()
  -> handle_multi_doc()
  -> yield result["answer"]
```

因此它在前端也是一次性输出，不是 token 级流式。

## 6. 数据与存储

### `data/`

原始数据目录。

通常存放：

- epub
- txt
- 其他原始文档

### `data_cleaned/`

清洗后的文本目录。

作用：

- 作为索引构建的直接输入。
- 是当前主要知识源目录。

### `storage/`

LlamaIndex 持久化索引目录。

作用：

- 避免每次启动都重新构建索引。
- 支持上传文档后的持久化。

### `golden_dataset.json`

评测数据集。

作用：

- 存放测试问题、期望答案、关键字、来源等信息。
- 用于离线评估检索和生成质量。

### `eval_*.csv`

评测输出结果。

作用：

- 保存每轮评测明细。
- 用于版本比较和问题分析。

## 7. 当前关键设计说明

### 7.1 非流式和流式入口分离

系统保留两套查询入口：

- `query()`：完整结果接口，返回 answer 和 sources。
- `query_stream()`：前端展示接口，返回文本流。

这样可以避免一个 handler 同时承担 dict 和 generator 两种返回类型。

### 7.2 `HANDLER_MAP` 只服务非流式路由

`HANDLER_MAP` 中的 handler 都必须返回 dict。

流式 normal 不应放进 `HANDLER_MAP`，否则会影响 `/query`、CLI 和评测。

### 7.3 前端统一走 `/query_stream`

Streamlit 当前统一调用 `/query_stream`。

这带来的行为是：

- 用户体验统一，前端只需要处理文本流。
- normal 有真实流式体验。
- position / multi_doc 功能不会丢失，但仍是一次性输出。
- 前端暂时不显示 sources。

### 7.4 sources 仍由非流式路径负责

当前 sources 只在 `/query`、CLI、评测链路中完整保留。

如果后续希望前端也展示 sources，建议把 `/query_stream` 升级为结构化流协议，例如：

```text
token event
sources event
done event
error event
```

在升级前，不建议让前端为了 sources 再额外请求一次 `/query`，因为这会重复执行完整 RAG 流程。

### 7.5 位置类问题的 score 语义

`position` 路径主要通过 metadata 过滤 `TextNode`，不是标准检索结果。

因此这类 source 没有真实检索分数。展示层和评测层应区分：

- 有真实 score 的 `NodeWithScore`。
- 无真实 score 的 `TextNode`。

## 8. 当前已知问题

- `router.py` 仍承担较多职责，包括路由、position 解析、handler 实现、source 格式化。
- `position` 路径直接扫描 `docstore`，数据规模变大后扩展性一般。
- 前端 `/query_stream` 暂不返回 sources。
- `position` 和 `multi_doc` 在 `/query_stream` 下不是 token 级流式。
- 多文档路径中的 reranker 语义需要继续和实际实现保持一致。

## 9. 后续演进方向

1. 为 `/query_stream` 设计结构化流协议，支持 token、sources、done、error。
2. 将 `position` 和 `multi_doc` 的最终 LLM synthesis 改造成真正 streaming。
3. 收敛 `router.py` 职责，将 intent 路由、handler、source formatter 逐步拆分。
4. 增强 `position` 路径的健壮性和空结果处理。
5. 明确 `TextNode` source 和 `NodeWithScore` source 的数据模型差异。
6. 将常用 prompt 集中管理。
7. 视数据规模决定是否为位置查询建立专门索引。
8. 清理历史实验文件、缓存文件和编码异常文档。

## 10. 已知边界

1. 标题不含辅助标记关键字的导读章节可能被错标，例如“庄子的艺术特色”。
2. chunk 第一行不是 heading、第二行才是 heading 的情况可能被忽略，例如《庄子》逍遥游题解中第一行是日期、第二行才是标题。
3. 这两类情况目前相对少见，短期内可通过评估标注区分“事实问题”和“开放问题”来规避。
