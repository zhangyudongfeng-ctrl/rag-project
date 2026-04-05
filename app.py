"""
app.py：Streamlit前端
启动：streamlit run app.py
"""

import os
import streamlit as st
import requests
import time

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="RAG 问答系统",
    page_icon="🔍",
    layout="centered"
)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# ==========================================
# 自定义样式
# ==========================================
st.markdown("""
<style>
    /* 全局字体 */
    .main { font-family: 'Helvetica Neue', sans-serif; }
    
    /* 标题区域 */
    .title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-icon {
        font-size: 3rem;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.6; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    .title-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: #E0E0E0;
        margin-top: 0.5rem;
    }
    .subtitle {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.3rem;
    }
    
    /* 出处卡片 */
    .source-card {
        background: #1E1E2E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-header {
        color: #7B8CDE;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .source-text {
        color: #CCC;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    .score-badge {
        background: #2D2D3F;
        color: #7B8CDE;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 标题区域
# ==========================================
st.markdown("""
<div class="title-container">
    <div class="title-icon">⚛</div>
    <div class="title-text">RAG 智能问答系统</div>
    <div class="subtitle">基于文献检索的问答引擎 · Multi-Query + Hybrid Search + Rerank</div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ==========================================
# 侧边栏：文件上传
# ==========================================
with st.sidebar:
    st.markdown("### 📁 文档管理")
    
    uploaded_file = st.file_uploader(
        "上传文档（.txt）",
        type=["txt"],
        help="上传新文档到知识库"
    )
    
    if uploaded_file is not None:
        if st.button("📤 上传到知识库", use_container_width=True):
            with st.spinner("正在处理文档..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    resp = requests.post(f"{API_BASE}/upload", files=files)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"上传成功！新增 {result['chunks_added']} 个切片")
                    else:
                        st.error(f"上传失败：{resp.text}")
                except requests.ConnectionError:
                    st.error("无法连接后端服务，请确认 FastAPI 已启动")
    
    st.divider()
    
    st.markdown("### ⚙️ 系统信息")
    # 检查后端状态
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3)
        if health.status_code == 200:
            st.markdown("🟢 后端服务：运行中")
        else:
            st.markdown("🔴 后端服务：异常")
    except:
        st.markdown("🔴 后端服务：未连接")


# ==========================================
# 主界面：问答
# ==========================================
question = st.text_input("请输入问题")
ask = st.button("🔍 提问")

if ask and question.strip():
    with st.spinner("检索中..."):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"question": question},
                timeout=120
            )
            
            if resp.status_code == 200:
                data = resp.json()
                
                # 回答
                st.markdown("### 📝 回答")
                st.markdown(data["answer"])
                
                # 耗时
                st.caption(f"⏱ 响应耗时：{data['time_seconds']} 秒")
                
                # 出处
                st.markdown("### 📚 参考来源")
                for i, src in enumerate(data["sources"]):
                    with st.expander(
                        f"片段 {i+1} · 相关度 {src['score']:.3f} · {src['source_file']}"
                    ):
                        if src["heading"]:
                            st.markdown(f"**章节：** {src['heading']}")
                        if src["position"]:
                            st.markdown(f"**位置：** {src['position']}")
                        st.markdown(f"**内容：**\n\n{src['text']}")
            else:
                st.error(f"查询失败：{resp.text}")
                
        except requests.ConnectionError:
            st.error("无法连接后端服务，请确认 FastAPI 已启动")
        except requests.Timeout:
            st.error("查询超时，请稍后重试")

elif ask:
    st.warning("请输入问题")
