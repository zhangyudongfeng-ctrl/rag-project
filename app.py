"""
app.py：Streamlit前端
启动：streamlit run app.py
"""

import os
import streamlit as st
import requests
import time
from logging_config import setup_logging
setup_logging()

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
        padding: 1.5rem 0 1rem 0;
    }
    .radar-icon {
        position: relative;
        width: 156px;
        height: 156px;
        margin: 0 auto 1rem auto;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background:
            radial-gradient(circle at 50% 50%, rgba(255,255,255,0.16) 0 25%, transparent 26%),
            conic-gradient(from 210deg, #31d7ff, #7bff9d, #ffd166, #ff7a59, #31d7ff);
        box-shadow: 0 0 28px rgba(49, 215, 255, 0.22);
        animation: radar-glow 2.8s ease-in-out infinite;
    }
    .radar-icon::before {
        content: "";
        position: absolute;
        inset: 10px;
        border-radius: 50%;
        background:
            repeating-conic-gradient(from 0deg, rgba(255,255,255,0.88) 0deg 1.2deg, transparent 1.2deg 3.8deg),
            radial-gradient(circle, transparent 0 58%, rgba(8, 14, 20, 0.94) 59%);
        mask: radial-gradient(circle, transparent 0 45%, #000 46% 72%, transparent 73%);
    }
    .radar-icon::after {
        content: "";
        position: absolute;
        inset: 18px;
        border-radius: 50%;
        border: 5px solid rgba(255,255,255,0.92);
        box-shadow: inset 0 0 18px rgba(49, 215, 255, 0.18);
    }
    .radar-mark {
        position: relative;
        z-index: 2;
        width: 68px;
        height: 68px;
    }
    .radar-mark::before,
    .radar-mark::after {
        content: "";
        position: absolute;
        border: 7px solid #f7fbff;
        transform: rotate(45deg);
    }
    .radar-mark::before {
        width: 38px;
        height: 38px;
        left: 4px;
        top: 4px;
        border-right-color: transparent;
        border-bottom-color: transparent;
    }
    .radar-mark::after {
        width: 34px;
        height: 34px;
        right: 4px;
        bottom: 4px;
        border-left-color: transparent;
        border-top-color: transparent;
    }
    .radar-dot {
        position: absolute;
        z-index: 3;
        width: 20px;
        height: 20px;
        border: 5px solid #ff8a5c;
        border-radius: 4px;
        transform: rotate(45deg);
        background: rgba(255,255,255,0.08);
    }
    @keyframes radar-glow {
        0%, 100% { filter: saturate(1); transform: scale(1); }
        50% { filter: saturate(1.35); transform: scale(1.025); }
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
    <div class="radar-icon">
        <div class="radar-mark"></div>
        <div class="radar-dot"></div>
    </div>
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
with st.form("query_form", clear_on_submit=False):
    question = st.text_input("请输入问题")
    ask = st.form_submit_button("Search")

if ask and question.strip():
    with st.spinner("检索中..."):
        try:
            resp = requests.post(
                f"{API_BASE}/query_stream",
                json={"question": question},
                timeout=120,
                stream=True,
            )
            
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                st.markdown("### Answer")
                answer_box = st.empty()
                answer = ""
                for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
                    if chunk:
                        answer += chunk
                        answer_box.markdown(answer)
                
            else:
                st.error(f"查询失败：{resp.text}")
                
        except requests.ConnectionError:
            st.error("无法连接后端服务，请确认 FastAPI 已启动")
        except requests.Timeout:
            st.error("查询超时，请稍后重试")

elif ask:
    st.warning("请输入问题")
