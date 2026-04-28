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
# 自定义样式 (复刻赛博朋克彩色波纹效果)
# ==========================================
st.markdown("""
<style>
    /* 全局字体 */
    .main { font-family: 'Helvetica Neue', sans-serif; }
    
    /* 标题区域 */
    .title-container {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        overflow: hidden;
    }

    /* 赛博朋克波纹核心容器 */
    .idus-container {
        position: relative;
        width: 260px;
        height: 260px;
        margin: 0 auto 1.5rem auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* 最外层环境光晕 */
    .idus-glow {
        position: absolute;
        inset: -20px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(49,215,255,0.15) 0%, transparent 60%);
        animation: pulse-glow 3s ease-in-out infinite alternate;
    }

    /* ====== 外层彩色分段波纹 (核心视觉) ====== */
    .idus-wave-outer {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        /* 彩色渐变：青 -> 绿 -> 黄 -> 橙 -> 青 */
        background: conic-gradient(from 210deg, #31d7ff, #7bff9d, #ffd166, #ff7a59, #31d7ff);
        /* 核心技巧：使用双重遮罩 (径向镂空环 + 圆锥重复分割) 形成频谱条纹 */
        -webkit-mask:
            radial-gradient(circle, transparent 55%, #000 56%, #000 66%, transparent 67%),
            repeating-conic-gradient(from 0deg, #000 0deg, #000 1.5deg, transparent 1.5deg, transparent 3deg);
        -webkit-mask-composite: source-in;
        mask-composite: intersect;
        animation: spin-wave 20s linear infinite;
        filter: drop-shadow(0 0 8px rgba(123, 255, 157, 0.4));
    }

    /* ====== 内层细条纹辅助波纹 ====== */
    .idus-wave-inner {
        position: absolute;
        inset: 22px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #31d7ff, #7bff9d, #ffd166, #ff7a59, #31d7ff);
        -webkit-mask:
            radial-gradient(circle, transparent 61%, #000 62%, #000 65%, transparent 66%),
            repeating-conic-gradient(from 0deg, #000 0deg, #000 0.8deg, transparent 0.8deg, transparent 2deg);
        -webkit-mask-composite: source-in;
        mask-composite: intersect;
        animation: spin-wave-reverse 25s linear infinite;
        opacity: 0.85;
    }

    /* ====== 内部机械线圈 HUD ====== */
    .idus-ring-1 {
        position: absolute;
        inset: 52px;
        border-radius: 50%;
        border: 3px solid rgba(255, 255, 255, 0.9);
        border-left-color: transparent;
        border-right-color: transparent;
        animation: spin-fast 5s linear infinite;
        box-shadow: inset 0 0 10px rgba(49, 215, 255, 0.4), 0 0 10px rgba(49, 215, 255, 0.4);
    }

    .idus-ring-2 {
        position: absolute;
        inset: 64px;
        border-radius: 50%;
        border: 1px dashed rgba(49, 215, 255, 0.7);
        animation: spin-slow-reverse 15s linear infinite;
    }

    /* ====== 外围科幻准星标记 ====== */
    .idus-markers {
        position: absolute;
        inset: -15px;
        border-radius: 50%;
        border: 1px solid rgba(49, 215, 255, 0.15);
        box-shadow: 0 0 20px rgba(49,215,255,0.05);
    }
    .idus-markers::before, .idus-markers::after {
        content: "";
        position: absolute;
        background: rgba(49, 215, 255, 0.6);
    }
    .idus-markers::before {
        top: -8px; bottom: -8px; left: calc(50% - 0.5px); width: 1px;
    }
    .idus-markers::after {
        left: -8px; right: -8px; top: calc(50% - 0.5px); height: 1px;
    }

    /* ====== 中心 Logo 核心 ====== */
    .idus-core {
        position: relative;
        z-index: 10;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(10,15,25,0.9) 40%, rgba(10,15,25,0.4) 100%);
    }

    /* 类似图片中的中心镂空三角形 */
    .idus-triangle {
        width: 0; 
        height: 0; 
        border-left: 14px solid transparent;
        border-right: 14px solid transparent;
        border-bottom: 24px solid #fff;
        position: relative;
        margin-bottom: 4px;
        filter: drop-shadow(0 0 6px rgba(49,215,255,0.8));
    }
    .idus-triangle::after {
        content: "";
        position: absolute;
        top: 7px;
        left: -7px;
        width: 0;
        height: 0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        /* 模拟镂空 (此处使用 Streamlit 深色主题的近似背景色) */
        border-bottom: 12px solid #0e1117; 
    }

    .idus-text {
        font-family: 'Arial Black', Impact, sans-serif;
        font-size: 1.8rem;
        line-height: 1.2;
        font-weight: 900;
        letter-spacing: 4px;
        color: #fff;
        text-shadow: 0 0 8px rgba(255,255,255,0.5), 0 0 15px rgba(49,215,255,0.6);
    }

    .idus-subtext {
        font-family: 'Courier New', monospace;
        font-size: 0.45rem;
        color: #ffd166;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 2px;
        text-shadow: 0 0 5px rgba(255, 209, 102, 0.5);
    }

    /* ================= 动画关键帧 ================= */
    @keyframes spin-wave {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes spin-wave-reverse {
        0% { transform: rotate(360deg); }
        100% { transform: rotate(0deg); }
    }
    @keyframes spin-fast {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes spin-slow-reverse {
        0% { transform: rotate(360deg); }
        100% { transform: rotate(0deg); }
    }
    @keyframes pulse-glow {
        0% { transform: scale(0.95); opacity: 0.5; }
        100% { transform: scale(1.05); opacity: 0.8; }
    }

    /* 原有文本样式保留 */
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
</style>
""", unsafe_allow_html=True)


# ==========================================
# 标题区域 (替换为新的赛博朋克UI)
# ==========================================
st.markdown("""
<div class="title-container">
<div class="idus-container">
<div class="idus-glow"></div>
<div class="idus-wave-outer"></div>
<div class="idus-wave-inner"></div>
<div class="idus-ring-1"></div>
<div class="idus-ring-2"></div>
<div class="idus-markers"></div>
<div class="idus-core">
<div class="idus-triangle"></div>
<div class="idus-text">R.A.G</div>
<div class="idus-subtext">Intelligent System</div>
</div>
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
