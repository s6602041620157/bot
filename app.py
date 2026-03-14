from __future__ import annotations
from pathlib import Path
import streamlit as st
import base64
import os
from html import escape
from markdown_it import MarkdownIt
from rag.config import load_settings
from rag.pipeline import RAGPipeline

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="Askgiraffe - ผู้ช่วยคำถามภาควิชาคอมพิวเตอร์ศึกษาและภาควิชาครุศาสตร์โยธา", 
    page_icon="Askgiraffe.png",
    layout="wide"
)

# --- 2. ASSETS LOADING (Base64) ---
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return ""
    except: return ""

bg_image = get_base64_image("gb.jpg")
bot_logo = get_base64_image("Askgiraffe.png")

# --- 3. FULL CUSTOM CSS (ปรับปรุง Sidebar ให้สวยงามและอ่านง่าย) ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

    :root {{
        --font-main: 'Kanit', sans-serif !important;
    }}
    * {{ font-family: var(--font-main) !important; }}
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {{
        font-family: var(--font-main) !important;
    }}
    [data-testid="stIconMaterial"] {{
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
        font-variant-ligatures: normal !important;
        line-height: 1 !important;
    }}

    /* Background Setup */
    .stApp {{
        background-image: url('data:image/jpeg;base64,{bg_image}');
        background-size: cover; background-position: center;
        background-repeat: no-repeat; background-attachment: fixed;
    }}
    .stApp::before {{
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.85); z-index: -1;
    }}

    /* --- Sidebar Modern Emerald Design --- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #ECFDF5 0%, #D1FAE5 100%);
        border-right: 1px solid #A7F3D0;
    }}

    /* Sidebar Header */
    [data-testid="stSidebar"] h3 {{
        color: #064E3B !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-top: 1.5rem !important;
        border-left: 5px solid #10B981;
        padding-left: 10px !important;
    }}

    /* Sidebar Text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        
        font-weight: 500 !important;
    }}

    /* Sidebar Content Card */
    .sidebar-card {{
        background: white;
        padding: 1.25rem;
        border-radius: 1rem;
        border: 1px solid #A7F3D0;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }}

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        border-radius: 0.8rem !important;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.6rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.2) !important;
    }}
    [data-testid="stSidebar"] .stButton > button span {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(5, 150, 105, 0.3) !important;
    }}

    /* --- Chat Display Styling --- */
    .main-header {{
        font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .sub-header {{
        font-size: 1.3rem; font-weight: 600; color: #047857; text-align: center; margin-bottom: 2.5rem;
    }}
    .chat-thread {{
        width: 100%;
    }}
    .chat-row {{
        width: 100%;
        display: flex;
        margin-bottom: 0.95rem;
    }}
    .chat-row.user {{
        justify-content: flex-end;
    }}
    .chat-row.assistant {{
        justify-content: flex-start;
    }}
    .chat-bubble {{
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
        color: #1F2937;
    }}
    .chat-bubble.user {{
        width: min(62%, 900px);
        background: #FFFFFF;
        border: 2px solid #D1FAE5;
        border-right: 5px solid #10B981;
    }}
    .chat-bubble.assistant {{
        width: min(78%, 1100px);
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #059669;
    }}
    .bubble-label {{
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }}
    .bubble-label.user {{
        color: #047857;
        text-align: right;
    }}
    .bubble-label.assistant {{
        color: #065F46;
        text-align: left;
    }}
    .chat-bubble-content {{
        font-size: 1.02rem;
        line-height: 1.8;
        color: #111827;
        word-break: break-word;
    }}
    .chat-bubble-content p {{
        margin: 0.2rem 0 0.95rem;
    }}
    .chat-bubble-content p:last-child {{
        margin-bottom: 0;
    }}
    .chat-bubble-content ul,
    .chat-bubble-content ol {{
        margin: 0.35rem 0 1rem 1.2rem;
        padding-left: 0.55rem;
    }}
    .chat-bubble-content li {{
        margin: 0.2rem 0;
    }}
    .chat-bubble-content code {{
        background: #DCFCE7;
        color: #14532D;
        border-radius: 0.35rem;
        padding: 0.08rem 0.38rem;
        font-size: 0.94em;
    }}
    .chat-bubble-content pre {{
        margin: 0.7rem 0 1rem;
        background: #0F172A;
        color: #E2E8F0;
        border-radius: 0.8rem;
        padding: 0.95rem;
        overflow-x: auto;
    }}
    .chat-bubble-content pre code {{
        background: transparent;
        color: inherit;
        padding: 0;
    }}
    @media (max-width: 768px) {{
        .chat-bubble.user {{
            width: 90%;
        }}
        .chat-bubble.assistant {{
            width: 94%;
        }}
        .main-header {{
            font-size: 2.2rem;
        }}
        .sub-header {{
            font-size: 1.08rem;
            margin-bottom: 1.8rem;
        }}
    }}

    /* Welcome Stage */
    .welcome-stage {{ min-height: 48vh; display: flex; align-items: center; justify-content: center; }}
    .welcome-card {{
        width: 100%; max-width: 820px; padding: 2.25rem 2rem; border-radius: 1.25rem;
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #A7F3D0; box-shadow: 0 10px 26px rgba(5, 150, 105, 0.16); text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# --- 4. CORE LOGIC ---
@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()

_markdown_renderer = MarkdownIt("commonmark", {"html": False, "breaks": True})

def render_chat_bubble(message: dict[str, object]) -> None:
    role = "user" if str(message.get("role", "")) == "user" else "assistant"
    label = "ผู้ถาม" if role == "user" else "Askgiraffe"
    raw_content = str(message.get("content", ""))
    content_html = _markdown_renderer.render(escape(raw_content))
    st.markdown(
        f"""
        <div class="chat-thread">
            <div class="chat-row {role}">
                <div class="chat-bubble {role}">
                    <div class="bubble-label {role}">{label}</div>
                    <div class="chat-bubble-content">{content_html}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _recent_history(messages: list[dict], turns: int) -> list[dict]:
    max_messages = max(1, turns * 2)
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages if msg.get("role") in {"user", "assistant"}][-max_messages:]

# --- 5. SIDEBAR (ปรับปรุงให้สวยงามและอ่านง่าย) ---
settings = load_settings()
pipeline = get_pipeline()
auto_heal_status = pipeline.ensure_pdf_index_healthy()
index_stats = pipeline.get_index_stats()

with st.sidebar:
    # Branding Area
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem; background: white; padding: 1.5rem; border-radius: 1.5rem; border: 2px solid #10B981; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <img src="data:image/png;base64,{bot_logo}" style="width: 70px; margin-bottom: 10px;">
            <div style="font-size: 1.6rem; font-weight: 800; color: #064E3B;">Askgiraffe</div>
            <div style="font-size: 0.95rem; color: #374151; font-weight: 500; margin-top: 5px;">
                ผู้ช่วยตอบคำถามภาควิชาคอมพิวเตอร์ศึกษาและ<div>ภาควิชาครุศาสตร์โยธา คณะครุศาสตร์อุตสาหกรรม มจพ.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # System Control
    st.markdown("### ⚙️ จัดการระบบ")
    if st.button("🗑️ ล้างประวัติการสนทนา"):
        st.session_state["messages"] = []
        st.rerun()

    # Document Stats
    st.markdown("### 📊 ระบบเอกสาร")
    pdf_files = sorted(Path(settings.pdf_dir).glob("*.pdf"))
    st.markdown(f"""
        <div class="sidebar-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1rem; color: #111827;">ไฟล์ในระบบ:</span>
                <span style="background: #064E3B; color: white; padding: 4px 12px; border-radius: 8px; font-weight: 700;">
                    {len(pdf_files)} ไฟล์
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="sidebar-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                <span style="font-size: 1rem; color: #111827;">Index health:</span>
                <span style="background: #065F46; color: white; padding: 4px 12px; border-radius: 8px; font-weight: 700;">
                    {index_stats.get("document_count", 0)} docs / {index_stats.get("chunk_count", 0)} chunks
                </span>
            </div>
            <div style="font-size: 0.88rem; color: #374151;">
                Auto-heal: {auto_heal_status.get("action", "unknown")}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 อัปเดต Index เอกสาร"):
        with st.spinner("กำลังอัปเดต..."):
            pipeline.index_pdfs(pdf_dir=str(settings.pdf_dir), reset=True)
        st.success("อัปเดตสำเร็จ!")

    # Contact Card
    st.markdown("### 📞 ติดต่อสอบถาม")
    st.markdown(f"""
        <div class="sidebar-card">
            <div style="color: #111827; line-height: 1.8;">
                📧 <a href="http://admission.kmutnb.ac.th" style="color: #059669; text-decoration: none; font-weight: 700;">admission.kmutnb.ac.th</a><br>
                ☎️ <span style="font-weight: 700;">02-555-2000</span><br>
                🏢 <span style="font-size: 0.9rem; font-weight: 600;">คณะครุศาสตร์อุตสาหกรรม</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 6. MAIN CONTENT AREA ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Welcome Screen
if len(st.session_state["messages"]) == 0:
    st.markdown(f"""
        <div class="welcome-stage">
            <div class="welcome-card">
                <div style="color: #047857; font-size: 2.2rem; font-weight: 800; margin-bottom: 0.75rem;">สวัสดีครับ 👋 ยินดีต้อนรับสู่ Askgiraffe</div>
                <p style="color: #1F2937; font-size: 1.1rem; font-weight: 500;">ผมพร้อมช่วยตอบคำถามเกี่ยวภาควิชาคอมพิวเตอร์ศึกษาและภาควิชาครุศาสตร์โยธา<br>พิมพ์คำถามของคุณได้เลย แล้วผมจะช่วยค้นหาคำตอบให้อย่างรวดเร็ว</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 2.5, 1])
    with center_col:
        st.markdown('<div style="text-align: center; color: #047857; font-weight: 600; margin-bottom: 0.5rem;">เริ่มถามคำถามแรกของคุณได้ที่นี่</div>', unsafe_allow_html=True)
        query = st.chat_input("✨ พิมพ์คำถามของคุณที่นี่...")
else:
    # Header
    st.markdown(f'<div class="main-header"><img src="data:image/png;base64,{bot_logo}" style="width: 80px; vertical-align: middle; margin-right: 15px;"> Askgiraffe</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ผู้ช่วยของภาควิชาคอมพิวเตอร์ศึกษาและภาควิชาครุศาสตร์โยธา</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #059669; font-weight: 600; border-left: 5px solid #10B981; padding-left: 15px;">💬 พื้นที่แชท</h3>', unsafe_allow_html=True)

    # Chat History
    for msg in st.session_state["messages"]:
        render_chat_bubble(msg)

    query = st.chat_input("✨ พิมพ์คำถามของคุณที่นี่...")

# --- 7. CHAT PROCESSING ---
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    st.rerun()

if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    last_query = st.session_state["messages"][-1]["content"]
    try:
        history = _recent_history(st.session_state["messages"][:-1], turns=settings.memory_turns)
        stream = pipeline.ask_stream(query=last_query, history=history)
        full_response = st.write_stream(stream)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response,
            "citations": [],
        })
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
    st.rerun()
