# -*- coding: utf-8 -*-
"""
Nep-RAG · नेपाली दस्तावेज पुनःप्राप्ति प्रणाली
Nepali Document Retrieval Augmented Generation — Streamlit frontend.

WHAT THIS FILE IS
------------------
A complete, runnable Streamlit UI for a Nepali-language RAG system: upload
PDFs/images (newspapers, books, magazines), ask questions in Devanagari or
English, and get answers grounded in cited sources — presented like a
ChatGPT/Claude-style chat, with expandable "archive ticket" citation cards.

WHAT IS MOCKED (clearly marked below)
--------------------------------------
There is no real embedding model / vector database / LLM wired in here,
because that is backend work, not frontend work. Retrieval is done with a
simple keyword-overlap scorer over a small in-memory corpus + your uploaded
files, and answer "generation" is template text built from the retrieved
snippets. Every place you need to swap in real logic is marked:

    # >>> REPLACE WITH REAL BACKEND CALL <<<

Run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import io
import re
import textwrap
from datetime import datetime

import streamlit as st

try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


# ============================================================================
# 0. PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Nep-RAG · नेपाली दस्तावेज पुनःप्राप्ति प्रणाली",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# 1. DEVANAGARI NUMERAL HELPERS
# ============================================================================

_DEV_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def dev_num(n: int) -> str:
    """Format an integer with thousands separators, in Devanagari digits."""
    return f"{n:,}".translate(_DEV_DIGITS)


# ============================================================================
# 2. MOCK KNOWLEDGE BASE  (illustrative demo content — not live data)
# ============================================================================

KB_STATS = {
    "newspapers": 4230,
    "books": 1065,
    "magazines": 673,
}
KB_STATS["total"] = sum(KB_STATS.values())

# Small illustrative corpus so the UI has something real to retrieve from.
# Replace with your actual indexed document store.
MOCK_CORPUS = [
    {
        "id": "kb-001",
        "type": "समाचारपत्र",
        "type_en": "Newspaper",
        "title": "गोरखापत्र — आर्थिक सर्वेक्षण विशेष अंक",
        "publisher": "गोरखापत्र संस्थान",
        "date": "२०८१ असार",
        "page": 4,
        "text": (
            "नेपालको चालु आर्थिक वर्षमा कुल गार्हस्थ्य उत्पादन (जीडीपी) वृद्धिदर "
            "कृषि, पर्यटन र रेमिट्यान्समा आधारित रहेको देखिन्छ। सरकारले पूर्वाधार "
            "लगानी र निजी क्षेत्रको सहभागिता बढाउने नीति अघि सारेको छ।"
        ),
    },
    {
        "id": "kb-002",
        "type": "पुस्तक",
        "type_en": "Book",
        "title": "नेपालको संवैधानिक इतिहास",
        "publisher": "साझा प्रकाशन",
        "date": "२०७५",
        "page": 112,
        "text": (
            "संविधान सभाको पहिलो निर्वाचन २०६४ सालमा भएको थियो। लामो राजनीतिक "
            "छलफल र सहमतिपछि नेपालको संविधान २०७२ सालमा जारी भयो, जसले "
            "संघीय लोकतान्त्रिक गणतन्त्रको आधार तयार पार्यो।"
        ),
    },
    {
        "id": "kb-003",
        "type": "पत्रिका",
        "type_en": "Magazine",
        "title": "हिमाल खबरपत्रिका — पहाडी जीवनशैली विशेषांक",
        "publisher": "हिमाल मिडिया",
        "date": "२०८० कार्तिक",
        "page": 27,
        "text": (
            "हिमाली क्षेत्रका समुदायहरूको जीवनशैली याक पालन, मौसमी बसाइँसराइ र "
            "स्थानीय चाडपर्वसँग गाँसिएको छ। पर्यटनको विस्तारले परम्परागत संस्कृतिमा "
            "नयाँ चुनौती र अवसर दुवै ल्याएको छ।"
        ),
    },
    {
        "id": "kb-004",
        "type": "समाचारपत्र",
        "type_en": "Newspaper",
        "title": "कान्तिपुर — शिक्षा पृष्ठ",
        "publisher": "कान्तिपुर प्रकाशन",
        "date": "२०८१ जेठ",
        "page": 8,
        "text": (
            "काठमाडौं उपत्यकाका सामुदायिक विद्यालयहरूमा डिजिटल पठनपाठन सामग्रीको "
            "प्रयोग बढेको छ। शिक्षा मन्त्रालयले शिक्षक तालिम र पाठ्यक्रम अद्यावधिकमा "
            "जोड दिने जनाएको छ।"
        ),
    },
]


# ============================================================================
# 3. SESSION STATE
# ============================================================================


def init_state():
    defaults = {
        "messages": [],          # chat transcript: list of {role, content, sources}
        "uploaded_docs": [],     # list of {name, size_mb, pages, text, kind}
        "mode": "chat",          # "chat" | "search"
        "dark": False,
        "pending_query": None,   # set by sample-question buttons
        "search_query": "",
        "search_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================================
# 4. THEME / CSS  — lokta-paper + government-seal (छाप) motif
# ============================================================================


def inject_css(dark: bool):
    tokens = dict(
        paper="#F7F3EB", paper_alt="#F7F3EB", ink="#221B16",
        ink_soft="#5A4F46", sindoor="#9D1730", sindoor_dark="#791428",
        indigo="#17324D", gold="#916A12", sage="#55684A",
        border="rgba(34,27,22,0.12)", shadow="rgba(34,27,22,0.10)",
    )

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Yatra+One&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Work+Sans:wght@400;500;600&display=swap');

    :root {{
        --paper: {tokens['paper']};
        --paper-alt: {tokens['paper_alt']};
        --ink: {tokens['ink']};
        --ink-soft: {tokens['ink_soft']};
        --sindoor: {tokens['sindoor']};
        --sindoor-dark: {tokens['sindoor_dark']};
        --indigo: {tokens['indigo']};
        --gold: {tokens['gold']};
        --sage: {tokens['sage']};
        --border: {tokens['border']};
        --shadow: {tokens['shadow']};
    }}

    html, body, [class*="css"] {{
        font-family: 'Noto Sans Devanagari', 'Work Sans', sans-serif;
    }}

    .stApp {{
        background-color: var(--paper);
        background-image: none;
        color: var(--ink);
    }}

    .main .block-container {{
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}

    .stApp, .stApp p, .stApp li, .stApp label, .stApp div, .stApp span {{
        color: var(--ink);
    }}

    .stApp p, .stApp li {{
        line-height: 1.72;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: var(--paper);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--ink); }}

    section[data-testid="stSidebar"] .stButton>button,
    section[data-testid="stSidebar"] [data-testid="stChatInput"] textarea,
    section[data-testid="stSidebar"] input {{
        background-color: var(--paper);
        color: var(--ink);
    }}

    /* ---------- Headings use the display faces ---------- */
    h1, h2, h3 {{
        font-family: 'Fraunces', 'Yatra One', serif;
        color: var(--ink);
        letter-spacing: 0.2px;
    }}
    .dev-display {{ font-family: 'Yatra One', 'Noto Sans Devanagari', serif; }}

    /* ---------- Buttons ---------- */
    .stButton>button {{
        background-color: var(--paper);
        color: var(--ink);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.15s ease;
    }}
    .stButton>button:hover {{
        border-color: var(--sindoor);
        color: var(--sindoor);
        transform: translateY(-1px);
        box-shadow: 0 4px 10px var(--shadow);
    }}
    div[data-testid="stFormSubmitButton"] button,
    .primary-btn button {{
        background-color: var(--sindoor) !important;
        color: #FBF6EA !important;
        border: none !important;
    }}

    /* ---------- Chat bubbles ---------- */
    [data-testid="stChatMessage"] {{
        background-color: var(--paper);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 8px 10px;
        box-shadow: 0 4px 16px var(--shadow);
    }}

    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] textarea,
    .stTextInput input {{
        font-family: 'Noto Sans Devanagari', sans-serif;
        color: var(--ink);
        background-color: var(--paper-alt);
    }}

    [data-testid="stChatInput"] textarea::placeholder,
    .stTextInput input::placeholder {{
        color: var(--ink-soft);
        opacity: 1;
    }}

    /* ---------- Containers with border (cards) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px !important;
        border-color: var(--border) !important;
        background-color: var(--paper);
        box-shadow: 0 4px 16px var(--shadow);
    }}

    /* ---------- Seal / logo mark ---------- */
    .seal-wrap {{
        display: flex; justify-content: center; margin-top: 8px;
        position: relative;
    }}
    .seal-wrap::before {{
        content: "";
        position: absolute; top: -40px; left: 50%; transform: translateX(-50%);
        width: 260px; height: 260px; border-radius: 50%;
        background: repeating-radial-gradient(circle, {tokens['gold']}22 0 2px, transparent 2px 22px);
        z-index: 0;
    }}
    .seal {{
        width: 84px; height: 84px; border-radius: 50%;
        background: var(--sindoor);
        border: 3px solid var(--gold);
        outline: 1px solid var(--sindoor-dark);
        outline-offset: 3px;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Yatra One', serif;
        font-size: 2.4rem; color: #FBF6EA;
        box-shadow: 0 6px 18px var(--shadow);
        position: relative; z-index: 1;
    }}

    .hero-title {{
        text-align: center; margin-top: 14px; margin-bottom: 0;
        font-family: 'Yatra One', serif; font-size: 2.3rem; color: var(--ink);
    }}
    .hero-sub-np {{ text-align: center; color: var(--sindoor); font-size: 1.05rem; margin-top: 2px; }}
    .hero-sub-en {{ text-align: center; color: var(--ink-soft); font-family: 'Fraunces', serif; font-size: 1.05rem; margin-top: 10px; }}
    .hero-sub-en2 {{ text-align: center; color: var(--ink-soft); font-size: 0.92rem; margin-top: 2px; }}

    .stCaption, .stCaption p {{ color: var(--ink-soft) !important; }}

    /* ---------- Citation ticket ---------- */
    .ticket {{
        background: var(--paper);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-top: 2px dotted var(--gold);
    }}
    .ticket-head {{
        display: flex; justify-content: space-between; align-items: baseline;
        font-family: 'Fraunces', serif; font-weight: 700; color: var(--indigo);
        font-size: 0.95rem;
    }}
    .ticket-tag {{
        font-size: 0.72rem; color: var(--sindoor); border: 1px solid var(--sindoor);
        border-radius: 20px; padding: 1px 8px; font-family: 'Work Sans', sans-serif;
    }}
    .ticket-meta {{ font-size: 0.78rem; color: var(--ink-soft); margin-top: 2px; }}
    .ticket-body {{ font-size: 0.9rem; color: var(--ink); margin-top: 6px; line-height: 1.5; }}

    /* ---------- Stat card ---------- */
    .stat-card {{
        background: var(--paper); border: 1px solid var(--border);
        border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
        box-shadow: 0 3px 12px var(--shadow);
    }}
    .stat-num {{ font-family: 'Fraunces', serif; font-size: 1.3rem; color: var(--sindoor); }}
    .stat-label-np {{ font-size: 0.85rem; color: var(--ink); }}
    .stat-label-en {{ font-size: 0.7rem; color: var(--ink-soft); }}

    .badge-demo {{
        display: inline-block; font-size: 0.72rem; color: var(--sage);
        border: 1px solid var(--sage); border-radius: 20px; padding: 1px 10px;
        margin-left: 8px; vertical-align: middle;
    }}

    hr {{ border-color: var(--border); }}

    .stInfo, .stWarning, .stSuccess, .stError {{
        border-radius: 12px;
        border-color: var(--border);
    }}

    @media (max-width: 640px) {{
        .hero-title {{ font-size: 1.6rem; }}
        .seal {{ width: 64px; height: 64px; font-size: 1.8rem; }}
        .main .block-container {{ padding-top: 1rem; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_css(st.session_state.dark)


# ============================================================================
# 5. RETRIEVAL + "GENERATION"  (mock — swap for your real backend)
# ============================================================================

_WORD_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")


def _tokenize(text: str):
    return set(w.lower() for w in _WORD_RE.findall(text))


def build_corpus():
    """Combine the mock KB with any user-uploaded document chunks."""
    corpus = list(MOCK_CORPUS)
    for doc in st.session_state.uploaded_docs:
        chunks = textwrap.wrap(doc["text"], 400) or [""]
        for i, chunk in enumerate(chunks[:6]):
            corpus.append({
                "id": f"upload-{doc['name']}-{i}",
                "type": "अपलोड गरिएको फाइल",
                "type_en": "Uploaded file",
                "title": doc["name"],
                "publisher": "तपाईंको अपलोड · Your upload",
                "date": doc.get("uploaded_at", ""),
                "page": i + 1,
                "text": chunk,
            })
    return corpus


def retrieve(query: str, k: int = 3):
    """
    Naive keyword-overlap retrieval over the demo corpus.

    >>> REPLACE WITH REAL BACKEND CALL <<<
    Swap this for a real semantic search: embed `query` with your Nepali-aware
    embedding model, search your vector store (FAISS/Milvus/pgvector/etc.),
    and return the top-k chunks with their metadata.
    """
    q_tokens = _tokenize(query)
    scored = []
    for entry in build_corpus():
        overlap = len(q_tokens & _tokenize(entry["text"] + " " + entry["title"]))
        if overlap > 0:
            scored.append((overlap, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [e for _, e in scored[:k]]
    if not top:
        top = MOCK_CORPUS[:k]  # fallback so the UI always demonstrates citations
    return top


def generate_answer(query: str, sources: list):
    """
    Template-based "answer" built from retrieved snippets.

    >>> REPLACE WITH REAL BACKEND CALL <<<
    This is where you'd call your LLM (e.g. the Anthropic API) with the
    retrieved `sources` as grounding context, something like:

        from anthropic import Anthropic
        client = Anthropic()  # reads ANTHROPIC_API_KEY from env
        context = "\n\n".join(f"[{s['title']}] {s['text']}" for s in sources)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": (
                    f"तलका स्रोतहरूको आधारमा प्रश्नको जवाफ देवनागरीमा दिनुहोस्।\n\n"
                    f"स्रोतहरू:\n{context}\n\nप्रश्न: {query}"
                ),
            }],
        )
        return response.content[0].text
    """
    intro = f"तपाईंको प्रश्न — “{query}” — सम्बन्धी उपलब्ध स्रोतहरूमा भेटिएको जानकारी:"
    body = " ".join(s["text"] for s in sources[:2])
    outro = "(यो डेमो उत्तर हो — वास्तविक प्रयोगमा यहाँ भाषा मोडेलको जवाफ देखिनेछ।)"
    return f"{intro}\n\n{body}\n\n*{outro}*"


# ============================================================================
# 6. SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:38px;height:38px;border-radius:50%;background:var(--sindoor);
                        border:2px solid var(--gold); display:flex;align-items:center;
                        justify-content:center;color:#FBF6EA;font-family:'Yatra One',serif;
                        font-size:1.2rem;">न</div>
            <div>
                <div style="font-family:'Fraunces',serif;font-weight:700;font-size:1.05rem;">Nep-RAG</div>
                <div style="font-size:0.75rem;color:var(--ink-soft);">देवनागरी · Devanagari</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("नेपाली दस्तावेज पुनः प्राप्ति प्रणाली · Nepali Document Retrieval System")
    st.divider()

    st.markdown("**📤 दस्तावेज थप्नुस् · Upload Documents**")
    uploaded = st.file_uploader(
        "PDF वा छवि अपलोड गर्नुस् · Upload PDF or Image",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            if any(d["name"] == f.name for d in st.session_state.uploaded_docs):
                continue
            kind = "pdf" if f.name.lower().endswith(".pdf") else "image"
            text, pages = "", None
            if kind == "pdf" and PDF_OK:
                try:
                    reader = PdfReader(io.BytesIO(f.getvalue()))
                    pages = len(reader.pages)
                    text = "\n".join((p.extract_text() or "") for p in reader.pages)
                except Exception:
                    text = ""
            elif kind == "image":
                # >>> REPLACE WITH REAL BACKEND CALL <<<
                # Plug OCR here (e.g. pytesseract) to extract Devanagari text
                # from scanned newspaper/book pages before indexing.
                text = ""
                pages = 1
            st.session_state.uploaded_docs.append({
                "name": f.name,
                "size_mb": round(len(f.getvalue()) / (1024 * 1024), 1),
                "pages": pages or "-",
                "text": text or f"({f.name} — पाठ अझै निकालिएको छैन · text not yet extracted)",
                "kind": kind,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d"),
            })
        st.rerun()

    if st.session_state.uploaded_docs:
        st.markdown("**अपलोड गरिएका · Uploaded**")
        for i, d in enumerate(st.session_state.uploaded_docs):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"<div class='stat-card' style='padding:8px 10px;'>"
                    f"📄 <b>{d['name']}</b><br>"
                    f"<span class='stat-label-en'>{d['size_mb']} MB · {d['pages']} पृ.</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"rm_{i}", help="हटाउनुस् · Remove"):
                    st.session_state.uploaded_docs.pop(i)
                    st.rerun()

    st.divider()
    st.markdown("**🗂️ ज्ञान भण्डार · Knowledge Base**")
    kb_rows = [
        ("📰", "समाचारपत्र", "Newspapers", KB_STATS["newspapers"]),
        ("📖", "पुस्तकहरू", "Books", KB_STATS["books"]),
        ("🗞️", "पत्रिकाहरू", "Magazines", KB_STATS["magazines"]),
        ("🌐", "जम्मा", "Total", KB_STATS["total"]),
    ]
    for icon, np_label, en_label, count in kb_rows:
        st.markdown(
            f"""<div class="stat-card" style="display:flex;justify-content:space-between;align-items:center;">
                    <div>{icon} <span class="stat-label-np">{np_label}</span><br>
                    <span class="stat-label-en">{en_label}</span></div>
                    <div class="stat-num">{dev_num(count)}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    with st.expander("⚙️ सेटिङ · Settings"):
        st.session_state.dark = st.toggle(
            "🌙 अँध्यारो मोड · Dark mode", value=st.session_state.dark
        )
        if st.button("🗑️ कुराकानी मेटाउनुस् · Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================================
# 7. TOP BAR — mode toggle + how-to-use
# ============================================================================

top_l, top_r = st.columns([3, 2])
with top_l:
    st.session_state.mode = st.radio(
        "mode",
        options=["chat", "search"],
        format_func=lambda m: "💬 संवाद · Chat" if m == "chat" else "🔍 खोज · Search",
        horizontal=True,
        label_visibility="collapsed",
    )
with top_r:
    with st.popover("❓ कसरी प्रयोग गर्ने · How to use", use_container_width=True):
        st.markdown(
            """
**१. प्रश्न सोध्नुस् · Ask in Nepali** — तलको बक्समा देवनागरी वा English मा प्रश्न लेखेर पठाउनुस्।
*Type your question below in Devanagari or English and send it.*

**२. दस्तावेज अपलोड गर्नुस् · Upload documents** — बायाँ प्यानलबाट PDF वा JPG/PNG थप्नुस्।
*Add PDF or image files from the left panel.*

**३. स्रोत जाँच्नुस् · Check sources** — हरेक उत्तरमुनि उद्धृत स्रोत कार्डहरू विस्तार गर्न क्लिक गर्नुस्।
*Expand the citation cards under each answer to see exactly where it came from.*

**४. मोड छान्नुस् · Choose a mode** — माथि 'संवाद' (कुराकानी) वा 'खोज' (मात्र नतिजा) मोड छान्नुस्।
*Use Chat for a conversation, or Search for a plain ranked list of source snippets.*
            """
        )

st.divider()


# ============================================================================
# 8. CITATION RENDERING
# ============================================================================


def render_sources(sources: list, key_prefix: str):
    with st.expander(f"📎 स्रोतहरू हेर्नुस् · View {len(sources)} source(s)"):
        for s in sources:
            st.markdown(
                f"""
                <div class="ticket">
                    <div class="ticket-head">
                        <span>{s['title']}</span>
                        <span class="ticket-tag">{s['type']} · {s['type_en']}</span>
                    </div>
                    <div class="ticket-meta">{s['publisher']} · {s['date']} · पृष्ठ {dev_num(s['page']) if isinstance(s['page'], int) else s['page']}</div>
                    <div class="ticket-body">{s['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# 9A. CHAT MODE
# ============================================================================


def run_chat_turn(query: str):
    st.session_state.messages.append({"role": "user", "content": query, "sources": None})
    sources = retrieve(query)
    answer = generate_answer(query, sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if st.session_state.mode == "chat":

    if not st.session_state.messages:
        # ---- Welcome / hero state ----
        st.markdown('<div class="seal-wrap"><div class="seal">न</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Nep-RAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub-np">नेपाली दस्तावेज पुनः प्राप्ति प्रणाली</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub-en">Nepali Document Retrieval Augmented Generation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub-en2">समाचारपत्र, पुस्तक र पत्रिकाहरूबाट स्रोत-सहित सटीक उत्तर प्राप्त गर्नुस्<br>'
            'Get precise, source-cited answers from newspapers, books and magazines</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown("<h5 style='text-align:center;color:var(--ink-soft);'>नमूना प्रश्नहरू · Sample questions</h5>", unsafe_allow_html=True)
        samples = [
            ("नेपालको आर्थिक वृद्धिदर के छ?", "What is Nepal's economic growth rate?"),
            ("संविधान सभाको इतिहास बताउनुस्", "History of the Constituent Assembly"),
            ("हिमाली क्षेत्रको संस्कृति", "Culture of the Himalayan region"),
            ("काठमाडौंको शिक्षा प्रणाली", "Education system of Kathmandu"),
        ]
        cols = st.columns(2)
        for i, (np_q, en_q) in enumerate(samples):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{np_q}**")
                    st.caption(en_q)
                    if st.button("सोध्नुस् · Ask", key=f"sample_{i}", use_container_width=True):
                        st.session_state.pending_query = np_q
                        st.rerun()

    else:
        # ---- Transcript ----
        for i, msg in enumerate(st.session_state.messages):
            avatar = "🧑" if msg["role"] == "user" else "📜"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg["sources"]:
                    render_sources(msg["sources"], key_prefix=f"msg_{i}")

    # ---- Handle a sample-question click ----
    if st.session_state.pending_query:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        run_chat_turn(q)
        st.rerun()

    # ---- Chat input ----
    user_q = st.chat_input("नेपालीमा प्रश्न सोध्नुस्... · Ask a question in Nepali...")
    if user_q:
        run_chat_turn(user_q)
        st.rerun()


# ============================================================================
# 9B. SEARCH MODE
# ============================================================================

else:
    st.markdown("### 🔍 खोज मोड · Search mode")
    st.caption("कुराकानी बिना सिधै श्रेणीबद्ध स्रोत खण्डहरू हेर्नुस् · See ranked source snippets directly, without a conversation.")

    with st.form("search_form", clear_on_submit=False):
        c1, c2 = st.columns([5, 1])
        with c1:
            query = st.text_input(
                "खोज्नुस्",
                value=st.session_state.search_query,
                placeholder="नेपालीमा वा English मा खोज्नुस्... · Search in Nepali or English...",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("खोज्नुस् · Search", use_container_width=True)

    if submitted and query.strip():
        st.session_state.search_query = query
        st.session_state.search_results = retrieve(query, k=5)

    if st.session_state.search_results:
        st.markdown(f"**{len(st.session_state.search_results)} नतिजाहरू फेला परे · results found**"
                    f"<span class='badge-demo'>डेमो डाटा · demo data</span>", unsafe_allow_html=True)
        for r in st.session_state.search_results:
            st.markdown(
                f"""
                <div class="ticket">
                    <div class="ticket-head">
                        <span>{r['title']}</span>
                        <span class="ticket-tag">{r['type']} · {r['type_en']}</span>
                    </div>
                    <div class="ticket-meta">{r['publisher']} · {r['date']} · पृष्ठ {dev_num(r['page']) if isinstance(r['page'], int) else r['page']}</div>
                    <div class="ticket-body">{r['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif st.session_state.search_query:
        st.info("कुनै नतिजा फेला परेन · No results found. फरक शब्दहरू प्रयोग गरी हेर्नुस् · Try different keywords.")


# ============================================================================
# 10. FOOTER
# ============================================================================

st.divider()
st.markdown(
    "<div style='text-align:center;color:var(--ink-soft);font-size:0.8rem;'>"
    "Nep-RAG · नेपाली देवनागरी · स्रोत-सहित उत्तर · Source-cited responses"
    "</div>",
    unsafe_allow_html=True,
)
