import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import tempfile
import random
import json
import requests
from urllib.parse import urlparse

# 🔑 OPENROUTER CLIENT
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "VERISCOPE"
    }
)
def show_veriscope_badge():
    st.markdown("""
    <div style="
        display:flex;
        justify-content:left;
        margin-top:10px;
        margin-bottom:10px;
    ">
        <div style="
            padding:6px 18px;
            border-radius:999px;
            background: rgba(56,189,248,0.12);
            border: 1px solid rgba(56,189,248,0.5);
            color:#7dd3fc;
            font-size:18px;
            font-weight:500;
        ">
            ✦ VERISCOPE · AI Research Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

# 🔥 LLM FUNCTION - Document Q&A
def get_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant.\n"
                "Answer ONLY using the provided context below.\n"
                "Do NOT use any outside knowledge.\n"
                "If the answer is not in the context, say: 'Answer not found in document'.\n"
                "Give detailed answers based strictly on the context."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ],
        temperature=0, stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# 🔥 FAKE NEWS DETECTOR
def detect_fake_news(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a professional fake news detector.\n"
                "Analyze the given news article or text carefully.\n"
                "Respond ONLY in this exact JSON format:\n"
                '{"verdict": "REAL" or "FAKE" or "MISLEADING","confidence": <number between 0 and 100>,'
                '"explanation": "<detailed explanation in 3-4 sentences>",'
                '"red_flags": ["<flag1>", "<flag2>", "<flag3>"],'
                '"positive_signals": ["<signal1>", "<signal2>"]}\n'
                "Do not include any text outside the JSON."
            )},
            {"role": "user", "content": f"Analyze this news:\n\n{text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# 🔥 PLAGIARISM CHECKER
def check_plagiarism(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert plagiarism detection system.\n"
                "Respond ONLY in this exact JSON format with no other text:\n"
                '{"verdict": "ORIGINAL" or "LIKELY PLAGIARIZED" or "POSSIBLY PLAGIARIZED",'
                '"score": <plagiarism percentage 0-100>,'
                '"explanation": "<detailed explanation in 3-4 sentences>",'
                '"flagged_sentences": ["<s1>", "<s2>", "<s3>"],'
                '"reasons": ["<r1>", "<r2>", "<r3>"]}'
            )},
            {"role": "user", "content": f"Check this text for plagiarism:\n\n{text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# 🔥 AI CONTENT DETECTOR
def detect_ai_content(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert AI content detection system.\n"
                "Respond ONLY in this exact JSON format with no other text:\n"
                '{"verdict": "AI GENERATED" or "HUMAN WRITTEN" or "MIXED CONTENT",'
                '"ai_score": <percentage 0-100 likelihood of AI>,'
                '"explanation": "<detailed explanation in 3-4 sentences>",'
                '"reasons": ["<r1>", "<r2>", "<r3>"],'
                '"sentence_breakdown": [{"sentence": "<text>", "label": "AI" or "HUMAN", "confidence": <0-100>}]}\n'
                "Include ALL sentences from the text in sentence_breakdown.\n"
                "Do not include any text outside the JSON."
            )},
            {"role": "user", "content": f"Analyze this text for AI vs human authorship:\n\n{text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# 🔥 LANGUAGE TRANSLATOR
LANGUAGES = [
    "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Marathi", "Gujarati", "Punjabi", "Urdu",
    "French", "Spanish", "German", "Italian", "Portuguese", "Dutch", "Russian", "Chinese (Simplified)",
    "Chinese (Traditional)", "Japanese", "Korean", "Arabic", "Turkish", "Persian", "Swahili",
    "Greek", "Hebrew", "Polish", "Swedish", "Norwegian", "Danish", "Finnish", "Romanian", "Czech",
    "Hungarian", "Thai", "Vietnamese", "Indonesian", "Malay"
]

def translate_text(text, target_language):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"You are an expert translator. Translate the given text accurately into {target_language}.\n"
                "Preserve the original meaning, tone, and formatting.\n"
                "Return ONLY the translated text. No explanations, no notes, no original text."
            )},
            {"role": "user", "content": f"Translate this text to {target_language}:\n\n{text}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# 🔥 URL TRUST CHECKER
def fetch_url_content(url):
    """Fetch page content + metadata from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
    final_url  = resp.url
    status     = resp.status_code
    content_type = resp.headers.get("content-type", "")
    # Try newspaper for clean text
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        page_text = article.text[:4000]
        title     = article.title
    except Exception:
        from bs4 import BeautifulSoup
        soup      = BeautifulSoup(resp.text, "html.parser")
        page_text = soup.get_text(separator=" ", strip=True)[:4000]
        title     = soup.title.string if soup.title else ""

    parsed   = urlparse(final_url)
    domain   = parsed.netloc
    scheme   = parsed.scheme
    has_https = scheme == "https"
    tld       = domain.split(".")[-1] if "." in domain else ""

    return {
        "url": url,
        "final_url": final_url,
        "domain": domain,
        "scheme": scheme,
        "has_https": has_https,
        "tld": tld,
        "status_code": status,
        "title": title,
        "page_text": page_text,
        "redirected": url.rstrip("/") != final_url.rstrip("/"),
    }


def analyze_url_trust(url_data):
    prompt = f"""
URL: {url_data['url']}
Final URL after redirects: {url_data['final_url']}
Domain: {url_data['domain']}
HTTPS: {url_data['has_https']}
TLD: .{url_data['tld']}
HTTP Status: {url_data['status_code']}
Page Title: {url_data['title']}
Redirected: {url_data['redirected']}

Page content (first 3000 chars):
{url_data['page_text']}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert cybersecurity and web trust analyst.\n"
                "Analyze the given URL metadata and page content carefully.\n"
                "Assess trustworthiness based on: HTTPS usage, domain reputation, TLD, content quality, "
                "redirect behavior, presence of suspicious patterns, ad-heavy/spam content, phishing indicators, "
                "legitimacy of the site's purpose, and overall professionalism.\n"
                "Respond ONLY in this exact JSON format with no other text:\n"
                "{\n"
                '  "verdict": "TRUSTED" or "SUSPICIOUS" or "DANGEROUS",\n'
                '  "trust_score": <number 0-100, 100 = fully trusted>,\n'
                '  "what_it_does": "<1-2 sentence description of what this website/page does>",\n'
                '  "domain_info": {\n'
                '    "domain": "<domain name>",\n'
                '    "tld": "<tld>",\n'
                '    "https": <true or false>,\n'
                '    "category": "<e.g. News, E-commerce, Social Media, Government, Education, Unknown, etc.>"\n'
                '  },\n'
                '  "explanation": "<detailed explanation in 3-4 sentences>",\n'
                '  "red_flags": ["<flag1>", "<flag2>", "<flag3>"],\n'
                '  "positive_signals": ["<signal1>", "<signal2>", "<signal3>"]\n'
                "}\n"
                "Do not include any text outside the JSON."
            )},
            {"role": "user", "content": f"Analyze this URL for trustworthiness:\n{prompt}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32}
    )

def create_vectorstore(docs):
    return FAISS.from_documents(docs, load_embeddings())


# ---------------- CONFIG ----------------
st.set_page_config(page_title="VERISCOPE", layout="wide")

for key, val in {
    "page": "home", "source_name": "", "vectorIndex": None,
    "chat_history": [], "dark_mode": True
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------- LANDING PAGE ----------------
if st.session_state.page == "home":
    st.markdown("""
    <style>

    [data-testid="stAppViewContainer"] {
        background: #080c14;
        overflow: hidden;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background:
            radial-gradient(circle at 20% 30%, rgba(56,189,248,0.25), transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(129,140,248,0.25), transparent 40%);
        z-index: 0;
    }

    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(3px 3px at 10% 15%, rgba(255,255,255,0.8) 0%, transparent 100%),
            radial-gradient(3px 2px at 25% 40%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(2.5px 3.5px at 40% 10%, rgba(255,255,255,0.9) 0%, transparent 100%),
            radial-gradient(2px 3px at 55% 25%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(3px 2px at 70% 8%, rgba(255,255,255,0.7) 0%, transparent 100%),
            radial-gradient(3px 2px at 80% 35%, rgba(255,255,255,0.8) 0%, transparent 100%),
            radial-gradient(4px 4px at 90% 18%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(2px 2px at 15% 60%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(3px 4px at 30% 75%, rgba(255,255,255,0.7) 0%, transparent 100%),
            radial-gradient(4px 4px at 50% 55%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(4px 4px at 65% 70%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(5px 5px at 75% 60%, rgba(255,255,255,0.8) 0%, transparent 100%),
            radial-gradient(4px 4px at 88% 75%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(3px 3px at 5% 85%, rgba(255,255,255,0.7) 0%, transparent 100%),
            radial-gradient(3.5px 3.5px at 20% 90%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(4px 4px at 45% 88%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(4px 4px at 60% 92%, rgba(255,255,255,0.7) 0%, transparent 100%),
            radial-gradient(3.5px 3.5px at 85% 88%, rgba(255,255,255,0.8) 0%, transparent 100%),
            radial-gradient(3px 3px at 35% 20%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(3px 3px at 92% 50%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(3px 3px at 8% 45%, rgba(255,255,255,0.7) 0%, transparent 100%),
            radial-gradient(2.5px 2.5px at 48% 72%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(4px 4px at 72% 45%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(3px 3px at 18% 30%, rgba(255,255,255,0.8) 0%, transparent 100%),
            radial-gradient(3px 3px at 95% 30%, rgba(255,255,255,0.5) 0%, transparent 100%);
        z-index: 0;
        pointer-events: none;
        animation: starFloat 8s ease-in-out infinite alternate;
    }

    @keyframes starFloat {
        0%   { opacity: 0.5; transform: translateY(0px) translateX(0px); }
        25%  { opacity: 0.8; transform: translateY(-12px) translateX(6px); }
        50%  { opacity: 1.0; transform: translateY(-20px) translateX(-6px); }
        75%  { opacity: 0.7; transform: translateY(-10px) translateX(10px); }
        100% { opacity: 0.5; transform: translateY(0px) translateX(0px); }
    }

    .floating-box {
        position: fixed;
        border-radius: 12px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        backdrop-filter: blur(4px);
        animation: float 4s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes float {
        0%,100% { transform: translateY(0px); }
        50%      { transform: translateY(-18px); }
    }

    .box1  { width:100px; height:60px; top: 8%;  left: 8%;  animation-delay: 0s;   }
    .box2  { width: 70px; height:44px; top:12%;  left:18%;  animation-delay: 1.2s; }
    .box3  { width: 90px; height:55px; top: 7%;  right: 9%; animation-delay: 0.5s; }
    .box4  { width: 60px; height:38px; top:15%;  right:20%; animation-delay: 2s;   }
    .box5  { width: 80px; height:50px; top:40%;  left: 4%;  animation-delay: 1.5s; }
    .box6  { width: 55px; height:34px; top:47%;  left:14%;  animation-delay: 3s;   }
    .box7  { width: 90px; height:56px; top:38%;  right: 5%; animation-delay: 0.8s; }
    .box8  { width: 60px; height:36px; top:50%;  right:16%; animation-delay: 2.3s; }
    .box9  { width: 45px; height:28px; top:44%;  left:48%;  animation-delay: 1.8s; }
    .box10 { width:100px; height:62px; top:72%;  left: 7%;  animation-delay: 2s;   }
    .box11 { width: 65px; height:40px; top:80%;  left:20%;  animation-delay: 0.4s; }
    .box12 { width: 85px; height:52px; top:75%;  left:42%;  animation-delay: 2.8s; }
    .box13 { width: 55px; height:34px; top:83%;  left:58%;  animation-delay: 1.1s; }
    .box14 { width: 90px; height:56px; top:70%;  right: 8%; animation-delay: 3.5s; }
    .box15 { width: 60px; height:38px; top:82%;  right:22%; animation-delay: 0.9s; }

    .main-title { font-size: 70px; font-weight: 800; text-align: center; margin-top: 40px; }
    .subtitle { text-align: center; color: #64748b; margin-top: 10px;font-size:20px; font-weight:500; }

    .stButton>button {
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        border-radius: 10px;
        color: white;
        border: none;
        padding: 12px 20px;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(56,189,248,0.25);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(56,189,248,0.6);
    }

    .badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        background: rgba(56,189,248,0.12);
        border: 1px solid rgba(56,189,248,0.5);
        color: #7dd3fc;
        font-size: 18px;
        font-weight: 500;
    }
    .badge-wrapper { display: flex; justify-content: center; margin-top: 30px; }

    </style>

    <div class="floating-box box1"></div>
    <div class="floating-box box2"></div>
    <div class="floating-box box3"></div>
    <div class="floating-box box4"></div>
    <div class="floating-box box5"></div>
    <div class="floating-box box6"></div>
    <div class="floating-box box7"></div>
    <div class="floating-box box8"></div>
    <div class="floating-box box9"></div>
    <div class="floating-box box10"></div>
    <div class="floating-box box11"></div>
    <div class="floating-box box12"></div>
    <div class="floating-box box13"></div>
    <div class="floating-box box14"></div>
    <div class="floating-box box15"></div>

    <div class="badge-wrapper">
        <div class="badge">✦ VERISCOPE · AI Research Platform</div>
    </div>

    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='main-title' style="line-height:1.1;">
        <span style="color:white;">Truth starts with</span><br>
        <span style="background: linear-gradient(90deg,#38bdf8,#818cf8,#e879f9);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            knowing what's real
        </span>
    </div>
    <div class='subtitle' style='margin-top: 20px;'>
        All in one platform — Your AI Powered Research Assistant.
    </div>
    """, unsafe_allow_html=True)             
    st.markdown("""
    <style>

    /* cinematic reveal */
    .fade-section {
        opacity: 0;
        transform: translateY(80px) scale(0.96);
        filter: blur(10px);
        animation: cinematicReveal 1.2s cubic-bezier(0.22,1,0.36,1) forwards;
    }

    /* animation */
    @keyframes cinematicReveal {
        0% {
            opacity: 0;
            transform: translateY(80px) scale(0.96);
            filter: blur(10px);
        }
        60% {
            opacity: 1;
            transform: translateY(-5px) scale(1.02);
            filter: blur(2px);
        }
        100% {
            opacity: 1;
            transform: translateY(0px) scale(1);
            filter: blur(0);
        }
    }

    /* stagger timing (VERY IMPORTANT) */
    .delay-1 { animation-delay: 0.2s; }
    .delay-2 { animation-delay: 0.4s; }
    .delay-3 { animation-delay: 0.6s; }
    .delay-4 { animation-delay: 0.8s; }
    .delay-5 { animation-delay: 1s; }
    .delay-6 { animation-delay: 1.2s; }

    /* grid */
    .tools-grid {
        display:grid;
        grid-template-columns: repeat(3, 1fr);
        gap:30px;
        margin-top:70px;
    }

    /* cards */
    .card {
        padding:22px;
        border-radius:22px;
        background: rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        text-align:center;
        transition: all 0.35s ease;
        position:relative;
    }

    /* hover glow */
    .card:hover {
        transform: translateY(-8px) scale(1.04);
        box-shadow: 0 0 50px rgba(56,189,248,0.35);
        border-color: rgba(56,189,248,0.6);
    }

    /* glow pulse */
    .card::before {
        content:"";
        position:absolute;
        inset:-2px;
        border-radius:22px;
        background: linear-gradient(120deg,#38bdf8,#818cf8,#e879f9);
        opacity:0;
        transition:0.4s;
        z-index:-1;
    }

    .card:hover::before {
        opacity:0.3;
    }

    /* text */
    .title { color:white; font-weight:600; margin-top:8px; }
    .desc { color:#94a3b8; font-size:13px; margin-top:4px; }

    </style>

    <div class="tools-grid">

    <div class="card fade-section delay-1">📄<div class="title">Document Q&A</div><div class="desc">Chat with PDFs</div></div>
    <div class="card fade-section delay-2">🚨<div class="title">Fake News</div><div class="desc">Detect misinformation</div></div>
    <div class="card fade-section delay-3">🔍<div class="title">Plagiarism</div><div class="desc">Check originality</div></div>
    <div class="card fade-section delay-4">🤖<div class="title">AI Detector</div><div class="desc">Detect AI content</div></div>
    <div class="card fade-section delay-5">🌐<div class="title">Translator</div><div class="desc">Translate text</div></div>
    <div class="card fade-section delay-6">🔗<div class="title">URL Checker</div><div class="desc">Check safety</div></div>

    </div>
    """, unsafe_allow_html=True)
            
    st.markdown("""
    <div style="text-align:center; margin-top:60px;">
        <div style="
            font-size:26px;
            font-weight:700;
            color:white;
        ">
            ✨ Explore What Veriscope Can Do
        </div>
        <div style="
            font-size:16px;
            color:#94a3b8;
            margin-top:6px;
        ">
            Powerful AI tools — all in one platform
        </div>
    </div>
           
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
        if st.button("Try VERISCOPE 🚀", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()




# ---------------- DASHBOARD PAGE ----------------
# ---------------- DASHBOARD ----------------
if st.session_state.page == "dashboard":

    st.markdown("""
    <style>

    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
    }

    .floating-box {
        display: none !important;
    }

    .block-container {
        padding-top: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* CARD */
    .card {
        background: #1e293b;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        height: 190px;
        transition: all 0.3s ease;
        overflow: hidden;
    }

    .card:hover {
        transform: translateY(-6px) scale(1.02);
        background: #243044;
        box-shadow: 0 0 25px rgba(56,189,248,0.25);
        border-color: rgba(56,189,248,0.6);
    }

    /* FIX — icon box must NOT stretch */
    .c-icon {
        width: 46px !important;
        height: 46px !important;
        min-width: 46px !important;
        border-radius: 12px;
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
        font-size: 20px;
    }

    .c-title {
        font-size: 15px;
        font-weight: 600;
        color: white;
    }

    .c-desc {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Try → button */
    div.stButton > button {
        background: transparent;
        border: none;
        color: #4ade80;
        font-weight: 600;
        text-align: left;
        font-size: 13px;
        margin-top: 8px;
        padding: 0;
    }

    div.stButton > button:hover {
        color: #86efac;
        background: transparent;
        border: none;
        box-shadow: none;
        transform: translateX(4px);
    }

    </style>
    """, unsafe_allow_html=True)

    # TOP BAR (LOGO LEFT + HOME RIGHT)
    col1, col2 = st.columns([6,1])

    show_veriscope_badge()

    with col2:
        if st.button("← Home"):
            st.session_state.page = "home"
            st.rerun()

# TITLE BELOW
    st.markdown("""
    <div style="
        font-size:28px;
        font-weight:700;
        color:white;
        margin-top:8px;
    ">
        What do you want to verify today?
    </div>
    """, unsafe_allow_html=True)

# SUBTEXT
    st.markdown("""
    <div style="
        font-size:18px;
        color:#94a3b8;
        margin-top:5px;
    ">
    Choose a tool to get started — all AI-powered, all instant.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # FIRST ROW
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class='card fade-row1'>
        <div class='c-icon' style='background:rgba(59,130,246,0.15)'>📄</div>
        <div class='c-title'>Document Q&A</div>
        <div class='c-desc'>Upload PDFs or paste a URL and ask anything</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Try →", key="doc"):
            st.session_state.page = "tool"
            st.rerun()

    with col2:
        st.markdown("""<div class='card fade-row1'>
        <div class='c-icon' style='background:rgba(234,179,8,0.15)'>🚨</div>
        <div class='c-title'>Fake News Detector</div>
        <div class='c-desc'>Verify if a news article is credible or misleading</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Try →", key="fake"):
            st.session_state.page = "fake_news"
            st.rerun()

    with col3:
        st.markdown("""<div class='card fade-row1'>
        <div class='c-icon' style='background:rgba(239,68,68,0.15)'>🔍</div>
        <div class='c-title'>Plagiarism Checker</div>
        <div class='c-desc'>Detect copied or unoriginal content in any text</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Try →", key="plag"):
            st.session_state.page = "plagiarism"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # SECOND ROW
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""<div class='card fade-row1'>
        <div class='c-icon' style='background:rgba(34,197,94,0.15)'>🤖</div>
        <div class='c-title'>AI Content Detector</div>
        <div class='c-desc'>Find out if text was written by AI or a human</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Try →", key="ai"):
            st.session_state.page = "ai_detector"
            st.rerun()

    with col5:
        st.markdown("""<div class='card fade-row1'>
        <div class='c-icon' style='background:rgba(139,92,246,0.15)'>🌐</div>
        <div class='c-title'>Language Translator</div>
        <div class='c-desc'>Translate any document or text to any language</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Try →", key="lang"):
            st.session_state.page = "translator"
            st.rerun()

    with col6:
        st.markdown("""<div class='card fade-row1'>
        <div class='c-icon' style='background:rgba(249,115,22,0.15)'>🔗</div>
        <div class='c-title'>URL Trust Checker</div>
        <div class='c-desc'>Check if a website or link is safe and trustworthy</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Try →", key="url"):
            st.session_state.page = "url_checker"
            st.rerun()


# ---------------- URL TRUST CHECKER PAGE ----------------
if st.session_state.page == "url_checker":
    show_veriscope_badge()
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stMarkdownContainer"] p { color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔗 URL Trust Checker")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = "dashboard"; st.rerun()

    st.markdown("### Paste any URL to check if it's safe and trustworthy")
    st.markdown("---")

    url_input = st.text_input("🔗 Paste URL here", placeholder="https://example.com")
    check_btn = st.button("🔗 Check URL Trust", use_container_width=True)

    if check_btn:
        if not url_input.strip():
            st.warning("⚠️ Please paste a URL to check."); st.stop()
        if not url_input.startswith("http"):
            st.warning("⚠️ URL must start with http:// or https://"); st.stop()

        with st.spinner("🌐 Fetching page content..."):
            try:
                url_data = fetch_url_content(url_input)
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to this URL. It may be offline or blocked."); st.stop()
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The website took too long to respond."); st.stop()
            except Exception as e:
                st.error(f"❌ Could not fetch URL: {e}"); st.stop()

        with st.spinner("🧠 Analyzing trust & safety..."):
            try:
                raw    = analyze_url_trust(url_data)
                clean  = raw.strip().replace("```json","").replace("```","").strip()
                result = json.loads(clean)

                verdict        = result.get("verdict", "UNKNOWN")
                trust_score    = result.get("trust_score", 0)
                what_it_does   = result.get("what_it_does", "")
                domain_info    = result.get("domain_info", {})
                explanation    = result.get("explanation", "")
                red_flags      = result.get("red_flags", [])
                positive_sigs  = result.get("positive_signals", [])

                st.markdown("<br>", unsafe_allow_html=True)

                if verdict == "TRUSTED":
                    color = "#4ade80"; bg = "rgba(34,197,94,0.1)"; icon = "✅"
                elif verdict == "DANGEROUS":
                    color = "#f87171"; bg = "rgba(239,68,68,0.1)"; icon = "🚫"
                else:
                    color = "#fbbf24"; bg = "rgba(251,191,36,0.1)"; icon = "⚠️"

                # ── VERDICT + SCORE ──
                st.markdown(f"""
                <div style='background:{bg}; border:1px solid {color}40; border-radius:16px; padding:1.5rem; margin-bottom:1.2rem;'>
                    <div style='display:flex; align-items:center; gap:12px; margin-bottom:14px;'>
                        <span style='font-size:36px;'>{icon}</span>
                        <div>
                            <div style='font-size:24px; font-weight:700; color:{color};'>{verdict}</div>
                            <div style='font-size:13px; color:#94a3b8; margin-top:2px;'>{url_data["domain"]}</div>
                        </div>
                    </div>
                    <div style='font-size:13px; color:#94a3b8; margin-bottom:6px;'>Trust Score</div>
                    <div style='background:rgba(255,255,255,0.08); border-radius:99px; height:12px; margin-bottom:6px;'>
                        <div style='background:{color}; width:{trust_score}%; height:12px; border-radius:99px;'></div>
                    </div>
                    <span style='font-size:22px; font-weight:700; color:{color};'>{trust_score}/100</span>
                </div>
                """, unsafe_allow_html=True)

                # ── WHAT THIS SITE DOES ──
                st.markdown(f"""
                <div style='background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;'>
                    <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:6px;'>🌐 What this site does</div>
                    <div style='font-size:14px; color:#e6edf3; line-height:1.7;'>{what_it_does}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── DOMAIN INFO CARDS ──
                https_color  = "#4ade80" if domain_info.get("https") else "#f87171"
                https_label  = "✅ HTTPS" if domain_info.get("https") else "❌ No HTTPS"
                st.markdown(f"""
                <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:1.2rem;'>
                    <div style='background:#1e293b; border-radius:10px; padding:1rem; text-align:center;'>
                        <div style='font-size:11px; color:#64748b; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;'>Domain</div>
                        <div style='font-size:13px; font-weight:600; color:#e2e8f0;'>{domain_info.get("domain","—")}</div>
                    </div>
                    <div style='background:#1e293b; border-radius:10px; padding:1rem; text-align:center;'>
                        <div style='font-size:11px; color:#64748b; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;'>TLD</div>
                        <div style='font-size:13px; font-weight:600; color:#e2e8f0;'>.{domain_info.get("tld","—")}</div>
                    </div>
                    <div style='background:#1e293b; border-radius:10px; padding:1rem; text-align:center;'>
                        <div style='font-size:11px; color:#64748b; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;'>Security</div>
                        <div style='font-size:13px; font-weight:600; color:{https_color};'>{https_label}</div>
                    </div>
                    <div style='background:#1e293b; border-radius:10px; padding:1rem; text-align:center;'>
                        <div style='font-size:11px; color:#64748b; margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;'>Category</div>
                        <div style='font-size:13px; font-weight:600; color:#e2e8f0;'>{domain_info.get("category","—")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── EXPLANATION ──
                st.markdown(f"""
                <div style='background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;'>
                    <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:6px;'>📋 Analysis</div>
                    <div style='font-size:14px; color:#e6edf3; line-height:1.7;'>{explanation}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── RED FLAGS + POSITIVE SIGNALS ──
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("""<div style='background:#1e293b; border-radius:12px; padding:1.2rem;'>
                        <div style='font-size:13px; font-weight:600; color:#f87171; margin-bottom:10px;'>🚩 Red Flags</div>""", unsafe_allow_html=True)
                    if red_flags:
                        for flag in red_flags:
                            st.markdown(f"""
                            <div style='display:flex; gap:8px; margin-bottom:8px; align-items:flex-start;'>
                                <span style='color:#f87171; font-size:14px; margin-top:1px;'>•</span>
                                <span style='font-size:13px; color:#e6edf3; line-height:1.6;'>{flag}</span>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size:13px; color:#484f58;'>No red flags found.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_b:
                    st.markdown("""<div style='background:#1e293b; border-radius:12px; padding:1.2rem;'>
                        <div style='font-size:13px; font-weight:600; color:#4ade80; margin-bottom:10px;'>✅ Positive Signals</div>""", unsafe_allow_html=True)
                    if positive_sigs:
                        for sig in positive_sigs:
                            st.markdown(f"""
                            <div style='display:flex; gap:8px; margin-bottom:8px; align-items:flex-start;'>
                                <span style='color:#4ade80; font-size:14px; margin-top:1px;'>•</span>
                                <span style='font-size:13px; color:#e6edf3; line-height:1.6;'>{sig}</span>
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size:13px; color:#484f58;'>No positive signals found.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── REDIRECT WARNING ──
                if url_data["redirected"]:
                    st.markdown(f"""
                    <div style='background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3);
                                border-radius:10px; padding:10px 14px; margin-top:1rem;
                                font-size:13px; color:#fbbf24;'>
                        ⚠️ This URL redirected to: <strong>{url_data["final_url"]}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

# ---------------- LANGUAGE TRANSLATOR PAGE ----------------
if st.session_state.page == "translator":
    show_veriscope_badge()
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stMarkdownContainer"] p { color: white; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌐 Language Translator")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = "dashboard"; st.rerun()

    st.markdown("### Translate text or a document to any language")
    st.markdown("---")

    target_lang = st.selectbox("🌍 Select target language", LANGUAGES, index=0)
    tab1, tab2  = st.tabs(["📝 Translate Text", "📄 Translate Document"])

    with tab1:
        input_text = st.text_area("Paste your text here", height=220, placeholder="Type or paste the text you want to translate...", key="trans_text")
        word_count = len(input_text.split()) if input_text.strip() else 0
        st.caption(f"Word count: {word_count}")
        if st.button("🌐 Translate Text", use_container_width=True, key="btn_trans_text"):
            if not input_text.strip():
                st.warning("⚠️ Please paste some text to translate.")
            else:
                with st.spinner(f"🌐 Translating to {target_lang}..."):
                    try:
                        translated = translate_text(input_text, target_lang)
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_orig, col_trans = st.columns(2)
                        with col_orig:
                            st.markdown(f"""<div style='background:#1e293b; border-radius:12px; padding:1.2rem; height:100%;'>
                                <div style='font-size:12px; font-weight:600; color:#64748b; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px;'>📄 Original Text</div>
                                <div style='font-size:14px; color:#e2e8f0; line-height:1.8; white-space:pre-wrap;'>{input_text}</div></div>""", unsafe_allow_html=True)
                        with col_trans:
                            st.markdown(f"""<div style='background:#1e293b; border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:1.2rem; height:100%;'>
                                <div style='font-size:12px; font-weight:600; color:#a78bfa; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px;'>🌐 Translated — {target_lang}</div>
                                <div style='font-size:14px; color:#e2e8f0; line-height:1.8; white-space:pre-wrap;'>{translated}</div></div>""", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button("📥 Download Translation",
                            data=f"ORIGINAL:\n{input_text}\n\n{'─'*60}\n\nTRANSLATED ({target_lang}):\n{translated}",
                            file_name=f"translation_{target_lang.lower().replace(' ','_')}.txt", mime="text/plain", use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Translation failed: {e}")

    with tab2:
        uploaded_pdf = st.file_uploader("Upload PDF to translate", type=["pdf"], key="trans_pdf")
        doc_url      = st.text_input("🔗 Or paste a URL to translate", key="trans_url")
        if st.button("🌐 Translate Document", use_container_width=True, key="btn_trans_doc"):
            if not uploaded_pdf and not doc_url.strip():
                st.warning("⚠️ Please upload a PDF or paste a URL.")
            else:
                with st.spinner("📖 Extracting document content..."):
                    try:
                        raw_text = ""
                        if doc_url.strip() and doc_url.startswith("http"):
                            from newspaper import Article
                            article = Article(doc_url); article.download(); article.parse()
                            raw_text = article.text
                            if not raw_text.strip(): st.error("❌ Could not extract content."); st.stop()
                        elif uploaded_pdf:
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                            tmp.write(uploaded_pdf.read()); tmp.flush()
                            loader = PyMuPDFLoader(tmp.name); pages = loader.load()
                            raw_text = "\n\n".join([p.page_content for p in pages[:30]])
                        if not raw_text.strip(): st.error("❌ Could not extract text."); st.stop()
                        words = raw_text.split()
                        if len(words) > 3000:
                            raw_text = " ".join(words[:3000])
                            st.info("ℹ️ Document truncated to 3000 words for translation.")
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {e}"); st.stop()

                with st.spinner(f"🌐 Translating to {target_lang}..."):
                    try:
                        translated = translate_text(raw_text, target_lang)
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_orig, col_trans = st.columns(2)
                        with col_orig:
                            st.markdown(f"""<div style='background:#1e293b; border-radius:12px; padding:1.2rem;'>
                                <div style='font-size:12px; font-weight:600; color:#64748b; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px;'>📄 Original Document</div>
                                <div style='font-size:13px; color:#e2e8f0; line-height:1.8; max-height:420px; overflow-y:auto; white-space:pre-wrap;'>{raw_text}</div></div>""", unsafe_allow_html=True)
                        with col_trans:
                            st.markdown(f"""<div style='background:#1e293b; border:1px solid rgba(139,92,246,0.3); border-radius:12px; padding:1.2rem;'>
                                <div style='font-size:12px; font-weight:600; color:#a78bfa; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px;'>🌐 Translated — {target_lang}</div>
                                <div style='font-size:13px; color:#e2e8f0; line-height:1.8; max-height:420px; overflow-y:auto; white-space:pre-wrap;'>{translated}</div></div>""", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button("📥 Download Translation",
                            data=f"ORIGINAL:\n{raw_text}\n\n{'─'*60}\n\nTRANSLATED ({target_lang}):\n{translated}",
                            file_name=f"translation_{target_lang.lower().replace(' ','_')}.txt", mime="text/plain", use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Translation failed: {e}")


# ---------------- AI CONTENT DETECTOR PAGE ----------------
if st.session_state.page == "ai_detector":
    show_veriscope_badge()
    st.markdown("""<style>[data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stMarkdownContainer"] p { color: white; }</style>""", unsafe_allow_html=True)
    st.title("🤖 AI Content Detector")
    if st.button("⬅️ Back to Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    st.markdown("### Paste your text below to detect if it was written by AI or a human")
    st.markdown("---")
    input_text = st.text_area("📝 Paste your text here", height=250, placeholder="Paste the text you want to analyze...")
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"Word count: {word_count}")
    if st.button("🤖 Detect AI Content", use_container_width=True):
        if not input_text.strip(): st.warning("⚠️ Please paste some text to analyze."); st.stop()
        if word_count < 20: st.warning("⚠️ Please enter at least 20 words."); st.stop()
        with st.spinner("🧠 Analyzing text for AI patterns..."):
            try:
                raw=detect_ai_content(input_text); clean=raw.strip().replace("```json","").replace("```","").strip()
                result=json.loads(clean)
                verdict=result.get("verdict","UNKNOWN"); ai_score=result.get("ai_score",0); human_score=100-ai_score
                explanation=result.get("explanation",""); reasons=result.get("reasons",[]); breakdown=result.get("sentence_breakdown",[])
                st.markdown("<br>", unsafe_allow_html=True)
                if verdict=="AI GENERATED": color="#f87171"; bg="rgba(239,68,68,0.1)"; icon="🤖"
                elif verdict=="HUMAN WRITTEN": color="#4ade80"; bg="rgba(34,197,94,0.1)"; icon="✍️"
                else: color="#fbbf24"; bg="rgba(251,191,36,0.1)"; icon="⚖️"
                st.markdown(f"""<div style='background:{bg}; border:1px solid {color}40; border-radius:16px; padding:1.5rem; margin-bottom:1.2rem;'>
                    <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px;'><span style='font-size:32px;'>{icon}</span>
                    <span style='font-size:24px; font-weight:700; color:{color};'>{verdict}</span></div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:16px;'>
                    <div><div style='font-size:12px; color:#94a3b8; margin-bottom:5px;'>🤖 AI Score</div>
                    <div style='background:rgba(255,255,255,0.08); border-radius:99px; height:10px; margin-bottom:5px;'>
                    <div style='background:#f87171; width:{ai_score}%; height:10px; border-radius:99px;'></div></div>
                    <span style='font-size:20px; font-weight:700; color:#f87171;'>{ai_score}%</span></div>
                    <div><div style='font-size:12px; color:#94a3b8; margin-bottom:5px;'>✍️ Human Score</div>
                    <div style='background:rgba(255,255,255,0.08); border-radius:99px; height:10px; margin-bottom:5px;'>
                    <div style='background:#4ade80; width:{human_score}%; height:10px; border-radius:99px;'></div></div>
                    <span style='font-size:20px; font-weight:700; color:#4ade80;'>{human_score}%</span></div></div></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style='background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;'>
                    <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:6px;'>📋 Analysis</div>
                    <div style='font-size:14px; color:#e6edf3; line-height:1.7;'>{explanation}</div></div>""", unsafe_allow_html=True)
                if reasons:
                    st.markdown("""<div style='background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;'>
                        <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:10px;'>🔎 Key Indicators</div>""", unsafe_allow_html=True)
                    for r in reasons:
                        st.markdown(f"""<div style='display:flex; gap:8px; margin-bottom:10px;'><span style='color:#fbbf24;'>•</span>
                        <span style='font-size:13px; color:#e6edf3; line-height:1.6;'>{r}</span></div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                if breakdown:
                    st.markdown("""<div style='background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1rem;'>
                        <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:14px;'>📝 Sentence-Level Breakdown
                        <span style='font-size:11px; font-weight:400; margin-left:10px;'><span style='color:#f87171;'>■</span> AI &nbsp;<span style='color:#4ade80;'>■</span> Human</span></div>""", unsafe_allow_html=True)
                    for item in breakdown:
                        sentence=item.get("sentence",""); label=item.get("label","HUMAN"); confidence=item.get("confidence",50)
                        if label=="AI": s_color="#f87171"; s_bg="rgba(239,68,68,0.08)"; s_border="#f87171"; s_badge="🤖 AI"; s_badge_bg="rgba(239,68,68,0.2)"
                        else: s_color="#4ade80"; s_bg="rgba(34,197,94,0.06)"; s_border="#4ade80"; s_badge="✍️ Human"; s_badge_bg="rgba(34,197,94,0.2)"
                        st.markdown(f"""<div style='background:{s_bg}; border-left:3px solid {s_border}; border-radius:6px; padding:10px 14px; margin-bottom:8px;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                            <span style='background:{s_badge_bg}; color:{s_color}; font-size:11px; font-weight:600; padding:2px 8px; border-radius:99px;'>{s_badge}</span>
                            <span style='font-size:11px; color:#64748b;'>{confidence}% confidence</span></div>
                            <div style='font-size:13px; color:#e6edf3; line-height:1.6;'>{sentence}</div></div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")


# ---------------- PLAGIARISM CHECKER PAGE ----------------
if st.session_state.page == "plagiarism":
    show_veriscope_badge()
    st.markdown("""<style>[data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stMarkdownContainer"] p { color: white; }</style>""", unsafe_allow_html=True)
    st.title("🔍 Plagiarism Checker")
    if st.button("⬅️ Back to Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    st.markdown("### Paste your text below to check for plagiarism")
    st.markdown("---")
    input_text = st.text_area("📝 Paste your text here", height=250, placeholder="Paste the text you want to check for plagiarism...")
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"Word count: {word_count}")
    if st.button("🔍 Check Plagiarism", use_container_width=True):
        if not input_text.strip(): st.warning("⚠️ Please paste some text to check."); st.stop()
        if word_count < 20: st.warning("⚠️ Please enter at least 20 words."); st.stop()
        with st.spinner("🧠 Analyzing text for plagiarism..."):
            try:
                raw=check_plagiarism(input_text); clean=raw.strip().replace("```json","").replace("```","").strip()
                result=json.loads(clean)
                verdict=result.get("verdict","UNKNOWN"); score=result.get("score",0)
                explanation=result.get("explanation",""); flagged=result.get("flagged_sentences",[]); reasons=result.get("reasons",[])
                st.markdown("<br>", unsafe_allow_html=True)
                if verdict=="ORIGINAL": color="#4ade80"; bg="rgba(34,197,94,0.1)"; icon="✅"
                elif verdict=="LIKELY PLAGIARIZED": color="#f87171"; bg="rgba(239,68,68,0.1)"; icon="❌"
                else: color="#fbbf24"; bg="rgba(251,191,36,0.1)"; icon="⚠️"
                st.markdown(f"""<div style='background:{bg}; border:1px solid {color}40; border-radius:16px; padding:1.5rem; margin-bottom:1.2rem;'>
                    <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'><span style='font-size:32px;'>{icon}</span>
                    <span style='font-size:24px; font-weight:700; color:{color};'>{verdict}</span></div>
                    <div style='font-size:13px; color:#94a3b8; margin-bottom:6px;'>Plagiarism Score</div>
                    <div style='background:rgba(255,255,255,0.08); border-radius:99px; height:12px; margin-bottom:6px;'>
                    <div style='background:{color}; width:{score}%; height:12px; border-radius:99px;'></div></div>
                    <span style='font-size:22px; font-weight:700; color:{color};'>{score}%</span>
                    <span style='font-size:13px; color:#94a3b8; margin-left:8px;'>plagiarism detected</span></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style='background:#1e293b; border-radius:12px; padding:1.2rem; margin-bottom:1.2rem;'>
                    <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:6px;'>📋 Analysis</div>
                    <div style='font-size:14px; color:#e6edf3; line-height:1.7;'>{explanation}</div></div>""", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("""<div style='background:#1e293b; border-radius:12px; padding:1.2rem;'>
                        <div style='font-size:13px; font-weight:600; color:#f87171; margin-bottom:10px;'>🚩 Flagged Sentences</div>""", unsafe_allow_html=True)
                    if flagged:
                        for i,s in enumerate(flagged,1):
                            st.markdown(f"""<div style='background:rgba(239,68,68,0.08); border-left:3px solid #f87171;
                                border-radius:6px; padding:10px 12px; margin-bottom:8px; font-size:13px; color:#e6edf3; line-height:1.6;'>
                                <span style='color:#f87171; font-weight:600; font-size:11px;'>#{i}</span><br>{s}</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size:13px; color:#484f58;'>No sentences flagged.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown("""<div style='background:#1e293b; border-radius:12px; padding:1.2rem;'>
                        <div style='font-size:13px; font-weight:600; color:#94a3b8; margin-bottom:10px;'>🔎 Why it was flagged</div>""", unsafe_allow_html=True)
                    if reasons:
                        for r in reasons:
                            st.markdown(f"""<div style='display:flex; gap:8px; margin-bottom:10px;'><span style='color:#fbbf24;'>•</span>
                            <span style='font-size:13px; color:#e6edf3; line-height:1.6;'>{r}</span></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-size:13px; color:#484f58;'>No specific reasons found.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")


# ---------------- FAKE NEWS DETECTOR PAGE ----------------
if st.session_state.page == "fake_news":
    show_veriscope_badge()
    st.markdown("""<style>[data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stMarkdownContainer"] p { color: white; }</style>""", unsafe_allow_html=True)
    col_title, col_back = st.columns([10, 1])
    with col_title: st.title("🚨 Fake News Detector")
    with col_back: st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Back to Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    st.markdown("### Paste a news URL or type/paste the news text below")
    st.markdown("---")
    news_url  = st.text_input("🔗 Paste news article URL (optional)")
    news_text = st.text_area("📝 Or paste news text directly", height=200, placeholder="Paste the news article text here...")
    analyze   = st.button("🔍 Analyze News", use_container_width=True)
    if analyze:
        final_text = ""
        if news_url and news_url.startswith("http"):
            with st.spinner("📖 Fetching article from URL..."):
                try:
                    from newspaper import Article
                    article=Article(news_url); article.download(); article.parse()
                    final_text=article.text
                    if not final_text.strip(): st.error("❌ Could not extract content. Try pasting the text directly."); st.stop()
                except Exception as e: st.error(f"❌ Could not fetch URL: {e}"); st.stop()
        elif news_text.strip(): final_text=news_text.strip()
        else: st.warning("⚠️ Please paste a URL or enter news text."); st.stop()
        with st.spinner("🧠 Analyzing news credibility..."):
            try:
                raw=detect_fake_news(final_text); clean=raw.strip().replace("```json","").replace("```","").strip()
                result=json.loads(clean)
                verdict=result.get("verdict","UNKNOWN"); confidence=result.get("confidence",0)
                explanation=result.get("explanation",""); red_flags=result.get("red_flags",[]); positive_signals=result.get("positive_signals",[])
                st.markdown("<br>", unsafe_allow_html=True)
                if verdict=="REAL": color="#4ade80"; bg="rgba(34,197,94,0.1)"; icon="✅"
                elif verdict=="FAKE": color="#f87171"; bg="rgba(239,68,68,0.1)"; icon="❌"
                else: color="#fbbf24"; bg="rgba(251,191,36,0.1)"; icon="⚠️"
                st.markdown(f"""<div style='background:{bg}; border:1px solid {color}40; border-radius:16px; padding:1.5rem; margin-bottom:1rem;'>
                    <div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'><span style='font-size:32px;'>{icon}</span>
                    <span style='font-size:28px; font-weight:700; color:{color};'>{verdict}</span></div>
                    <span style='font-size:13px; color:#94a3b8;'>Confidence Score</span>
                    <div style='background:rgba(255,255,255,0.08); border-radius:99px; height:10px; margin-top:6px;'>
                    <div style='background:{color}; width:{confidence}%; height:10px; border-radius:99px;'></div></div>
                    <span style='font-size:13px; color:{color}; font-weight:600;'>{confidence}%</span></div>""", unsafe_allow_html=True)
                st.markdown(f"**📋 Explanation:** {explanation}")
                col_a,col_b=st.columns(2)
                with col_a:
                    if red_flags:
                        st.markdown("**🚩 Red Flags:**")
                        for flag in red_flags: st.markdown(f"- {flag}")
                with col_b:
                    if positive_signals:
                        st.markdown("**✅ Positive Signals:**")
                        for sig in positive_signals: st.markdown(f"- {sig}")
            except Exception as e: st.error(f"❌ Analysis failed: {e}")


# ---------------- TOOL PAGE ----------------
if st.session_state.page == "tool":
    show_veriscope_badge()
    
    st.markdown("""
    <div style="
        font-size:30px;
        font-weight:700;
        color:white;
        margin-top:5px;
        margin-bottom:px;
    ">
    📄 Document Q&A
    </div>
    """, unsafe_allow_html=True)
    col_title,col_toggle=st.columns([10,1])
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🌙" if st.session_state.dark_mode else "☀️"):
            st.session_state.dark_mode=not st.session_state.dark_mode; st.rerun()
    bg_color="#0f172a" if st.session_state.dark_mode else "#f8fafc"
    text_color="white" if st.session_state.dark_mode else "#0f172a"
    st.markdown(f"""<style>[data-testid="stAppViewContainer"]{{background-color:{bg_color};color:{text_color};}}
    [data-testid="stMarkdownContainer"] p{{color:{text_color};}}</style>""", unsafe_allow_html=True)
    if st.button("⬅️ Back to Dashboard"): st.session_state.page="dashboard"; st.rerun()
    st.markdown("### 📂 Upload PDF or paste URL")
    uploaded_file=st.file_uploader("Upload PDF (Max 3 PDFs)",type=["pdf"],accept_multiple_files=True)
    if uploaded_file and len(uploaded_file)>3: st.warning("⚠️ Maximum 3 PDFs allowed."); st.stop()
    if uploaded_file:
        for f in uploaded_file:
            if f.size>10*1024*1024: st.warning(f"⚠️ '{f.name}' is too large."); st.stop()
    url=st.text_input("🔗 Paste URL here"); process=st.button("Process")
    if process:
        st.session_state.vectorIndex=None; st.session_state.chat_history=[]
        if not url and not uploaded_file: st.warning("⚠️ Provide URL or upload file"); st.stop()
        with st.spinner(random.choice(["📖 Reading your document...","🔍 Extracting content...","🧠 Building knowledge base...","⚡ Almost ready..."])):
            try:
                data=[]
                if url:
                    if not url.startswith("http"): st.warning("⚠️ URL must start with http://"); st.stop()
                    from newspaper import Article
                    article=Article(url); article.download(); article.parse()
                    text=article.text
                    if not text.strip(): st.error("❌ Could not extract content."); st.stop()
                    data.append(Document(page_content=text,metadata={"source":url}))
                if uploaded_file:
                    for file in uploaded_file:
                        temp=tempfile.NamedTemporaryFile(delete=False); temp.write(file.read())
                        loader=PyMuPDFLoader(temp.name); pages=loader.load(); data.extend(pages[:50])
                splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
                docs=splitter.split_documents(data)
                st.session_state.vectorIndex=create_vectorstore(docs); st.session_state.chat_history=[]
                st.session_state.source_name=url if url else ", ".join([f.name for f in uploaded_file])
                st.session_state.page="chat"; st.rerun()
            except Exception as e:
                error=str(e)
                if "404" in error: st.error("❌ URL not found.")
                elif "403" in error or "401" in error: st.error("❌ Website blocked or requires login.")
                elif "timeout" in error.lower(): st.error("❌ Request timed out.")
                elif "list index out of range" in error: st.error("❌ Could not extract text from this PDF.")
                elif "no such file" in error.lower(): st.error("❌ File not found.")
                else: st.error(f"❌ Something went wrong: {error}")
                st.stop()


# ---------------- CHAT PAGE ----------------
if st.session_state.page == "chat":
    show_veriscope_badge()

    # Dark/Light toggle
    col_title, col_toggle = st.columns([10, 1])
    with col_toggle:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🌙" if st.session_state.dark_mode else "☀️"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # Apply theme
    if st.session_state.dark_mode:
        bg_color = "#0f172a"
        text_color = "white"
    else:
        bg_color = "#f8fafc"
        text_color = "#0f172a"

    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        color: {text_color};
    }}
    [data-testid="stMarkdownContainer"] p {{
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Split screen
    left, right = st.columns([1, 2])

    with left:
        st.markdown("### 📄 Source")
        for source in st.session_state.source_name.split(", "):
            st.info(source)
        st.markdown("---")
        if st.button("⬅️ Upload New Document", use_container_width=True):
            st.session_state.page = "tool"
            st.session_state.vectorIndex = None
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.vectorIndex = None
            st.session_state.chat_history = []
            st.rerun()
    with right:
        st.markdown("### 💬 Chat")

        # Show chat history
        i = 0
        while i < len(st.session_state.chat_history):
            chat = st.session_state.chat_history[i]
            if chat["role"] == "user":
                col1, col2 = st.columns([11, 1])
                with col1:
                    with st.chat_message("user"):
                        st.write(chat["content"])
                with col2:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.chat_history.pop(i)
                        if i < len(st.session_state.chat_history):
                            st.session_state.chat_history.pop(i)
                        st.rerun()
            else:
                with st.chat_message("assistant"):
                    st.write(chat["content"])
            i += 1

        # Chat input
        # Chat input OUTSIDE columns
        question = st.chat_input("Ask anything...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            # Show question immediately
            with st.chat_message("user"):
                st.write(question)

            docs = st.session_state.vectorIndex.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])

            if not context.strip() or len(context) < 100:
                full_answer = "⚠️ Not enough data found. Try another question.\n\n⚠️ Not found in document"
                with st.chat_message("assistant"):
                    st.write(full_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
                st.rerun()
            else:
                with st.chat_message("assistant"):
                    streamed = st.write_stream(get_answer(question, context))
                indicator = "⚠️ Not found in document" if "Answer not found in document" in streamed else "✅ Found in document"
                full_answer = f"{streamed}\n\n{indicator}"
                st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
                st.rerun()
