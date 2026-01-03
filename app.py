# ==========================================================
# MultiCancer Ultimate — FULL APP (Phase 1 + 2 + 3) RAG & UI UPGRADED
# ==========================================================

import os, io, pickle, csv, time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt
import bcrypt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from pdf2image import convert_from_bytes
import pytesseract
import re
import os
from dotenv import load_dotenv
load_dotenv()
from deep_translator import GoogleTranslator
# Translate from English to Urdu
translated_text = GoogleTranslator(source='en', target='ur').translate("Hello")
print(translated_text)

# ---------------- SAFE IMPORTS ----------------
def try_import(name):
    try:
        return __import__(name)
    except:
        return None

torch_mod = try_import("torch")
cv2 = try_import("cv2")
faiss = try_import("faiss") or try_import("faiss_cpu")
pdfplumber = try_import("pdfplumber")
docx_mod = try_import("docx")
openpyxl = try_import("openpyxl")
pytesseract = try_import("pytesseract")
speech_recognition = try_import("speech_recognition")

from sentence_transformers import SentenceTransformer

# ---------------- TTS ----------------
import pyttsx3
from gtts import gTTS
from io import BytesIO

# ---------------- TORCH ----------------
if torch_mod:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

# ---------------- GEMINI INITIALIZATION ----------------
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        st.session_state["gemini_model"] = gemini_model
    except Exception as e:
        gemini_model = None


import threading

# Initialize TTS stop flag
if "tts_stop" not in st.session_state:
    st.session_state.tts_stop = False

# Function to play text with stop control in a separate thread
def speak_text_thread_controlled(text):
    import pyttsx3

    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        if st.session_state.tts_stop:
            engine.stop()

    # Run TTS in a thread to avoid "loop already started"
    t = threading.Thread(target=_speak)
    t.start()

def highlight_keywords(text, query):
    """
    Highlights keywords from the query in the given text.
    Used for showing preview of matched text in RAG answers.
    """
    for word in query.split():
        # Wrap the keyword in ** ** for bold effect in Streamlit
        text = re.sub(f"({re.escape(word)})", r"**\1**", text, flags=re.IGNORECASE)
    return text


# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config("🧬 MultiCancer Ultimate", layout="wide")

# ---------------- LOGGING ----------------
LOG_FILE = "usage_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","user","model","label",
            "confidence","image","query","sources"
        ])

def log_event(user, model=None, label=None, confidence=None, image=None, query=None, sources=None):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            user, model, label, confidence, image, query, sources
        ])

# ---------------- TTS ENGINE ----------------
import threading
from gtts import gTTS
from io import BytesIO
import pyttsx3
import speech_recognition
import streamlit as st

# Initialize pyttsx3 engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

# Thread-safe pyttsx3
def speak_text_thread(text):
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# Thread-safe gTTS
def speak_urdu_thread(text):
    tts = gTTS(text=text, lang="ur")
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    st.audio(fp.read(), format="audio/mp3")

# Voice Input (Speech-to-Text)
def listen_question():
    r = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source:
        st.info("🎙️ Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except Exception as e:
            st.error(f"❌ Could not recognize speech: {e}")
            return ""

# ---------- TRANSLATOR + TTS (Multi-Tenant Cached) ----------
if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}  # {tenant_id: {original_text: translated_text}}

from deep_translator import GoogleTranslator

def get_translated_text(text, tenant_id, target_lang="ur"):
    tenant_cache = st.session_state.translation_cache.setdefault(tenant_id, {})
    if text in tenant_cache:
        return tenant_cache[text]

    translated = GoogleTranslator(source="en", target=target_lang).translate(text)
    tenant_cache[text] = translated
    return translated


def speak_text(text):
    """
    Plays English text using TTS
    """
    speak_text_thread(text)

def speak_urdu(text):
    """
    Plays Urdu text using TTS
    """
    speak_urdu_thread(text)

def handle_answer_tts(ans, lang_choice="English", tenant_id="guest"):
    """
    Handles multi-tenant TTS with translation caching
    """
    if st.session_state.get("tts_stop", False):
        return

    if lang_choice.lower() == "urdu":
        urdu_text = get_translated_text(ans, tenant_id=tenant_id, target_lang="ur")
        st.info("🔊 Playing Urdu audio...")
        speak_urdu(urdu_text)
    else:
        st.info("🔊 Playing English audio...")
        speak_text(ans)

# ================= IMAGE VISUALIZATIONS =================
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb

def generate_original(img: Image.Image) -> Image.Image:
    """Return the original RGB image."""
    return img.convert("RGB")

def generate_saliency(img: Image.Image, model=None) -> Image.Image:
    if model is None:
        return generate_original(img)
    
    model.eval()
    transform = transform_default
    x = transform(img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)
    
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    score = out.max()
    
    # 🔹 Fix: clone to avoid inplace error
    score_clone = score.clone()
    model.zero_grad(set_to_none=True)
    score_clone.backward()
    
    grad = x.grad.abs().mean(dim=1)[0].cpu().numpy()
    grad = (grad - grad.min()) / (grad.max() + 1e-8)
    grad = (grad * 255).astype("uint8")
    
    heatmap = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(heatmap).resize(img.size)
    
    return Image.blend(img.convert("RGB"), heatmap, alpha=0.5)

def generate_edge_highlight(img: Image.Image) -> Image.Image:
    """Highlight edges in the image."""
    img_cv = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edges_img = Image.fromarray(edges_colored).resize(img.size)
    return Image.blend(img.convert("RGB"), edges_img, alpha=0.5)

def generate_superpixel(img: Image.Image, segments=100) -> Image.Image:
    """Generate superpixel visualization using SLIC."""
    img_np = np.array(img.convert("RGB"))
    labels = slic(img_np, n_segments=segments, compactness=10, start_label=1)
    img_label = label2rgb(labels, img_np, kind='avg')
    img_label = (img_label * 255).astype(np.uint8)
    return Image.fromarray(img_label)


# ---------------- PATHS ----------------
MODEL_DIR = Path(os.environ.get("MULTICANCER_MODEL_DIR", r"C:\cancer project\Models"))
RAG_INDEX_FILE = Path("rag_index.faiss")
RAG_META_FILE = Path("rag_meta.pkl")

# ---------------- MODELS ----------------
MODEL_PATHS = {
    "BCN": MODEL_DIR / "bcn_resnet50_best.pth",
    "BRAIN": MODEL_DIR / "best_brain_model3.pth",
    "GASTRIC": MODEL_DIR / "best_gastric_model_optimized.pth",
    "BREAST": MODEL_DIR / "breakhis_resnet50_best.pth",
    "PADUFES": MODEL_DIR / "padufes_resnet50_best.pth",
    "HAM10000": MODEL_DIR / "ham10000_resnet50_best.pth",
    "LUNG": MODEL_DIR / "best_lung_model.pth",
    "OVARIAN": MODEL_DIR / "ovarian_resnet50_best.pth",
    "GENERALIST": MODEL_DIR / "generalist_resnet50_best.pth",
}

# ---------------- LABELS ----------------

GENERALIST_LABELS = [
    'Benign (All Types – Non-Cancerous Tissue with No Malignant Features)',
    'Early Stage Cancer (Localized Malignancy with High Treatment Success)',
    'Pre-Cancer (Abnormal Cellular Changes with Cancer Risk)',
    'Progressive Cancer (Advanced Malignant Disease with Active Spread)',

    'Brain – Glioma (Tumor Originating from Glial Cells)',
    'Brain – Meningioma (Typically Benign Tumor from Brain Coverings)',
    'Brain – Tumor (Abnormal Intracranial Tissue Growth)',

    'Breast – Benign (Non-Cancerous Breast Tissue Changes)',
    'Breast – Malignant (Confirmed Cancerous Breast Tumor)',

    'Cervix – Dyskeratosis (Abnormal Cervical Cell Development)',
    'Cervix – Koilocytosis (HPV-Associated Cellular Changes)',
    'Cervix – Metaplasia (Transformation of Cervical Cell Type)',
    'Cervix – Parabasal (Immature Cervical Epithelial Cells)',
    'Cervix – Superficial (Mature Surface Cervical Cells)',

    'Kidney – Normal (Healthy Renal Tissue)',
    'Kidney – Tumor (Abnormal Renal Mass Detected)',

    'Colon – Adenocarcinoma (Malignant Glandular Colon Cancer)',
    'Colon – Benign (Non-Cancerous Colon Tissue)',

    'Lung – Adenocarcinoma (Malignant Glandular Lung Cancer)',
    'Lung – Benign (Non-Cancerous Lung Tissue)',
    'Lung – Squamous Cell Carcinoma (Aggressive Lung Cancer)',

    'Lymph – CLL (Chronic Lymphocytic Leukemia – Slow Growing)',
    'Lymph – FL (Follicular Lymphoma – Indolent Lymph Cancer)',
    'Lymph – MCL (Mantle Cell Lymphoma – Aggressive Type)',

    'Oral – Normal (Healthy Oral Tissue)',
    'Oral – SCC (Oral Squamous Cell Carcinoma – Malignant)'
]

LABELS = {

    # ================= SKIN =================
    "HAM10000": {
        0: "AKIEC – Actinic Keratosis / Intraepithelial Carcinoma (Pre-Cancerous Skin Lesion)",
        1: "BCC – Basal Cell Carcinoma (Low-Grade Skin Cancer)",
        2: "BKL – Benign Keratosis-Like Lesion (Non-Cancerous Growth)",
        3: "DF – Dermatofibroma (Benign Fibrous Skin Tumor)",
        4: "MEL – Melanoma (Highly Aggressive Skin Cancer)",
        5: "NV – Melanocytic Nevus (Benign Pigmented Mole)",
        6: "SCC – Squamous Cell Carcinoma (Invasive Skin Cancer)",
        7: "VASC – Vascular Lesion (Blood Vessel Abnormality)"
    },

    "BCN": {
        0: "AK – Actinic Keratosis (Sun-Induced Pre-Cancerous Lesion)",
        1: "BCC – Basal Cell Carcinoma (Most Common Skin Cancer)",
        2: "BKL – Benign Keratosis-Like Lesion",
        3: "DF – Dermatofibroma (Benign Skin Tumor)",
        4: "MEL – Melanoma (Life-Threatening Skin Cancer)",
        5: "NV – Melanocytic Nevus (Benign Mole)",
        6: "SCC – Squamous Cell Carcinoma (Aggressive Skin Cancer)",
        7: "VASC – Vascular Skin Lesion"
    },

    "PADUFES": {
        0: "ACK – Actinic Cheilitis / Keratosis (Pre-Cancerous Lip Lesion)",
        1: "BCC – Basal Cell Carcinoma",
        2: "MEL – Melanoma",
        3: "NEV – Nevus (Benign Pigmented Lesion)",
        4: "SCC – Squamous Cell Carcinoma",
        5: "SEK – Seborrheic Keratosis (Benign Skin Growth)"
    },

    # ================= OVARIAN =================
    "OVARIAN": {
        0: "Clear Cell – Clear Cell Ovarian Carcinoma (Aggressive Subtype)",
        1: "Endometrioid – Endometrioid Ovarian Carcinoma",
        2: "Mucinous – Mucinous Ovarian Tumor",
        3: "Non-Cancerous – Normal or Benign Ovarian Tissue",
        4: "Serous – Serous Ovarian Carcinoma (Most Common Type)"
    },

    # ================= GASTRIC =================
    "GASTRIC": {
        0: "Adenocarcinoma – Malignant Gastric Glandular Cancer",
        1: "Debris – Non-Diagnostic Tissue Artifacts",
        2: "Lymphoma – Cancer of Gastric Lymphatic Tissue",
        3: "Mucosa – Stomach Inner Lining Tissue",
        4: "Muscle – Gastric Muscular Layer",
        5: "Normal – Healthy Stomach Tissue",
        6: "Stroma – Supportive Connective Tissue",
        7: "Tumor – Abnormal Gastric Tumor Mass"
    },

    # ================= BRAIN =================
    "BRAIN": {
        0: "No Tumor – Normal Brain Tissue",
        1: "Tumor Present – Brain Tumor Detected"
    },

    # ================= LUNG =================
    "LUNG": {
        0: "Lung Adenocarcinoma – Malignant Lung Glandular Cancer",
        1: "Normal Lung – Healthy Lung Tissue",
        2: "Lung Squamous Cell Carcinoma – Aggressive Lung Cancer"
    },

    # ================= BREAST =================
    "BREAST": {
        "type": {
            0: "Benign – Non-Cancerous Breast Lesion",
            1: "Malignant – Confirmed Breast Cancer"
        },
        "stage": {
            0: "Stage I – Early Localized Cancer",
            1: "Stage II – Regional Spread Detected",
            2: "Stage III – Locally Advanced Cancer",
            3: "Stage IV – Metastatic Breast Cancer"
        },
        "subtype": {
            0: "Adenosis – Benign Glandular Condition",
            1: "Fibroadenoma – Benign Fibrous Tumor",
            2: "Phyllodes Tumor – Rare Fibroepithelial Tumor",
            3: "Tubular Adenoma – Benign Tubular Growth",
            4: "Ductal Carcinoma – Invasive Ductal Breast Cancer",
            5: "Lobular Carcinoma – Invasive Lobular Cancer",
            6: "Mucinous Carcinoma – Mucus-Producing Cancer",
            7: "Papillary Carcinoma – Papillary Breast Cancer"
        }
    },

    # ================= GENERALIST =================
    "GENERALIST": {i: GENERALIST_LABELS[i] for i in range(len(GENERALIST_LABELS))}
}


# ---------------- IMAGE TRANSFORMS ----------------
IMG_SIZE = 224
transform_default = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

IMG_SIZE_300 = 300
transform_300 = transforms.Compose([
    transforms.Resize((IMG_SIZE_300, IMG_SIZE_300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- AUTH ----------------
USERS = {
    "admin": bcrypt.hashpw(b"admin123", bcrypt.gensalt()),
    "researcher": bcrypt.hashpw(b"researcher123", bcrypt.gensalt())
}
ROLES = {"admin":"admin","researcher":"researcher"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = "guest"
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "rag_sessions" not in st.session_state:
    st.session_state.rag_sessions = []
#---------------------------------------------------
from streamlit_lottie import st_lottie
import requests

def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None


# ---------------- LOGIN UI 
def login_ui():
    lottie_animation = load_lottie(
        "https://assets10.lottiefiles.com/packages/lf20_0fhlytwe.json"
    )
    if lottie_animation:
        st_lottie(lottie_animation, height=180, key="login_anim")

    st.title("🔐 Welcome to MultiCancer Ultimate AI")
    username = st.text_input("Username", placeholder="Enter username", key="login_username")
    password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")

    if st.button("Login", key="login_btn"):
        if username in USERS and bcrypt.checkpw(password.encode(), USERS[username]):
            st.session_state.logged_in = True
            st.session_state.user_role = ROLES.get(username, "researcher")
            st.session_state.chat_history = []
            st.success(f"✅ Welcome, {username}!")
            st.stop()
        else:
            st.error("❌ Invalid credentials")



# ---------------- USERS / ROLES ----------------
import bcrypt

USERS = {
    "admin": bcrypt.hashpw(b"admin123", bcrypt.gensalt()),
    "researcher": bcrypt.hashpw(b"researcher123", bcrypt.gensalt())
}
ROLES = {"admin": "admin", "researcher": "researcher"}

# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_model(model_key):
    raw = torch.load(
    MODEL_PATHS[model_key],
    map_location="cpu",
    weights_only=True
    )

    if model_key=="BREAST":
        class BreastNet(nn.Module):
            def __init__(self):
                super().__init__()
                base = models.resnet50(weights=None)
                f = base.fc.in_features
                base.fc = nn.Identity()
                self.backbone = base
                self.head_type = nn.Linear(f,2)
                self.head_stage = nn.Linear(f,4)
                self.head_sub = nn.Linear(f,8)
            def forward(self,x):
                f = self.backbone(x)
                return self.head_type(f), self.head_stage(f), self.head_sub(f)
        model = BreastNet()
    else:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(LABELS[model_key]))
    model.load_state_dict(raw, strict=False)
    model.to(DEVICE).eval()
    return model

# ---------------- PREDICTION ----------------
def predict_image(model, key, img):
    x = transform_300(img).unsqueeze(0).to(DEVICE) if key=="GASTRIC" else transform_default(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        if key=="BREAST":
            t,s,sub = out
            res = {
                "type": LABELS[key]["type"][int(t.argmax())],
                "confidence_type": round(float(F.softmax(t,1)[0][int(t.argmax())])*100,2),
                "stage": LABELS[key]["stage"][int(s.argmax())],
                "confidence_stage": round(float(F.softmax(s,1)[0][int(s.argmax())])*100,2),
                "subtype": LABELS[key]["subtype"][int(sub.argmax())],
                "confidence_subtype": round(float(F.softmax(sub,1)[0][int(sub.argmax())])*100,2)
            }
        else:
            probs = F.softmax(out,1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            res = {
                "label": LABELS[key][idx],
                "confidence": round(float(probs[idx])*100,2)
            }
    return res
# ================= SAAS-GRADE PROFESSIONAL RAG BACKEND =================
# Logic: FAISS + BM25 + Gemini toggle + smart offline summarization + memory + citations + confidence

import os, pickle, time
from datetime import datetime

try:
    import faiss
except:
    import faiss_cpu as faiss

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Optional DOCX support
try:
    from docx import Document
except:
    Document = None

# ================= CONFIG =================
MAX_FILE_MB = 15
MAX_CONTEXT_CHUNKS = 8
MAX_MEMORY = 5
DEFAULT_TOP_K = 5
PERSIST_MEMORY = True

RAG_INDEX_FILE = "saas_rag_indices.pkl"
RAG_META_FILE  = "saas_rag_metas.pkl"
RAG_BM25_FILE  = "saas_rag_bm25.pkl"
RAG_MEMORY_FILE = "saas_rag_memory.pkl"
ANALYTICS_FILE = "saas_rag_analytics.pkl"

# ================= INIT =================
@st.cache_resource(show_spinner=False)
def init_rag():
    embed = SentenceTransformer("all-mpnet-base-v2")
    try:
        return {
            "embed": embed,
            "indices": pickle.load(open(RAG_INDEX_FILE, "rb")),
            "metas": pickle.load(open(RAG_META_FILE, "rb")),
            "bm25": pickle.load(open(RAG_BM25_FILE, "rb")),
            "memory": pickle.load(open(RAG_MEMORY_FILE, "rb")),
            "analytics": pickle.load(open(ANALYTICS_FILE, "rb"))
        }
    except:
        return {"embed": embed, "indices": {}, "metas": {}, "bm25": {}, "memory": {}, "analytics": {}}

RAG = init_rag()

# ================= MEMORY =================
if "rag_memory" not in st.session_state:
    st.session_state.rag_memory = RAG.get("memory", {})

def update_memory(tenant, q, a):
    mem = st.session_state.rag_memory.setdefault(tenant, [])
    mem.append({"q": q, "a": a})
    st.session_state.rag_memory[tenant] = mem[-MAX_MEMORY:]
    if PERSIST_MEMORY:
        pickle.dump(st.session_state.rag_memory, open(RAG_MEMORY_FILE, "wb"))

# ================= SECURITY =================
def is_prompt_injection(q):
    return any(x in q.lower() for x in ["ignore all rules", "system prompt", "bypass", "malicious"])

# ================= GEMINI STATUS =================
def gemini_status():
    model = st.session_state.get("gemini_model")
    if not model:
        return "❌ Gemini client NOT initialized"
    try:
        r = model.generate_content("Reply with OK")
        return "✅ Gemini API ACTIVE" if r and r.text else "⚠️ Gemini reachable but empty response"
    except Exception as e:
        return f"❌ Gemini ERROR: {str(e)}"

# ================= TEXT PROCESSING =================
def chunk_text(text, size=500, overlap=50):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        c = " ".join(words[i:i+size])
        if len(c.strip()) > 30:
            chunks.append(c)
        i += max(1, size - overlap)
    return chunks

def extract_text_with_ocr(file):
    text = ""
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t and len(t.strip()) > 10:
                    text += t + "\n"
    except:
        pass
    if len(text.strip()) < 50:
        try:
            file.seek(0)
            for img in convert_from_bytes(file.read(), dpi=200):
                o = pytesseract.image_to_string(img)
                if o.strip():
                    text += o + "\n"
        except:
            pass
    return text.strip()

# ================= FAISS =================
def build_faiss(vectors):
    dim, n = vectors.shape[1], vectors.shape[0]
    index = faiss.IndexFlatIP(dim) if n < 50 else faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, max(50, n//2))
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    return index

# ================= DOCUMENT INGEST =================
def add_docs_from_file(file):
    if file.size > MAX_FILE_MB * 1024 * 1024:
        st.warning("File too large")
        return
    tenant = st.session_state.get("tenant_id", "guest")
    ext = file.name.split(".")[-1].upper()
    if ext == "PDF":
        raw = extract_text_with_ocr(file)
    elif ext == "TXT":
        raw = file.read().decode("utf-8", errors="ignore")
    elif ext == "CSV":
        raw = "\n".join(" ".join(map(str, r)) for r in pd.read_csv(file).values)
    elif ext == "XLSX":
        raw = "\n".join(" ".join(map(str, r)) for r in pd.read_excel(file).values)
    elif ext == "DOCX" and Document:
        raw = "\n".join(p.text for p in Document(file).paragraphs)
    else:
        st.warning("Unsupported file")
        return

    chunks = chunk_text(raw)
    if not chunks:
        st.warning("No content extracted")
        return

    RAG["indices"].setdefault(tenant, None)
    RAG["metas"].setdefault(tenant, [])
    RAG["bm25"].setdefault(tenant, None)

    metas = [{"text": c, "file": file.name} for c in chunks]
    vecs = RAG["embed"].encode([m["text"] for m in metas], normalize_embeddings=True).astype("float32")
    if RAG["indices"][tenant] is None:
        RAG["indices"][tenant] = build_faiss(vecs)
    else:
        RAG["indices"][tenant].add(vecs)

    RAG["metas"][tenant].extend(metas)
    RAG["bm25"][tenant] = BM25Okapi([m["text"].split() for m in RAG["metas"][tenant]])

    pickle.dump(RAG["indices"], open(RAG_INDEX_FILE, "wb"))
    pickle.dump(RAG["metas"], open(RAG_META_FILE, "wb"))
    pickle.dump(RAG["bm25"], open(RAG_BM25_FILE, "wb"))

# ================= CITATIONS =================
def add_citations(answer, used_chunks):
    cited = []
    for sent in answer.split("."):
        sent = sent.strip()
        if not sent:
            continue
        srcs = set()
        for ch in used_chunks:
            if sent.lower()[:20] in ch["text"].lower():
                srcs.add(ch["file"])
        if srcs:
            cited.append(f"{sent}. ({', '.join(srcs)})")
        else:
            cited.append(f"{sent}.")
    return " ".join(cited)

# ================= CONFIDENCE =================
def confidence_score(chunks, query):
    if not chunks:
        return 0
    hits = 0
    for ch in chunks:
        if any(word.lower() in ch["text"].lower() for word in query.split()):
            hits += 1
    return min(100, int((hits / len(chunks)) * 100))

# ================= PROFESSIONAL STREAM ANSWER =================
def stream_answer(q, top_k=DEFAULT_TOP_K, max_chars=1500):
    start = time.time()
    tenant = st.session_state.get("tenant_id", "guest")

    if is_prompt_injection(q):
        yield "⚠️ Unsafe query blocked."
        return

    if tenant not in RAG["indices"] or RAG["indices"][tenant] is None:
        yield "Upload documents first."
        return

    # FAISS + BM25 retrieval
    q_vec = RAG["embed"].encode([q], normalize_embeddings=True).astype("float32")
    _, I = RAG["indices"][tenant].search(q_vec, top_k)
    bm_scores = RAG["bm25"][tenant].get_scores(q.split())
    bm_top = bm_scores.argsort()[-top_k:][::-1]

    idxs = list(dict.fromkeys(list(I[0]) + list(bm_top)))
    used_chunks = [RAG["metas"][tenant][i] for i in idxs if i < len(RAG["metas"][tenant])]
    context_texts = [ch["text"] for ch in used_chunks]

    if not context_texts:
        yield "Not found in uploaded documents."
        return

    # ✅ Polished professional offline summary
    polished_paragraphs = []
    for chunk in context_texts[:5]:  # top 5 chunks
        lines = [line.strip() for line in chunk.split("\n") if len(line.strip()) > 5]
        paragraph = " ".join(lines)
        polished_paragraphs.append(paragraph)
    polished_answer = "\n\n".join(polished_paragraphs)

    # Limit answer length & add citations
    answer = add_citations(polished_answer, used_chunks)[:max_chars]

    # Evidence & confidence
    agreement = len(set(I[0]) & set(bm_top))
    confidence = (
        "VERY HIGH" if agreement >= 3 else
        "HIGH" if agreement == 2 else
        "MEDIUM" if agreement == 1 else
        "LOW"
    )
    score = confidence_score(used_chunks, q)
    update_memory(tenant, q, answer)

    # Debug info
    with st.expander("🧠 RAG Debug"):
        st.write(f"Chunks used: {len(used_chunks)}")
        st.write(f"Evidence agreement: {agreement}")
        st.write(f"Gemini status: {gemini_status()}")
        st.write(f"ENV Gemini Key Loaded: {bool(os.getenv('GEMINI_API_KEY'))}")
        st.write(f"Answer Confidence: {score}%")
        st.write("Gemini Used:", st.session_state.get("gemini_used", False))

    yield f"""
💡 Answer:
{answer}

📊 Evidence Strength: {confidence}
⏱ Response Time: {round(time.time()-start,2)}s
"""

## ---------------- HOSPITAL ENTERPRISE PDF REPORT ----------------
from reportlab.lib import colors
from reportlab.platypus import (
    Table, TableStyle, Paragraph,
    SimpleDocTemplate, Image as RLImage,
    Spacer, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.graphics.shapes import Drawing, Rect
from datetime import datetime
import tempfile
import uuid
import streamlit as st


# ---------------- CONFIDENCE BAR ----------------
class ConfidenceBar(Flowable):
    def __init__(self, confidence, width=300, height=12):
        super().__init__()
        self.confidence = confidence
        self.width = width
        self.height = height

    def draw(self):
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height)
        self.canv.setFillColor(colors.green if self.confidence >= 85 else colors.orange if self.confidence >= 60 else colors.lightgreen)
        self.canv.rect(0, 0, self.width * (self.confidence / 100), self.height, fill=1)


# ---------------- HEADER / FOOTER ----------------
def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawString(30, 770, "MultiCancer Ultimate — Clinical AI Report")
    canvas.drawRightString(580, 770, f"Page {doc.page}")
    canvas.drawString(30, 20, "For clinical assistance only — Not a diagnostic replacement")
    canvas.restoreState()


def create_pdf_report(results, language="en"):
    pdf_file = "MultiCancer_Hospital_Report.pdf"
    prediction_id = str(uuid.uuid4())[:8]

    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=60,
        bottomMargin=40
    )

    elements = []
    styles = getSampleStyleSheet()
    styleH = styles["Heading1"]
    styleB = styles["Heading2"]
    styleN = styles["Normal"]

    # ---------------- TITLE ----------------
    elements.append(Paragraph("🧬 MultiCancer Ultimate — Clinical AI Prediction Report", styleH))
    elements.append(Paragraph(f"Prediction ID: <b>{prediction_id}</b>", styleN))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styleN
    ))
    elements.append(Spacer(1, 12))

    progress = st.progress(0)
    status = st.empty()
    total = len(results)

    for i, res in enumerate(results):
        status.text(f"Processing {res['image_name']} ({i+1}/{total})")

        pred = res.get("prediction", {})
        model_name = res.get("model", "N/A")
        conf = float(pred.get("confidence", 0))

        # ---------------- RISK LOGIC ----------------
        if conf >= 85:
            risk = "High Risk"
        elif conf >= 60:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # ---------------- SUMMARY TABLE ----------------
        if model_name == "BREAST":
            summary_data = [
                ["Field", "Value"],
                ["Image Name", res["image_name"]],
                ["Tumor Type", pred.get("type", "N/A")],
                ["Type Confidence", f"{pred.get('confidence_type', 'N/A')}%"],
                ["Stage", pred.get("stage", "N/A")],
                ["Subtype", pred.get("subtype", "N/A")],
                ["Risk Level", risk],
                ["Model Used", model_name],
            ]
        else:
            summary_data = [
                ["Field", "Value"],
                ["Image Name", res["image_name"]],
                ["Detected Class", pred.get("label", "N/A")],
                ["Confidence", f"{conf}%"],
                ["Risk Level", risk],
                ["Model Used", model_name],
            ]

        table = Table(summary_data, colWidths=[160, 340])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))

        elements.append(Paragraph("AI Prediction Summary", styleB))
        elements.append(table)
        elements.append(Spacer(1, 8))

        # ---------------- CONFIDENCE BAR ----------------
        elements.append(Paragraph("Prediction Confidence", styleB))
        elements.append(ConfidenceBar(conf))
        elements.append(Spacer(1, 12))

        # ---------------- SAVE IMAGES ----------------
        def save_temp(img):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(tmp.name)
            return tmp.name

        img_paths = []
        img_paths.append(save_temp(res["image_file"]))

        for v in res.get("visuals", {}).values():
            img_paths.append(save_temp(v))

        rows, row = [], []
        for p in img_paths:
            row.append(RLImage(p, width=160, height=160))
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        elements.append(Paragraph("AI Visual Evidence", styleB))
        elements.append(Table(rows))
        elements.append(Spacer(1, 12))

        # ---------------- DOCTOR NOTES ----------------
        elements.append(Paragraph("📝 Clinical Notes (To be completed by physician)", styleB))
        elements.append(Spacer(1, 40))

        # ---------------- SIGNATURE ----------------
        elements.append(Paragraph("👨‍⚕️ Authorized Clinician Signature:", styleB))
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("Name: _____________________   Date: ____________", styleN))

        elements.append(Spacer(1, 30))
        progress.progress((i + 1) / total)

    # ---------------- DISCLAIMER ----------------
    elements.append(Paragraph("⚠️ Medical Disclaimer", styleB))
    elements.append(Paragraph(
        "This AI system assists clinicians and researchers. "
        "It does NOT provide a medical diagnosis and must not "
        "replace professional medical judgment.",
        styleN
    ))

    doc.build(elements, onFirstPage=add_header_footer, onLaterPages=add_header_footer)

    progress.empty()
    status.text("✅ Hospital-grade PDF generated successfully")

    return pdf_file

## ================= STREAMLIT UI =================
if not st.session_state.get("logged_in", False):
    login_ui()
else:
    # ---------------- SIDEBAR ----------------
    st.sidebar.title("🧬 MultiCancer Ultimate")
    st.sidebar.write(
        f"Logged in as: **{st.session_state.get('user_role','User')}**"
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.session_state.chat_history = []
        st.rerun()

    # ------------------ CUSTOM TAB CSS ------------------
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] > div > button[data-tab-active="true"] {
            background-color: #ff4b4b !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 0px !important;
            height: 55px !important;
            padding: 0 25px !important;
            font-size: 16px !important;
        }

        div[data-baseweb="tab-list"] > div > button {
            background-color: #ffe5e5 !important;
            color: #ff4b4b !important;
            border-radius: 0px !important;
            height: 55px !important;
            padding: 0 25px !important;
            font-size: 16px !important;
        }

        div[data-baseweb="tab-list"] > div > button > span {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ------------------ TABS ------------------
    tabs = st.tabs([
        "🖼️ Image Prediction",
        "📚 RAG Q&A",
        "📄 Upload Docs",
        "⚙️ Settings",
        "🧬 About"
    ])

    # ---------- IMAGE PREDICTION ----------

        # ---------- IMAGE PREDICTION ----------
    with tabs[0]:
        st.header("🖼️ Cancer Image Prediction")
        results = []

        uploaded_files = st.file_uploader(
        "Upload image(s) for prediction",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="img_upload"
        )

        model_choice = st.selectbox("Select Model", list(MODEL_PATHS.keys()))

    # ---------- SAFE VISUAL FUNCTION ----------
        def safe_visualize(func, img, model=None):
            try:
               return func(img, model) if model else func(img)
            except Exception as e:
               st.warning(f"Visualization failed: {e}")
               return img

        if uploaded_files and st.button("Predict & Visualize"):
            results = []

        # ---------- LOAD MODEL ----------
            try:
               model = load_model(model_choice)
               st.success(f"✅ Model loaded: {model_choice}")
            except Exception as e:
               st.error(f"❌ Failed to load model {model_choice}: {e}")
               model = None

            for file in uploaded_files:
                try:
                   # ---------- LOAD IMAGE ----------
                   img = Image.open(file).convert("RGB")

                   # ---------- PREDICTION ----------
                   pred = predict_image(model, model_choice, img) if model else {}

                   # ---------- VISUALIZATIONS ----------
                   saliency_img = safe_visualize(generate_saliency, img, model)
                   edge_img = safe_visualize(generate_edge_highlight, img)
                   superpixel_img = safe_visualize(generate_superpixel, img)

                   # ---------- STORE FOR PDF ----------
                   results.append({
                     "image_name": file.name,
                     "image_file": img,
                     "prediction": pred,
                     "model": model_choice,
                     "visuals": {
                        "saliency": saliency_img,
                        "edge": edge_img,
                        "superpixel": superpixel_img
                     }
                   })

                # ==========================================================
                # 🧾 AI DECISION SUMMARY (HOSPITAL STYLE)
                # ==========================================================
                   if model_choice == "BREAST":
                      st.success(f"""
### 🧾 AI Decision Summary — Breast Cancer
• **Image:** {file.name}  
• **Tumor Type:** {pred['type']} ({pred['confidence_type']}%)  
• **Cancer Stage:** {pred['stage']} ({pred['confidence_stage']}%)  
• **Histological Subtype:** {pred['subtype']} ({pred['confidence_subtype']}%)  
• **Model Used:** {model_choice}
""")
                   else:
                      conf = float(pred.get("confidence", 0))

                      if conf >= 85:
                          risk = "🔴 High Risk"
                          reliability = "✔ High Reliability"
                      elif conf >= 60:
                          risk = "🟠 Medium Risk"
                          reliability = "⚠ Medium Reliability"
                      else:
                          risk = "🟢 Low Risk"
                          reliability = "ℹ Low Reliability"
 
                          st.success(f"""
### 🧾 AI Decision Summary
• **Image:** {file.name}  
• **Detected Class:** {pred['label']}  
• **Confidence:** {conf}%  
• **Risk Level:** {risk}  
• **Prediction Reliability:** {reliability}  
• **Model Used:** {model_choice}
""")

                # ==========================================================
                # 🖼️ SIDE-BY-SIDE VIEW
                # ==========================================================
                   col1, col2 = st.columns([1, 1])

                   with col1:
                       st.image(img, caption="🖼️ Original Image", width=320)

                   with col2:
                       st.markdown("### 🧬 Prediction Result")

                       if model_choice == "BREAST":
                           st.metric("Tumor Type", pred["type"])
                           st.metric("Type Confidence", f"{pred['confidence_type']}%")

                           st.metric("Cancer Stage", pred["stage"])
                           st.metric("Stage Confidence", f"{pred['confidence_stage']}%")

                           st.metric("Subtype", pred["subtype"])
                           st.metric("Subtype Confidence", f"{pred['confidence_subtype']}%")
                       else:
                           st.metric("Detected Class", pred["label"])
                           st.metric("Confidence", f"{conf}%")

                           st.markdown("### 🔍 Prediction Confidence")
                           st.progress(conf / 100)

                           if conf >= 85:
                               st.error("🔴 High Confidence Cancer Detection")
                           elif conf >= 60:
                               st.warning("🟠 Medium Confidence – Needs Review")
                           else:
                               st.success("🟢 Low Confidence – Likely Benign")

                # ==========================================================
                # 🧠 VISUAL EVIDENCE
                # ==========================================================
                   st.markdown("### 🧠 AI Visual Evidence")

                   visuals = {
                     "Saliency Heatmap": saliency_img,
                     "Edge Highlight": edge_img,
                     "Superpixel": superpixel_img
                   }

                   st.image(
                     list(visuals.values()),
                     caption=list(visuals.keys()),
                     width=260
                   )

                # ==========================================================
                # 🔍 EXPLAINABLE AI
                # ==========================================================
                   with st.expander("🔍 Why did the AI make this prediction?"):
                      st.markdown("""
- **Saliency Heatmap**: Regions with strongest model attention  
- **Edge Highlight**: Structural boundaries and irregularities  
- **Superpixel**: Texture-based region grouping  

These visuals improve transparency but do not represent a medical diagnosis.
""")

                   st.info("""
⚠️ This AI system assists clinical decision-making.
It does NOT replace professional medical diagnosis.
Always consult a qualified medical expert.
""")

                   buf = BytesIO()
                   saliency_img.save(buf, format="PNG")

                   st.download_button(
                        "⬇️ Download Saliency Visualization",
                        buf.getvalue(),
                        file_name=f"{file.name}_saliency.png",
                        mime="image/png"
                    )

                except Exception as e:
                  st.error(f"❌ Prediction failed for {file.name}: {e}")
                  continue

    st.session_state.last_results = results

# ---------- PDF REPORT ----------
    try:
       if results:
          pdf_file = create_pdf_report(results, language="en")
          with open(pdf_file, "rb") as f:
             st.download_button(
                "📄 Download PDF Report",
                f,
                file_name=pdf_file,
                mime="application/pdf"
             )
    except Exception as e:
          st.error(f"PDF generation failed: {e}")

    # ---------- RAG Q&A SYSTEM ----------
    import uuid
    if "session_uid" not in st.session_state:
        st.session_state.session_uid = str(uuid.uuid4())

    tenant_id = st.session_state.get("tenant_id", "guest")
    user_role = st.session_state.get("user_role", "guest")

    with tabs[1]:
        st.header("💬 RAG Q&A System")

        if tenant_id not in RAG["metas"] or not RAG["metas"][tenant_id]:
            st.warning("📄 Upload documents first.")
        else:
            # Single text input with unique key
            q_input = st.text_input(
                "Ask a question about your data",
                key=f"rag_q_{tenant_id}_{st.session_state.session_uid}"
            )

            # Voice input button
            if st.button(
                "🎤 Speak Question",
                key=f"rag_voice_{tenant_id}_{st.session_state.session_uid}"
            ):
                q_input = listen_question()

            # Language and answer mode selection
            lang_choice = st.radio(
                "Language", ["English", "Urdu"],
                key=f"rag_lang_{tenant_id}_{st.session_state.session_uid}"
            )
            summary_mode = st.radio(
                "Answer Mode", ["Brief", "Detailed"], index=0,
                key=f"rag_mode_{tenant_id}_{st.session_state.session_uid}"
            )

            # Layout columns for answer button and stop audio
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(
                    "Get Answer",
                    key=f"rag_answer_{tenant_id}_{st.session_state.session_uid}"
                ) and q_input.strip():
                    st.session_state.tts_stop = False
                    max_chars = 500 if summary_mode == "Brief" else 2000

                    for ans in stream_answer(
                       q_input,
                       top_k=5,
                       max_chars=max_chars
                    ):
                       st.write(ans)
                       handle_answer_tts(ans, lang_choice=lang_choice, tenant_id=tenant_id)

                       if st.session_state.tts_stop:
                          try:
                            tts_engine.stop()
                          except:
                             pass
                          break


            with col2:
                if st.button(
                    "⏹ Stop Audio",
                    key=f"rag_stop_{tenant_id}_{st.session_state.session_uid}"
                ):
                    st.session_state.tts_stop = True
                    try:
                        tts_engine.stop()
                    except:
                        pass
                    st.success("Audio stopped ✅")

    # ------------------ UPLOAD DOCS ------------------
    with tabs[2]:
        st.header("📂 Upload Documents for RAG")

        uploaded_docs = st.file_uploader(
            "Upload PDF, DOCX, XLSX, CSV, TXT",
            type=["pdf","docx","xlsx","csv","txt"],
            accept_multiple_files=True,
            key=f"doc_upload_{tenant_id}"
        )

        if uploaded_docs and st.button("Add Documents to RAG", key=f"doc_add_{tenant_id}"):
            progress = st.progress(0)
            total = len(uploaded_docs)
            for i, doc in enumerate(uploaded_docs):
                add_docs_from_file(doc)
                handle_answer_tts(f"Document {doc.name} uploaded successfully.", lang_choice="English", tenant_id=tenant_id)
                progress.progress((i+1)/total)

            st.session_state["rag_ready"] = True
            st.success("✅ Documents indexed successfully!")

            st.subheader("📄 Uploaded Documents & Metadata")
            tenant_metas = RAG["metas"].get(tenant_id, [])
            for m in sorted(tenant_metas, key=lambda x: x.get("uploaded", 0), reverse=True):
                st.write(
                    f"{m.get('file','N/A')} | "
                    f"Uploaded: {datetime.utcfromtimestamp(m.get('uploaded',0)).strftime('%Y-%m-%d %H:%M:%S')}"
                    )

    # ---------- SETTINGS ----------
    with tabs[3]:
        st.header("⚙️ Settings")

        api_input = st.text_input("Gemini API Key", GEMINI_API_KEY)

        if st.button("Save API Key"):
           if api_input:
                try:
                  genai.configure(api_key=api_input)
                  st.session_state["gemini_model"] = genai.GenerativeModel("gemini-1.5-flash")
                  st.success("✅ Gemini API key activated.")
                except Exception as e:
                  st.error(f"❌ Gemini error: {e}")



    # ---------- ABOUT PAGE ----------
    with tabs[4]:
        st.markdown("# 🧬 MultiCancer Ultimate AI")

        st.markdown("""
        ## 🔍 Project Overview
        **MultiCancer Ultimate AI** is an advanced research-grade medical artificial intelligence system designed for **multi-organ cancer detection**, **histopathological image analysis**, and **medical knowledge retrieval**.

        The platform combines **deep learning (CNN-based models)** with a **Retrieval-Augmented Generation (RAG) chatbot**, allowing users to:
        - Analyze cancer images
        - Understand predictions with confidence scores
        - Ask document-based medical questions
        - Receive text and voice responses

        ⚠️ This system is intended strictly for **research and educational purposes** and **does not replace professional medical diagnosis**.
        """)

        st.markdown("---")

        st.subheader("🏥 AI Models & Cancer Types (Explained Clearly)")

        with st.expander("🩺 HAM10000 – Skin Cancer Detection"):
            st.write("""
            Trained on dermoscopic skin lesion images.

            **Detected Classes (Full Forms):**
            - **AKIEC** – Actinic Keratoses & Intraepithelial Carcinoma  
            - **BCC** – Basal Cell Carcinoma  
            - **BKL** – Benign Keratosis-like Lesions  
            - **DF** – Dermatofibroma  
            - **MEL** – Melanoma (high-risk malignant skin cancer)  
            - **NV** – Melanocytic Nevi (moles)  
            - **SCC** – Squamous Cell Carcinoma  
            - **VASC** – Vascular Skin Lesions  
            """)

        with st.expander("🧴 BCN – Skin Lesion Classification (Clinical-Level)"):
            st.write("""
            The **BCN model** focuses on **common benign and malignant skin lesion categories**
            encountered in dermatology clinics.

            This model is useful for **early screening**, **academic analysis**, and
            understanding lesion morphology across different skin conditions.

            **Detected Skin Lesion Subtypes:**
            - **Benign Keratosis**
            - **Seborrheic Keratosis**
            - **Actinic Keratosis**
            - **Basal Cell Carcinoma (BCC)**
            - **Squamous Cell Carcinoma (SCC)**
            - **Melanoma**
            - **Nevus (Mole)**
            - **Vascular Lesions**

            🎯 **Use Case:**
            - Dermatology research
            - Early lesion risk assessment
            - Model comparison with HAM10000
            """)

        with st.expander("🧬 PAD-UFES – High-Resolution Skin Lesion Analysis"):
            st.write("""
            The **PAD-UFES model** is trained on **high-quality dermoscopic images**
            with fine-grained annotation, making it suitable for **detailed skin lesion analysis**.

            This model excels at distinguishing visually similar benign and malignant patterns.

            **Detected Skin Lesion Subtypes:**
            - **Melanoma**
            - **Basal Cell Carcinoma (BCC)**
            - **Squamous Cell Carcinoma (SCC)**
            - **Actinic Keratosis**
            - **Seborrheic Keratosis**
            - **Benign Nevus**
            - **Dermatofibroma**
            - **Vascular Lesions**

            🔬 **Strengths:**
            - High-resolution feature learning
            - Reduced false positives
            - Research-grade lesion differentiation
            """)

        with st.expander("🧠 BRAIN – Brain Tumor Detection"):
            st.write("""
            MRI-based deep learning model that detects:
            - **No Tumor** – Normal brain tissue
            - **Tumor Present** – Abnormal tumor growth
            """)

        with st.expander("🫁 LUNG – Lung Cancer Detection"):
            st.write("""
            Classifies lung tissue into:
            - Lung Adenocarcinoma
            - Lung Squamous Cell Carcinoma
            - Normal Lung Tissue
            """)

        with st.expander("🩷 BREAST – Multi-Task Breast Cancer Model"):
            st.write("""
            Advanced **multi-head neural network** predicting three aspects simultaneously:

            **1️⃣ Tumor Type**
            - Benign
            - Malignant

            **2️⃣ Cancer Stage**
            - Stage I (Early)
            - Stage II
            - Stage III
            - Stage IV (Advanced)

            **3️⃣ Histopathological Subtypes**
            - Adenosis
            - Fibroadenoma
            - Phyllodes Tumor
            - Tubular Adenoma
            - Ductal Carcinoma
            - Lobular Carcinoma
            - Mucinous Carcinoma
            - Papillary Carcinoma
            """)

        with st.expander("🍽️ GASTRIC – Gastric Tissue Classification"):
            st.write("""
            Detects multiple gastric tissue components:
            - Adenocarcinoma
            - Lymphoma
            - Tumor
            - Normal Tissue
            - Muscle, Mucosa, Stroma, Debris
            """)

        with st.expander("🧬 OVARIAN – Ovarian Cancer Detection"):
            st.write("""
            Identifies ovarian tissue into:
            - Clear Cell Carcinoma
            - Endometrioid Carcinoma
            - Mucinous Carcinoma
            - Serous Carcinoma
            - Non-Cancerous Tissue
            """)

        with st.expander("🌍 GENERALIST – Multi-Organ & Multi-Subtype Cancer Model"):
            st.write("""
            The **GENERALIST model** is a high-capacity deep learning classifier designed
            for **cross-organ cancer recognition** using histopathological and medical imaging data.

            It enables **broad cancer screening** when the organ-specific model is unknown
            or when mixed datasets are used.

            ### 🧠 Supported Organs & Cancer Subtypes (≈26 Classes)

            **Brain**
            - Glioma
            - Meningioma
            - Pituitary Tumor
            - Normal Brain Tissue

            **Breast**
            - Ductal Carcinoma
            - Lobular Carcinoma
            - Benign Breast Tissue

            **Lung**
            - Lung Adenocarcinoma
            - Lung Squamous Cell Carcinoma
            - Normal Lung Tissue

            **Colon**
            - Colon Adenocarcinoma
            - Benign Colon Tissue

            **Kidney**
            - Renal Cell Carcinoma
            - Papillary Renal Cell Carcinoma
            - Normal Kidney Tissue

            **Cervical**
            - Cervical Squamous Cell Carcinoma
            - Cervical Adenocarcinoma
            - Normal Cervical Tissue

            **Lymphatic System**
            - Lymphoma
            - Normal Lymph Node

            **Oral / Head & Neck**
            - Oral Squamous Cell Carcinoma
            - Normal Oral Tissue

            **Other Tissue Classes**
            - Tumor (Generic)
            - Non-Tumor / Normal Tissue

            ### 🎯 Use Case
            - Cross-organ cancer screening
            - Research & dataset exploration
            - Fallback model when organ-specific models are unavailable
            """)

        st.markdown("---")

        st.subheader("📚 RAG-Powered Medical Chatbot")
        st.write("""
        The chatbot is powered by **Retrieval-Augmented Generation (RAG)**:
        - Medical documents are embedded and indexed using **FAISS**
        - Answers are generated **only from uploaded documents**
        - Prevents hallucinations and ensures factual grounding
        - Supports continuous follow-up questions
        """)

        st.markdown("""
        ### 🤖 Optional Gemini AI Integration
        - Users may optionally connect a **Gemini API key**
        - Enhances reasoning and natural language explanations
        - The system works **fully without Gemini** using pure RAG

        ✔ No API key → Fully functional  
        ✔ API key added → Enhanced intelligence
        """)

        st.markdown("---")

        st.subheader("✨ Key Features")
        st.write("""
        ✔ Multi-organ cancer detection  
        ✔ Deep learning using ResNet-based CNNs  
        ✔ Multi-image upload & analysis  
        ✔ Confidence-based risk alerts  
        ✔ RAG-based medical document Q&A  
        ✔ Text-to-Speech (English & Urdu)  
        ✔ Role-based authentication (Admin / Researcher)  
        ✔ Usage logging & audit trail  
        ✔ PDF medical report generation  
        ✔ Modular, scalable architecture  
        """)

        st.markdown("---")

        st.subheader("🧰 Technology Stack")
        st.write("""
        - **Deep Learning:** PyTorch, TorchVision  
        - **Embeddings:** SentenceTransformers  
        - **Vector Search:** FAISS  
        - **Frontend:** Streamlit  
        - **Optional LLM:** Google Gemini API  
        - **Security:** bcrypt authentication  
        - **Reporting:** PDF generation  
        """)

        st.markdown("---")

        st.subheader("🛡️ AI Safety & Disclaimer")
        st.write("""
        ⚠️ This platform is designed for **research and educational purposes only**.
        It is **not a certified medical diagnostic tool** and must not be used as a substitute
        for professional medical advice or clinical decision-making.
        """)
