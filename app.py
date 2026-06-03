import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import json
import os
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import torch
from zoneinfo import ZoneInfo

# ==========================================
# KONFIGURASI PERSISTENCE (PENYIMPANAN DATA)
# ==========================================
HISTORY_FILE = "data_history.json"

def load_history_from_file():
    """Memuat riwayat dari file JSON jika tersedia"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history_to_file(history):
    """Menyimpan riwayat ke file JSON"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA MODERN
# ==========================================
st.set_page_config(
    page_title="✨ Nerkor",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ner-korupsi',
        'Report a bug': 'https://github.com/yourusername/ner-korupsi/issues',
        'About': "# ✨ NER Korupsi Indonesia v3.0\nAplikasi Named Entity Recognition dengan animasi modern untuk ekstraksi entitas berita korupsi."
    }
)

# ==========================================
# 2. CUSTOM CSS MODERN + ANIMATIONS + GLASSMORPHISM
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 30%, #f093fb 70%, #f5576c 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', 'Poppins', sans-serif;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(135deg, rgba(245,247,250,0.95) 0%, rgba(195,207,226,0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(102,126,234,0.95) 0%, rgba(118,75,162,0.95) 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.8rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 1.8rem 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin: 1rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: cardEntrance 0.5s ease-out both;
    }
    
    @keyframes cardEntrance {
        from { opacity: 0; transform: scale(0.95) translateY(20px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: countUp 1s ease-out;
    }
    
    @keyframes countUp {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .entity-box {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.2rem 1rem;
        border-radius: 16px;
        margin: 0.6rem 0;
        border-left: 4px solid transparent;
        border-image: linear-gradient(180deg, #667eea, #764ba2) 1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInLeft 0.4s ease-out both;
        position: relative;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .entity-box:nth-child(2) { animation-delay: 0.1s; }
    .entity-box:nth-child(3) { animation-delay: 0.2s; }
    .entity-box:nth-child(4) { animation-delay: 0.3s; }
    .entity-box:nth-child(5) { animation-delay: 0.4s; }
    
    .entity-box:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25);
        border-left-width: 6px;
    }
    
    .entity-name {
        font-weight: 600;
        color: #2d3748;
        font-size: 1.15rem;
        display: inline-block;
        transition: color 0.3s ease;
    }
    
    .entity-box:hover .entity-name {
        color: #667eea;
    }
    
    .entity-type {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.8rem;
        border-radius: 25px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.6rem;
        animation: bounceIn 0.5s ease-out both;
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .person-tag {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #1565c0;
        box-shadow: 0 2px 8px rgba(21, 101, 192, 0.15);
    }
    
    .org-tag {
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
        color: #6a1b9a;
        box-shadow: 0 2px 8px rgba(106, 27, 154, 0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        color: white !important;
        border: none;
        padding: 0.9rem 2.2rem;
        border-radius: 14px;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background-position: 100% 100%;
    }
    
    .stTextArea textarea, .stTextInput input {
        border-radius: 16px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        font-size: 15px !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
        transform: scale(1.01);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,249,250,0.98) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.5);
        padding: 8px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(227,242,253,0.9), rgba(187,222,251,0.9));
        padding: 1.2rem;
        border-radius: 16px;
        border-left: 5px solid #2196f3;
        margin: 1.2rem 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .loading-dots {
        display: inline-flex;
        gap: 4px;
    }
    
    .loading-dots span {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite both;
    }
    
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
    
    .floating-icon {
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .pulse-badge {
        animation: pulseBadge 2s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes pulseBadge {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }
    
    .url-input {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.3rem;
        border-radius: 16px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .url-input:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 14px !important;
        border-left: 5px solid !important;
        animation: slideInRight 0.4s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .app-footer {
        text-align: center;
        padding: 2.5rem 2rem 1.5rem;
        color: rgba(102, 102, 102, 0.8);
        font-size: 0.95rem;
        animation: fadeIn 1s ease-out 0.5s both;
    }
    
    .app-footer p { margin: 0.3rem 0; }
    
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .main-header p { font-size: 1rem; }
        .metric-value { font-size: 2.2rem; }
        .metric-card { padding: 1.2rem 1rem; }
    }
    
    html { scroll-behavior: smooth; }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.3); border-radius: 10px; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. INISIALISASI SESSION STATE & LOAD DATA
# ==========================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = load_history_from_file()
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# ==========================================
# 4. MEMUAT MODEL CUSTOM NER (TANPA FALLBACK)
# ==========================================
@st.cache_resource
def load_custom_model():
    """
    Load model custom langsung dari root folder 'models'
    """
    # Mengarah langsung ke folder "models" utama tempat file berada
    custom_model_path = "models"
    
    # Validasi: Folder model harus ada
    if not os.path.exists(custom_model_path):
        st.error(f"""
         **Model Custom Tidak Ditemukan!**
        
        Path yang dicari: `{custom_model_path}`
        
        **Solusi:**
        1. Pastikan folder `models/` ada di direktori utama repositori Anda.
        2. Pastikan file model utama sudah berada langsung di dalam folder models/
        """)
        st.stop()
    
    # Validasi: File konfigurasi model harus ada di folder models/
    required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(custom_model_path, f))]
    
    if missing_files:
        st.error(f"""
         **File Model Tidak Lengkap!**
        
        File yang hilang di folder `models/`: {', '.join(missing_files)}
        
        **Solusi:**
        1. Pastikan semua file penunjang model (config, tokenizer) sudah dipindahkan ke folder models/.
        """)
        st.stop()
    
    try:
        with st.spinner(f" Memuat model custom dari: {custom_model_path}"):
            tokenizer = AutoTokenizer.from_pretrained(custom_model_path)
            model = AutoModelForTokenClassification.from_pretrained(custom_model_path)
            
            # Buat pipeline NER
            nlp_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1 # Jalankan di CPU
            )
            return nlp_pipeline
            
    except Exception as e:
        st.error(f"""
         **Gagal Memuat Model Custom!**
         Error: {str(e)}
        """)
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Inisialisasi variabel global dengan nilai awal None agar mencegah NameError
nlp_ner = None

# Jalankan pemuatan model saat aplikasi pertama kali dimuat
with st.spinner(" Memuat model custom NER... Mohon tunggu..."):
    try:
        nlp_ner = load_custom_model()
    except Exception as e:
        st.error(f"Gagal memanggil fungsi pemuatan model: {e}")

# Simpan status model di session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = True

# ==========================================
# 5. FUNGSI HELPER
# ==========================================
def truncate_text_for_bert(text, max_words=350):
    """Truncate teks agar aman untuk model BERT (max 512 tokens)"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + "..."

def extract_article_from_url(url):
    """Ekstrak konten artikel dari URL berita"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe']):
            element.decompose()
        
        title = soup.find('h1') or soup.find('title')
        title_text = title.get_text().strip() if title else "Artikel Tanpa Judul"
        
        content_selectors = [
            {'tag': 'article', 'class': None},
            {'tag': 'div', 'class': 'article-content'},
            {'tag': 'div', 'class': 'post-content'},
            {'tag': 'div', 'class': 'entry-content'},
            {'tag': 'div', 'id': 'article-body'},
            {'tag': 'div', 'class': 'detail__body'},
            {'tag': 'div', 'class': 'read__content'},
        ]
        
        content = ""
        for selector in content_selectors:
            element = soup.find(selector['tag'], class_=selector['class']) if selector['class'] else soup.find(selector['tag'])
            if element:
                paragraphs = element.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                break
        
        if not content.strip():
            paragraphs = soup.find('body').find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
        
        content = re.sub(r'\s+', ' ', content).strip()
        content = truncate_text_for_bert(content, max_words=350)
        
        return title_text, content
        
    except requests.exceptions.RequestException as e:
        return None, f"❌ Error koneksi: {str(e)}"
    except Exception as e:
        return None, f"❌ Error parsing: {str(e)}"

def extract_entities(text):
    """Ekstraksi entitas dari teks menggunakan custom model"""
    if nlp_ner is None:
        st.error("❌ Model tidak tersedia!")
        return {}, {}, []
    
    try:
        results = nlp_ner(text)
        
        individuals = {}
        organizations = {}
        
        for entity in results:
            word = entity['word'].strip()
            entity_group = entity.get('entity_group', '')
            score = float(entity['score'])
            
            # Mapping label sesuai output custom model Anda
            # Sesuaikan dengan label yang digunakan saat training
            if entity_group in ['PER', 'PERSON', 'per', 'person', 'I-PER', 'B-PER']:
                if word and word not in individuals:
                    individuals[word] = score
                elif word and score > individuals.get(word, 0):
                    individuals[word] = score
            
            elif entity_group in ['ORG', 'ORGANIZATION', 'org', 'organization', 'I-ORG', 'B-ORG']:
                if word and word not in organizations:
                    organizations[word] = score
                elif word and score > organizations.get(word, 0):
                    organizations[word] = score
        
        return individuals, organizations, results
        
    except Exception as e:
        st.error(f"❌ Error saat ekstraksi: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return {}, {}, []

def create_visualization(individuals, organizations):
    """Buat visualisasi interaktif dengan animasi Plotly"""
    all_entities, all_scores, all_types = [], [], []
    
    for name, score in individuals.items():
        all_entities.append(name)
        all_scores.append(score)
        all_types.append('Individu')
    
    for name, score in organizations.items():
        all_entities.append(name)
        all_scores.append(score)
        all_types.append('Instansi')
    
    if not all_entities:
        return None, None
    
    fig_bar = px.bar(
        x=all_entities, y=all_scores, color=all_types,
        title='📊 Confidence Score Entitas',
        labels={'x': 'Entitas', 'y': 'Confidence Score'},
        color_discrete_map={'Individu': '#667eea', 'Instansi': '#764ba2'},
        height=400
    )
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True, hovermode='x unified',
        font=dict(family='Inter, sans-serif'),
        transition=dict(duration=500, easing='cubic-in-out')
    )
    
    fig_pie = px.pie(
        values=[len(individuals), len(organizations)],
        names=['Individu', 'Instansi'],
        title='🥧 Distribusi Entitas',
        color_discrete_map={'Individu': '#667eea', 'Instansi': '#764ba2'},
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=400, font=dict(family='Inter, sans-serif'))
    
    return fig_bar, fig_pie

def export_to_csv(individuals, organizations, timestamp, source="Manual"):
    """Export hasil ke CSV"""
    data = []
    for name, score in individuals.items():
        data.append({'Timestamp': timestamp, 'Source': source, 'Entitas': name, 'Tipe': 'PERSON', 'Confidence Score': score})
    for name, score in organizations.items():
        data.append({'Timestamp': timestamp, 'Source': source, 'Entitas': name, 'Tipe': 'ORGANIZATION', 'Confidence Score': score})
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8-sig')

# ==========================================
# 6. SIDEBAR DENGAN INFO MODEL CUSTOM
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 1rem;'>
        <div class='floating-icon' style='font-size: 3.5rem;'>🇮🇩</div>
        <h3 style='color: #667eea; margin: 0.8rem 0; font-weight: 700;'>✨ Nerkor</h3>
        <p style='font-size: 0.85rem; color: #666;'>Advanced NLP System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    menu = st.radio(
        "📋 Menu Navigasi",
        ["🏠 Beranda", "📊 Analisis", "📜 Riwayat", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    if len(st.session_state.analysis_history) > 0:
        st.markdown("### 📈 Statistik Cepat")
        total_analisis = len(st.session_state.analysis_history)
        st.metric("🔍 Total Analisis", total_analisis, delta=f"+{total_analisis} hari ini")
    
    st.divider()
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem; font-size: 0.8rem; color: #888;'>
        <p>© 2026 - Kelompok 4</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. KONTEN UTAMA DENGAN ANIMASI
# ==========================================

if menu == "🏠 Beranda":
    st.markdown("""
    <div class='main-header'>
        <h1>🇮🇩 Ekstraksi Entitas Berita Korupsi</h1>
        <p>Named Entity Recognition dengan Custom Model IndoBERT untuk Analisis Cerdas Berita Korupsi Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='animation-delay: 0.1s;'>
            <div style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🤖</div>
            <h3 style='color: #667eea; margin: 0.5rem 0;'>Custom AI Model</h3>
            <p style='color: #666; line-height: 1.5;'>Model IndoBERT yang telah di-fine-tune khusus untuk domain berita korupsi Indonesia</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='animation-delay: 0.2s;'>
            <div style='font-size: 3.5rem; margin-bottom: 0.5rem;'>🌐</div>
            <h3 style='color: #667eea; margin: 0.5rem 0;'>URL Support</h3>
            <p style='color: #666; line-height: 1.5;'>Analisis langsung dari link berita online Kompas, Detik, Tempo, dan lainnya</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card' style='animation-delay: 0.3s;'>
            <div style='font-size: 3.5rem; margin-bottom: 0.5rem;'>📊</div>
            <h3 style='color: #667eea; margin: 0.5rem 0;'>Visualisasi</h3>
            <p style='color: #666; line-height: 1.5;'>Grafik interaktif dan export data CSV untuk analisis lanjutan</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📖 Panduan Penggunaan")
    
    steps = [
        ("1️⃣", "Input Teks/URL", "Tempel teks berita korupsi ATAU masukkan link artikel dari situs berita terpercaya"),
        ("2️⃣", "Klik Analisis", "Tekan tombol '🚀 Analisis' untuk memulai proses ekstraksi entitas dengan custom model"),
        ("3️⃣", "Lihat Hasil", "Periksa daftar individu dan instansi yang terdeteksi beserta confidence score"),
        ("4️⃣", "Export Data", "Unduh hasil analisis dalam format CSV untuk keperluan dokumentasi")
    ]
    
    for i, (icon, title, desc) in enumerate(steps):
        col_a, col_b = st.columns([1, 5])
        with col_a:
            st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{icon}</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.7); padding: 1rem 1.2rem; border-radius: 12px; border-left: 4px solid #667eea;'>
                <strong style='color: #667eea;'>{title}</strong><br>
                <span style='color: #555; font-size: 0.95rem;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("### 💡 Contoh Input")
    
    with st.expander("📝 Klik untuk melihat contoh teks & URL"):
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("**📰 Teks Manual:**")
            st.code("""
KPK menjadwalkan pemeriksaan terhadap mantan pejabat 
Kementerian Keuangan, Rafael Alun Trisambodo, terkait 
dugaan korupsi pajak. Dalam kasus ini, Direktorat Jenderal 
Pajak diduga menerima suap dari beberapa pengusaha...
            """, language="text")
        with col_ex2:
            st.markdown("**🔗 URL Berita:**")
            st.code("""
https://www.kompas.com/nasional/read/2024/...
https://www.detik.com/berita/d-7123456/...
https://nasional.tempo.co/read/1234567/...
            """, language="text")

elif menu == "📊 Analisis":
    st.markdown("""
    <div class='main-header'>
        <h1>🔍 Analisis Teks Berita</h1>
        <p>Ekstraksi entitas individu dan instansi dengan custom model NLP</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_manual, tab_url = st.tabs(["✍️ Input Manual", "🌐 Dari URL"])
    
    individuals, organizations = {}, {}
    timestamp = ""
    source_type = "Manual"
    text_preview = ""
    analysis_done = False
    
    with tab_manual:
        teks_berita = st.text_area(
            label="📄 Teks Berita Korupsi:",
            height=300,
            placeholder="Contoh: KPK menjadwalkan pemeriksaan terhadap mantan pejabat...",
            key="manual_text"
        )
        
        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            show_confidence = st.checkbox("Tampilkan Score", value=True, key="chk_manual")
            auto_visualize = st.checkbox("Auto Visualisasi", value=True, key="viz_manual")
        with col_opt2:
            analyze_manual = st.button("🚀 Analisis Teks", type="primary", use_container_width=True)
        
        if analyze_manual and teks_berita.strip():
            # REVISI: Mengunci zona waktu ke WIB (Asia/Jakarta)
            timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

            source_type = "Manual"
            text_preview = teks_berita[:100] + "..." if len(teks_berita) > 100 else teks_berita
            
            with st.spinner("🔄 Custom model sedang menganalisis teks..."):
                try:
                    teks_processed = truncate_text_for_bert(teks_berita, max_words=350)
                    individuals, organizations, raw_results = extract_entities(teks_processed)
                    
                    if not individuals and not organizations:
                        st.warning("""
                        ⚠️ **Tidak ada entitas terdeteksi.**
                        
                        Kemungkinan penyebab:
                        - Teks tidak mengandung nama orang/instansi yang jelas
                        - Format teks berbeda dengan data training
                        - Model memerlukan teks yang lebih panjang/kontekstual
                        
                        💡 **Tips:** Gunakan teks berita formal dengan nama lengkap dan instansi yang jelas.
                        """)
                    
                    analysis_done = True
                    
                    st.session_state.current_results = {
                        'timestamp': timestamp, 'source': source_type, 'text': teks_berita,
                        'individuals': individuals, 'organizations': organizations
                    }
                    
                    st.session_state.analysis_history.append({
                        'timestamp': timestamp, 'source': source_type,
                        'individuals': individuals, 'organizations': organizations,
                        'individuals_count': len(individuals), 'organizations_count': len(organizations),
                        'text_preview': text_preview
                    })
                    
                    save_history_to_file(st.session_state.analysis_history)
                    
                except Exception as e:
                    st.error(f"❌ Error saat analisis: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        elif analyze_manual and not teks_berita.strip():
            st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu!")
    
    with tab_url:
        st.markdown("""
        <div class='url-input'>
            <strong>🔗 Masukkan Link Berita:</strong><br>
            <small>Mendukung: Kompas, Detik, Tempo, CNN Indonesia, dan situs berita umum</small>
        </div>
        """, unsafe_allow_html=True)
        
        url_input = st.text_input(
            label="URL Artikel Berita:",
            placeholder="https://www.kompas.com/nasional/read/2024/...",
            key="url_input"
        )
        
        col_opt3, col_opt4 = st.columns([2, 1])
        with col_opt3:
            show_confidence_url = st.checkbox("Tampilkan Score", value=True, key="chk_url")
            auto_visualize_url = st.checkbox("Auto Visualisasi", value=True, key="viz_url")
        with col_opt4:
            analyze_url = st.button("🌐 Analisis URL", type="primary", use_container_width=True)
        
        if analyze_url and url_input.strip():
            # REVISI: Mengunci zona waktu ke WIB (Asia/Jakarta)
            timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")

            source_type = "URL"
            
            with st.spinner("🔄 Mengambil dan menganalisis artikel dari URL..."):
                try:
                    title, content = extract_article_from_url(url_input)
                    
                    if content and "Error" not in content and len(content.strip()) > 50:
                        text_preview = f"[{title}] {content[:50]}..."
                        
                        individuals, organizations, raw_results = extract_entities(content)
                        
                        if not individuals and not organizations:
                            st.warning("""
                            ⚠️ **Tidak ada entitas terdeteksi dari URL.**
                            
                            Kemungkinan penyebab:
                            - Konten artikel tidak mengandung nama jelas
                            - Struktur HTML artikel tidak didukung scraper
                            - Model memerlukan konteks yang lebih spesifik
                            """)
                        
                        analysis_done = True
                        
                        st.session_state.current_results = {
                            'timestamp': timestamp, 'source': source_type, 'text': content,
                            'individuals': individuals, 'organizations': organizations
                        }
                        
                        st.session_state.analysis_history.append({
                            'timestamp': timestamp, 'source': source_type,
                            'individuals': individuals, 'organizations': organizations,
                            'individuals_count': len(individuals), 'organizations_count': len(organizations),
                            'text_preview': text_preview
                        })
                        
                        save_history_to_file(st.session_state.analysis_history)
                        
                        with st.expander("📰 Preview Artikel yang Diekstrak"):
                            st.markdown(f"**Judul:** {title}")
                            st.markdown(f"**Panjang:** {len(content)} karakter")
                            st.markdown(f"**Preview:** {content[:300]}...")
                            
                    elif "Error" in content:
                        st.error(content)
                    else:
                        st.warning("⚠️ Konten artikel terlalu pendek atau tidak valid.")
                        
                except Exception as e:
                    st.error(f"❌ Error saat analisis URL: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        elif analyze_url and not url_input.strip():
            st.warning("⚠️ Silakan masukkan URL terlebih dahulu!")
    
    if analysis_done:
        st.success("✨ Analisis Berhasil! Data telah disimpan ke riwayat.")
        st.divider()
        
        # Debug info - opsional, bisa di-collapse
        with st.expander("🔧 Debug Info", expanded=False):
            st.write(f"**Sumber:** {source_type}")
            st.write(f"**Jumlah Individu:** {len(individuals)}")
            st.write(f"**Jumlah Instansi:** {len(organizations)}")
            if raw_results:
                st.write(f"**Total entities dari model:** {len(raw_results)}")
                st.write("**Sample raw output:**", raw_results[:3] if len(raw_results) >= 3 else raw_results)
            if individuals:
                st.write("**Individu terdeteksi:**", list(individuals.keys())[:5])
            if organizations:
                st.write("**Instansi terdeteksi:**", list(organizations.keys())[:5])
        
        st.markdown("### 📊 Ringkasan Hasil")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(individuals)}</div>
                <div class='metric-label'>👤 Individu</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(organizations)}</div>
                <div class='metric-label'>🏢 Instansi</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat3:
            total = len(individuals) + len(organizations)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{total}</div>
                <div class='metric-label'>🎯 Total Entitas</div>
            </div>
            """, unsafe_allow_html=True)
        
        show_conf = show_confidence if source_type == "Manual" else show_confidence_url
        auto_viz = auto_visualize if source_type == "Manual" else auto_visualize_url
        
        if auto_viz and (individuals or organizations):
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Visualisasi Data")
            
            fig_bar, fig_pie = create_visualization(individuals, organizations)
            
            if fig_bar and fig_pie:
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_viz2:
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("ℹ️ Tidak ada data untuk divisualisasikan.")
        
        st.divider()
        
        col_ind, col_org = st.columns(2)
        
        with col_ind:
            st.markdown("### 👤 Daftar Individu (PERSON)")
            if individuals:
                for nama, score in sorted(individuals.items(), key=lambda x: x[1], reverse=True):
                    if show_conf:
                        st.markdown(f"""
                        <div class='entity-box'>
                            <span class='entity-name'>{nama}</span>
                            <span class='person-tag'>PERSON</span>
                            <div style='margin-top: 0.5rem; font-size: 0.85rem; color: #666;'>
                                🔹 Confidence: <strong>{score:.4f}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Tidak ada entitas individu yang terdeteksi.")
        
        with col_org:
            st.markdown("### 🏢 Daftar Instansi (ORGANIZATION)")
            if organizations:
                for instansi, score in sorted(organizations.items(), key=lambda x: x[1], reverse=True):
                    if show_conf:
                        st.markdown(f"""
                        <div class='entity-box'>
                            <span class='entity-name'>{instansi}</span>
                            <span class='org-tag'>ORGANIZATION</span>
                            <div style='margin-top: 0.5rem; font-size: 0.85rem; color: #666;'>
                                🔹 Confidence: <strong>{score:.4f}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Tidak ada entitas instansi yang terdeteksi.")
        
        st.divider()
        st.markdown("### 💾 Export Hasil")
        
        csv_data = export_to_csv(individuals, organizations, timestamp, source_type)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"ner_results_{timestamp.replace(':', '-')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    elif (analyze_manual and not teks_berita.strip()) or (analyze_url and not url_input.strip()):
        st.error("⚠️ Silakan masukkan teks atau URL terlebih dahulu!")

elif menu == "📜 Riwayat":
    st.markdown("""
    <div class='main-header'>
        <h1>📜 Riwayat Analisis</h1>
        <p>Visualisasi interaktif dan statistik dari semua analisis yang telah dilakukan</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.analysis_history:
        total_analisis = len(st.session_state.analysis_history)
        total_individu = sum(item['individuals_count'] for item in st.session_state.analysis_history)
        total_instansi = sum(item['organizations_count'] for item in st.session_state.analysis_history)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🔍 Total Analisis", total_analisis)
        col2.metric("👤 Total Individu", total_individu)
        col3.metric("🏢 Total Instansi", total_instansi)
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📊 Tren Visual", "🏆 Top Entitas", "📋 Tabel Data"])
        
        with tab1:
            st.markdown("### 📈 Distribusi Entitas dari Waktu ke Waktu")
            
            df_history = pd.DataFrame(st.session_state.analysis_history)
            df_history['time_short'] = df_history['timestamp'].apply(lambda x: x.split(' ')[1] if ' ' in x else x)
            
            fig_tren = go.Figure()
            
            fig_tren.add_trace(go.Scatter(
                x=df_history['time_short'], y=df_history['individuals_count'],
                mode='lines+markers+text', name='Individu',
                line=dict(color='#667eea', width=3), marker=dict(size=8, symbol='circle'),
                text=df_history['individuals_count'], textposition='top center'
            ))
            
            fig_tren.add_trace(go.Scatter(
                x=df_history['time_short'], y=df_history['organizations_count'],
                mode='lines+markers+text', name='Instansi',
                line=dict(color='#764ba2', width=3), marker=dict(size=8, symbol='square'),
                text=df_history['organizations_count'], textposition='bottom center'
            ))
            
            fig_tren.update_layout(
                title="📊 Tren Jumlah Entitas Terdeteksi",
                xaxis_title="⏰ Waktu Analisis", yaxis_title="🔢 Jumlah Entitas",
                height=450, hovermode='x unified', plot_bgcolor='rgba(0,0,0,0.02)',
                font=dict(family='Inter, sans-serif')
            )
            
            st.plotly_chart(fig_tren, use_container_width=True)
            
            col_pie1, col_pie2 = st.columns(2)
            
            with col_pie1:
                fig_pie = px.pie(
                    values=[total_individu, total_instansi], names=['Individu', 'Instansi'],
                    title='🥧 Total Distribusi Keseluruhan',
                    color_discrete_map={'Individu': '#667eea', 'Instansi': '#764ba2'}, hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+label', hoverinfo='label+percent+value')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_pie2:
                st.markdown("### 📊 Statistik Ringkas")
                st.metric("Rata-rata Individu/Analisis", f"{total_individu/total_analisis:.2f}")
                st.metric("Rata-rata Instansi/Analisis", f"{total_instansi/total_analisis:.2f}")
                
                if 'source' in df_history.columns:
                    source_counts = df_history['source'].value_counts()
                    st.markdown("**📰 Analisis per Sumber:**")
                    for src, count in source_counts.items():
                        st.write(f"- {src}: **{count}** analisis")
        
        with tab2:
            st.markdown("### 🏆 Top 10 Entitas Berdasarkan Confidence Score")
            
            all_ind, all_org = {}, {}
            
            for analysis in st.session_state.analysis_history:
                for nama, score in analysis['individuals'].items():
                    if nama not in all_ind or score > all_ind[nama]:
                        all_ind[nama] = score
                for org, score in analysis['organizations'].items():
                    if org not in all_org or score > all_org[org]:
                        all_org[org] = score
            
            col_b1, col_b2 = st.columns(2)
            
            with col_b1:
                if all_ind:
                    df_ind = pd.DataFrame(list(all_ind.items()), columns=['Nama', 'Score'])
                    df_ind = df_ind.sort_values('Score', ascending=False).head(10)
                    fig = px.bar(df_ind, x='Score', y='Nama', orientation='h', title='👤 Top 10 Individu', color='Score', color_continuous_scale='Blues')
                    fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_b2:
                if all_org:
                    df_org = pd.DataFrame(list(all_org.items()), columns=['Instansi', 'Score'])
                    df_org = df_org.sort_values('Score', ascending=False).head(10)
                    fig = px.bar(df_org, x='Score', y='Instansi', orientation='h', title='🏢 Top 10 Instansi', color='Score', color_continuous_scale='Purples')
                    fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📋 Tabel Histori Lengkap")
            
            df_table = pd.DataFrame(st.session_state.analysis_history)
            cols_to_show = ['timestamp', 'source', 'text_preview', 'individuals_count', 'organizations_count']
            df_display = df_table[[c for c in cols_to_show if c in df_table.columns]].copy()
            df_display.columns = ['⏰ Waktu', '📰 Sumber', '📝 Preview', '👤 Individu', '🏢 Instansi']
            
            st.dataframe(
                df_display.sort_values('⏰ Waktu', ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "⏰ Waktu": st.column_config.TextColumn(width="medium"),
                    "📰 Sumber": st.column_config.TextColumn(width="small"),
                    "📝 Preview": st.column_config.TextColumn(width="large"),
                    "👤 Individu": st.column_config.NumberColumn(format="%d", width="small"),
                    "🏢 Instansi": st.column_config.NumberColumn(format="%d", width="small"),
                }
            )
            
            if st.button("📥 Export Semua Data (CSV)", type="secondary", use_container_width=True):
                export_data = []
                for analysis in st.session_state.analysis_history:
                    for nama, score in analysis['individuals'].items():
                        export_data.append({'Timestamp': analysis['timestamp'], 'Source': analysis.get('source', 'Manual'), 'Tipe': 'PERSON', 'Nama': nama, 'Score': f"{score:.4f}"})
                    for org, score in analysis['organizations'].items():
                        export_data.append({'Timestamp': analysis['timestamp'], 'Source': analysis.get('source', 'Manual'), 'Tipe': 'ORGANIZATION', 'Nama': org, 'Score': f"{score:.4f}"})
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="⬇️ Download Sekarang", data=csv,
                    file_name=f"all_ner_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", use_container_width=True
                )
            
            if st.button("🗑️ Hapus Semua Riwayat", type="secondary", use_container_width=True):
                st.session_state.analysis_history = []
                save_history_to_file([])
                st.rerun()
    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: rgba(255,255,255,0.8); border-radius: 20px; border: 2px dashed rgba(102,126,234,0.3);'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📭</div>
            <h3 style='color: #667eea; margin: 0.5rem 0;'>Belum Ada Riwayat</h3>
            <p style='color: #666;'>Silakan lakukan analisis teks berita terlebih dahulu untuk melihat histori di sini.</p>
            <div style='margin-top: 1.5rem;'>
                <a href='#' style='color: #667eea; text-decoration: none; font-weight: 500;'>🚀 Mulai Analisis Sekarang</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif menu == "ℹ️ Tentang":
    st.markdown("""
    <div class='main-header'>
        <h1>ℹ️ Tentang Aplikasi</h1>
        <p>Informasi lengkap mengenai sistem dan teknologi yang digunakan</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='text-align: left;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🎯 Tujuan Aplikasi</h3>
            <p style='color: #555; line-height: 1.7;'>
                Aplikasi ini mengimplementasikan <strong>Named Entity Recognition (NER)</strong> 
                berbasis <strong>Custom IndoBERT</strong> yang telah di-fine-tune khusus untuk 
                ekstraksi otomatis entitas individu dan instansi dari berita korupsi di Indonesia.
            </p>
            <h4 style='color: #667eea; margin: 1.2rem 0 0.5rem;'>✨ Manfaat:</h4>
            <ul style='color: #555; line-height: 1.8; padding-left: 1.2rem;'>
                <li>Mempercepat analisis berita korupsi dengan akurasi tinggi</li>
                <li>Mengidentifikasi pelaku dan institusi terkait secara otomatis</li>
                <li>Mendukung penelitian dan jurnalisme data investigatif</li>
                <li>Otomatisasi ekstraksi informasi penting dari teks berita</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card' style='text-align: left;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🛠️ Teknologi</h3>
            <ul style='color: #555; line-height: 2; padding-left: 1.2rem;'>
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>Model NLP:</strong> Custom IndoBERT NER (Fine-tuned)</li>
                <li><strong>Library:</strong> Transformers (Hugging Face)</li>
                <li><strong>Visualisasi:</strong> Plotly Interactive Charts</li>
                <li><strong>Web Scraping:</strong> BeautifulSoup + Requests</li>
                <li><strong>Bahasa:</strong> Python 3.x</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='text-align: left;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>📊 Spesifikasi Model</h3>
            <div style='background: rgba(243,229,245,0.6); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;'>
                <strong>🤖 Model:</strong> Custom IndoBERT NER (Fine-tuned)<br>
                <strong>🗣️ Bahasa:</strong> Indonesia<br>
                <strong>🎯 Entitas:</strong> PERSON, ORGANIZATION<br>
                <strong>🏗️ Arsitektur:</strong> BERT Base (12-layer, 768-hidden, 12-heads)<br>
                <strong>📚 Training:</strong> Custom Dataset Berita Korupsi Indonesia<br>
                <strong>📁 Lokasi:</strong> <code>models/checkpoint-80/</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card' style='text-align: left;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>👨‍💻 Developer</h3>
            <p style='color: #555; line-height: 1.7;'>
                <strong>By kelompok 4</strong><br>
                Universitas [Nama Universitas]<br>
                Mata Kuliah: Natural Language Processing<br><br>
                <strong>By Kelompok 4</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

# ==========================================
# 8. FOOTER DENGAN ANIMASI
# ==========================================
st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='app-footer'>
    <p>✨ © 2026 Kelompok 4 | NER Korupsi Indonesia</p>
    <p style='font-size: 0.85rem; color: #999;'>Proyek Mata Kuliah Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)
