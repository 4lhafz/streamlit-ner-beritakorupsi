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
            with open(HISTORY_FILE, \"r\", encoding=\"utf-8\") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history_to_file(history):
    """Menyimpan riwayat ke file JSON"""
    with open(HISTORY_FILE, \"w\", encoding=\"utf-8\") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA MODERN
# ==========================================
st.set_page_config(
    page_title="✨ Nerkor",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kustomisasi CSS Antarmuka Modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #4a5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4c51bf;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State Riwayat Analisis
if "ner_history" not in st.session_state:
    st.session_state.ner_history = load_history_from_file()

# ==========================================
# 2. INI LOAD MODEL & TOKENIZER (DENGAN CACHE)
# ==========================================
@st.cache_resource
def load_model_and_tokenizer():
    """Memuat model IndoBERT NER kustom secara efisien"""
    # Ganti path folder ini sesuai dengan folder penyimpanan lokal kamu
    model_path = "./hasil_model_ner"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Folder model tidak ditemukan di: `{model_path}`. Pastikan folder sudah diekstrak.")
        return None, None
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

tokenizer, model = load_model_and_tokenizer()

# ==========================================
# 3. FUNGSI PREDIKSI NER (SUDAH DIPERBAIKI SAKTI)
# ==========================================
def predict_ner_from_text(text):
    """
    Memprediksi entitas NER dari teks menggunakan Hugging Face Pipeline
    agar sub-words otomatis tersambung dan rapi.
    """
    if not text.strip() or model is None or tokenizer is None:
        return []
        
    try:
        # Inisialisasi Pipeline dengan aggregation_strategy untuk menyatukan sub-words
        device = 0 if torch.cuda.is_available() else -1
        nlp_ner = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="simple" # <- Menyatukan pecahan kata (##) otomatis
        )
        
        hasil_raw = nlp_ner(text)
        
        entities = []
        for ent in hasil_raw:
            # Saring hasil: HANYA mengambil kelas entitas penting (bukan 'O')
            if ent['entity_group'] != "O":
                entities.append({
                    "word": ent['word'],
                    "label": ent['entity_group'],
                    "score": float(ent['score'])
                })
        return entities
        
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat prediksi model: {e}")
        return []

# ==========================================
# 4. FUNGSI UTENSIL: WEB SCRAPING URL
# ==========================================
def scrape_article_from_url(url):
    """Mengambil teks isi berita dari link berita lokal"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return None, f"Gagal mengakses halaman (Status Code: {res.status_code})"
            
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Cari judul berita
        title = ""
        h1_tag = soup.find('h1')
        if h1_tag:
            title = h1_tag.get_text().strip()
            
        # Ekstraksi isi artikel berita dari tag paragraf (<p>)
        paragraphs = soup.find_all('p')
        text_content = []
        for p in paragraphs:
            p_text = p.get_text().strip()
            if len(p_text.split()) > 10:  # Hindari teks navigasi pendek
                text_content.append(p_text)
                
        # Gabungkan beberapa paragraf awal agar pas dalam batas token BERT
        clean_text = " ".join(text_content[:4])
        
        if not clean_text:
            return None, "Gagal mengesktrak isi teks paragraf dari artikel."
            
        return {"title": title, "text": clean_text}, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 5. STRUKTUR SIDEBAR (MENU NAVIGASI & RIWAYAT)
# ==========================================
with st.sidebar:
    st.markdown("<div style='text-align: center;'><h2 style='color:#667eea;font-weight:700;'>🇮🇩 NERKOR APP</h2></div>", unsafe_allow_html=True)
    st.markdown("Aplikasi Analisis Informasi Hukum Tindak Pidana Korupsi berbasis *Deep Learning*.")
    st.divider()
    
    st.subheader("📋 Riwayat Analisis")
    if len(st.session_state.ner_history) == 0:
        st.caption("Belum ada riwayat analisis.")
    else:
        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            st.session_state.ner_history = []
            save_history_to_file([])
            st.rerun()
            
        st.markdown("---")
        # Menampilkan daftar riwayat secara terbalik (terbaru di atas)
        for idx, item in enumerate(reversed(st.session_state.ner_history)):
            real_idx = len(st.session_state.ner_history) - 1 - idx
            with st.expander(f"🕒 {item['timestamp']} | {item['source']}"):
                st.caption(f"**Teks:** {item['text'][:100]}...")
                st.markdown(f"**Entitas Ditemukan:** `{item['counts']['PERSON']}` Person, `{item['counts']['ORGANIZATION']}` Org")
                if st.button("👁️ Muat Teks", key=f"load_{real_idx}", use_container_width=True):
                    st.session_state.loaded_text_from_history = item['text']
                    st.rerun()

# ==========================================
# 6. DASHBOARD HALAMAN UTAMA (TABS SYSTEM)
# ==========================================
st.markdown("<h1 class='main-title'>Ekstraksi Entitas Kasus Korupsi</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Deteksi otomatis nama Koruptor (PERSON) dan Instansi (ORGANIZATION) menggunakan IndoBERT NLP</p>", unsafe_allow_html=True)

# Inisialisasi default teks input jika tombol muat riwayat ditekan
default_input_text = ""
if "loaded_text_from_history" in st.session_state:
    default_input_text = st.session_state.loaded_text_from_history
    del st.session_state.loaded_text_from_history

tab_input, tab_stats, tab_info = st.tabs(["📝 Input Analisis", "📊 Visualisasi & Statistik", "ℹ️ Informasi Model"])

# ------------------------------------------
# TAB 1: KOTAK UTAMA INPUT DAN PROSES PREDIKSI
# ------------------------------------------
with tab_input:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📥 Masukkan Data Artikel")
        sub_tab_text, sub_tab_url = st.tabs(["✍️ Paste Teks Manual", "🔗 Scraping dari URL Berita"])
        
        input_text = ""
        
        with sub_tab_text:
            input_text = st.text_area(
                "Masukkan Artikel Berita:",
                value=default_input_text,
                height=250,
                placeholder="Paste paragraf berita hukum atau dakwaan korupsi di sini..."
            )
            btn_analyze = st.button("Ekstrak Entitas Teks 🚀", type="primary", use_container_width=True)
            
        with sub_tab_url:
            url_input = st.text_input("Masukkan URL Link Berita:", placeholder="https://nasional.kompas.com/read/...")
            btn_scrape = st.button("Ambil & Ekstrak Berita 🌐", type="secondary", use_container_width=True)
            
            if btn_scrape:
                if not url_input.strip():
                    st.warning("Silakan isi tautan URL terlebih dahulu!")
                else:
                    with st.spinner("🕷️ Sedang melakukan scraping artikel berita..."):
                        scraped_data, err = scrape_article_from_url(url_input)
                        if err:
                            st.error(f"Gagal melakukan scraping: {err}")
                        else:
                            st.success(f"Sukses Mengambil Berita: \"{scraped_data['title']}\"")
                            input_text = scraped_data['text']
                            # Memicu paksa proses analisis setelah sukses scraping
                            btn_analyze = True

    with col_right:
        st.markdown("### 💡 Panduan Entitas")
        st.markdown("""
        <div class='metric-card' style='text-align: left; margin-bottom: 1rem; border-left: 5px solid #4299e1;'>
            <h5 style='color:#2b6cb0; margin:0;'>👤 Label: PERSON</h5>
            <p style='color:#4a5568; font-size:0.85rem; margin-top:0.25rem;'>
                Mendeteksi nama koruptor, saksi, hakim, jaksa, atau tokoh manusia penegak hukum lainnya.<br>
                <b>Contoh:</b> Firli Bahuri, Surya Darmadi, Edhy Prabowo.
            </p>
        </div>
        <div class='metric-card' style='text-align: left; border-left: 5px solid #48bb78;'>
            <h5 style='color:#2f855a; margin:0;'>🏢 Label: ORGANIZATION</h5>
            <p style='color:#4a5568; font-size:0.85rem; margin-top:0.25rem;'>
                Mendeteksi nama lembaga negara, kementerian, perusahaan swasta, atau organisasi terkait.<br>
                <b>Contoh:</b> KPK, MA, Kementerian Sosial, PT Ratu Samban Mining.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # AKSI: PROSES JALANKAN PREDIKSI MODEL INDOBERT
    if btn_analyze or (btn_scrape and 'input_text' in locals() and input_text):
        if not input_text.strip():
            st.warning("⚠️ Kotak teks kosong. Masukkan teks artikel atau link berita valid sebelum memproses!")
        else:
            st.markdown("---")
            with st.spinner("⏳ Sedang menganalisis teks dengan kecerdasan IndoBERT..."):
                entities = predict_ner_from_text(input_text)
                
            if entities:
                # Membuat format tabel DataFrame dari data bersih pipeline
                df_data = []
                count_person = 0
                count_org = 0
                
                for ent in entities:
                    label_name = ent['label']
                    if label_name == 'PERSON':
                        count_person += 1
                    elif label_name == 'ORGANIZATION':
                        count_org += 1
                        
                    df_data.append({
                        "Teks / Nama Entitas": ent['word'],
                        "Kategori Label": label_name,
                        "Tingkat Keyakinan (Score)": f"{ent['score'] * 100:.2f}%"
                    })
                    
                df_results = pd.DataFrame(df_data)
                
                # Simpan hasil ke riwayat sistem (Persist ke JSON)
                now_wib = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
                history_entry = {
                    "timestamp": now_wib,
                    "source": "URL Berita" if btn_scrape else "Input Teks",
                    "text": input_text,
                    "entities": entities,
                    "counts": {"PERSON": count_person, "ORGANIZATION": count_org}
                }
                st.session_state.ner_history.append(history_entry)
                save_history_to_file(st.session_state.ner_history)
                
                # Menampilkan Kartu Metrik Ringkasan Hasil
                st.markdown("### 📊 Ringkasan Temuan")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(entities)}</div><div class='metric-label'>Total Entitas</div></div>", unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#3182ce;'>{count_person}</div><div class='metric-label'>Nama Orang (PERSON)</div></div>", unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#38a169;'>{count_org}</div><div class='metric-label'>Instansi (ORGANIZATION)</div></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Menampilkan Komponen Tabel Interaktif Utama
                st.markdown("<h4 style='color: #4a5568;'>📋 Tabel Daftar Hasil Deteksi</h4>", unsafe_allow_html=True)
                st.dataframe(df_results, use_container_width=True)
                
            else:
                st.info("ℹ️ Model selesai membaca, namun tidak ada entitas Nama Orang (PERSON) atau Instansi (ORGANIZATION) yang terdeteksi.")

# ------------------------------------------
# TAB 2: VISUALISASI GRAFIK INTERAKTIF (PLOTLY)
# ------------------------------------------
with tab_stats:
    st.markdown("### 📈 Statistik Akumulasi Riwayat Penggunaan")
    
    if len(st.session_state.ner_history) == 0:
        st.info("Belum ada statistik tersedia. Silakan lakukan proses prediksi teks terlebih dahulu.")
    else:
        # Menghitung total data akumulasi dari seluruh riwayat analisis
        total_p = sum([item['counts']['PERSON'] for item in st.session_state.ner_history])
        total_o = sum([item['counts']['ORGANIZATION'] for item in st.session_state.ner_history])
        
        if total_p == 0 and total_o == 0:
            st.warning("Belum ada data entitas positif yang tersimpan di dalam file riwayat.")
        else:
            g_col1, g_col2 = st.columns(2)
            
            with g_col1:
                st.markdown("#### 🍩 Perbandingan Distribusi Kelas Entitas")
                fig_pie = px.pie(
                    names=['PERSON (Nama Orang)', 'ORGANIZATION (Instansi)'],
                    values=[total_p, total_o],
                    color_discrete_sequence=['#3182ce', '#38a169'],
                    hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+value')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with g_col2:
                st.markdown("#### 📊 Grafik Frekuensi Kemunculan Entitas")
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=['PERSON', 'ORGANIZATION'], 
                        y=[total_p, total_o],
                        marker_color=['#3182ce', '#38a169'],
                        text=[total_p, total_o],
                        textposition='auto',
                    )
                ])
                fig_bar.update_layout(
                    yaxis_title="Jumlah Total Frekuensi Kata",
                    xaxis_title="Kategori Label NER",
                    template="plotly_white"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------
# TAB 3: INFORMASI METADATA MODEL & KELOMPOK
# ------------------------------------------
with tab_info:
    st.markdown("### ℹ️ Spesifikasi Teknis Sistem AI")
    
    i_col1, i_col2 = st.columns(2)
    with i_col1:
        st.markdown("""
        <div class='metric-card' style='text-align: left; background-color: #f7fafc;'>
            <h3 style='color: #4c51bf; margin-bottom: 1rem;'>🧠 Arsitektur Model</h3>
            <div style='background: rgba(102,119,233,0.05); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border: 1px solid rgba(102,119,233,0.15);'>
                <strong>🤖 Basis Model:</strong> IndoBERT Base (Fine-tuned)<br>
                <strong>🗣️ Bahasa Utama:</strong> Bahasa Indonesia (Formal & Berita)<br>
                <strong>🎯 Kategori Entitas:</strong> <code>PERSON</code>, <code>ORGANIZATION</code><br>
                <strong>🏗️ Ukuran Arsitektur:</strong> 12-layer, 768-hidden, 12-heads (110M Parameter)<br>
                <strong>📚 Data Latihan:</strong> Korpus Kustom Dokumen Berita Kasus Korupsi Indonesia<br>
                <strong>📁 Path Load Sistem:</strong> <code>./hasil_model_ner/</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with i_col2:
        st.markdown("""
        <div class='metric-card' style='text-align: left; background-color: #f7fafc;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>👨‍💻 Tim Pengembang</h3>
            <p style='color: #4a5568; line-height: 1.7;'>
                <strong>Dikembangkan oleh: Kelompok 4</strong><br>
                Proyek Tugas Mata Kuliah: <i>Natural Language Processing (NLP)</i><br>
                Program Studi: Ilmu Komputer / Teknik Informatika<br>
                Status Model: <b>Siap Digunakan (Production Mode)</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. FOOTER TAMPILAN APLIKASI
# ==========================================
st.markdown("<div style='height: 5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #a0aec0; font-size: 0.85rem; border-top: 1px solid #e2e8f0; padding-top: 1.5rem;'>
    🇮🇩 Nerkor App v2.0 © 2026 | Built with Streamlit & Hugging Face Transformers
</div>
""", unsafe_allow_html=True)