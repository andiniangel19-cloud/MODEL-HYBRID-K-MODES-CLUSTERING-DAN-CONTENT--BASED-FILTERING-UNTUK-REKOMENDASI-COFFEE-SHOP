# =========================================================
# STREAMLIT APP — FINAL FIXED VERSION
# =========================================================

import sys
import os
import urllib.parse
import streamlit as st
import joblib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.recommender import build_recommender, recommend

st.set_page_config(
    page_title="COFFE SHOP FINDER JOGJA",
    page_icon="☕",
    layout="wide"
)

# =========================================================
# CUSTOM CSS (FINAL)
# =========================================================
st.markdown("""
<style>
.stApp { background: #FFF5E6; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #FFE4B5, #FFD8A8);
    padding: 70px 30px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 50px;
}
.hero h1 { font-size: 48px; color: #3B270C; font-weight: 900; }
.hero p { font-size: 20px; color: #4B3A26; }

/* FORM */
.form-card {
    background-color: #3B270C;
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 50px;
}
.form-title {
    color: #3B270C;
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 25px;
}

/* FORM LABEL & INPUT */
label, .stTextInput label, .stSelectbox label {
    color: #3B270C !important;
    font-weight: 700;
}
input, textarea {
    color: #3B270C !important;
}

/* SEGMENT */
.segment-card {
    background-color: #F5E8D0;
    padding: 28px;
    border-radius: 20px;
    border-left: 8px solid #4B3A26;
    margin-bottom: 35px;
}
.segment-card h2, .segment-card p {
    color: #3B270C !important;
}

/* RECOMMENDATION TITLE */
.rec-title {
    color: #3B270C;
    font-size: 26px;
    font-weight: 800;
    margin: 30px 0 25px 0;
}

/* TOP 1 */
.top1-card {
    background: linear-gradient(135deg, #795C32, #A67C52);
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 30px;
    color: white;
    position: relative;
}
.top1-card * { color: white !important; }

.top-badge {
    position: absolute;
    top: 18px;
    right: 18px;
    background: #4B3A26;
    color: white;
    padding: 8px 14px;
    border-radius: 12px;
    font-weight: 900;
}

/* TOP 2–5 */
.recom-card {
    background-color: #F3E1C6;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 22px;
    position: relative;
}
.recom-card h4, .recom-card p {
    color: #3B270C;
}

.recom-badge {
    position: absolute;
    top: 16px;
    right: 16px;
    background: #A67C52;
    color: white;
    padding: 6px 12px;
    border-radius: 10px;
    font-weight: 800;
}

/* MAPS */
.maps-link {
    background-color: #4B3A26;
    color: white !important;
    padding: 8px 16px;
    border-radius: 8px;
    text-decoration: none;
    display: inline-block;
    margin-top: 10px;
}

/* FOOTER */
.watermark, .footer-text {
    text-align: center;
    color: #3B270C;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_resources():
    base_dir = os.path.join(os.path.dirname(__file__), "models")
    kmodes = joblib.load(os.path.join(base_dir, "kmodes_model.pkl"))
    category_mappings = joblib.load(os.path.join(base_dir, "category_mappings.pkl"))
    df, tfidf, tfidf_matrix, sbert, embeddings = build_recommender()
    return kmodes, category_mappings, df, tfidf, tfidf_matrix, sbert, embeddings

kmodes, category_mappings, df, tfidf, tfidf_matrix, sbert, embeddings = load_resources()

# =========================================================
# SEGMENT INFO (ASLI PUNYAMU)
# =========================================================
segment_info = {
    0: {"name": "Instagrammable & Aesthetic", "desc": "Kamu menyukai coffee shop dengan desain visual yang unik dan estetik, sangat cocok untuk berfoto."},
    1: {"name": "Casual Coffee Drinker (Lokal)", "desc": "Kamu menikmati suasana santai dengan pilihan kopi yang ramah di lidah dan nyaman untuk ngobrol."},
    2: {"name": "Premium Coffee Enthusiast", "desc": "Kamu mengutamakan kualitas biji kopi, teknik seduh manual, dan pengalaman rasa yang serius."},
    3: {"name": "Productive Work / Study", "desc": "Kamu membutuhkan ruang yang tenang, kursi yang nyaman, dan suasana yang mendukung fokus bekerjar."}
}

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
<h1>☕ COFFE SHOP FINDER JOGJA</h1>
<p>Temukan coffee shop terbaik di Yogyakarta dengan rekomendasi berbasis preferensi dan karakter unik Anda.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FORM
# =========================================================
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<div class="form-title">Preferensi Anda</div>', unsafe_allow_html=True)

with st.form("user_form"):
    col1, col2 = st.columns(2)

    with col1:
        tujuan = st.selectbox("Tujuan Kunjungan", list(category_mappings["Tujuan utama Anda ke coffee shop?"].values()))
        faktor = st.selectbox("Faktor Penentu", list(category_mappings["Faktor utama yang paling memengaruhi Anda dalam memilih coffee shop"].values()))

    with col2:
        minuman = st.selectbox("Minuman Favorit", list(category_mappings["Jenis minuman yang paling sering Anda pesan di coffee shop"].values()))
        duduk = st.selectbox("Tempat Duduk Favorit", list(category_mappings["Jenis tempat duduk favorit Anda"].values()))

    kebutuhan = st.text_input("Kebutuhan khusus", placeholder="wifi kencang, banyak colokan")
    area = st.text_input("Area (opsional)", placeholder="Gejayan, Jakal, UGM")

    submitted = st.form_submit_button("🔍 Cari Rekomendasi")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# OUTPUT
# =========================================================
if submitted:
    encoded = []
    for col, val in {
        "Tujuan utama Anda ke coffee shop?": tujuan,
        "Faktor utama yang paling memengaruhi Anda dalam memilih coffee shop": faktor,
        "Jenis minuman yang paling sering Anda pesan di coffee shop": minuman,
        "Jenis tempat duduk favorit Anda": duduk
    }.items():
        rev = {v: k for k, v in category_mappings[col].items()}
        encoded.append(rev[val])

    cluster = kmodes.predict([encoded])[0]
    seg = segment_info[cluster]

    st.markdown(f"""
    <div class="segment-card">
        <h2>{seg['name']}</h2>
        <p>{seg['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    results = recommend(df, tfidf, tfidf_matrix, sbert, embeddings, kebutuhan, seg["name"], area, 5)

    st.markdown('<div class="rec-title"> ☕ Rekomendasi Coffee Shop</div>', unsafe_allow_html=True)

    for i, (_, row) in enumerate(results.iterrows(), 1):
        query = urllib.parse.quote(f"{row['name']} {row['area']} Yogyakarta")
        maps = f"https://www.google.com/maps/search/?api=1&query={query}"

        if i == 1:
            st.markdown(f"""
            <div class="top1-card">
                <div class="top-badge">🏆 TOP 1</div>
                <h2>{row['name']}</h2>
                <p>⭐ {row['rating']:.2f} | 📍 {row['area']}</p>
                <p>{row['address']}</p>
                <a href="{maps}" target="_blank" class="maps-link">📍 Petunjuk Lokasi</a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="recom-card">
                <div class="recom-badge">TOP {i}</div>
                <h4>{row['name']}</h4>
                <p>⭐ {row['rating']:.2f} | 📍 {row['area']}</p>
                <p>{row['address']}</p>
                <a href="{maps}" target="_blank" class="maps-link">📍 Petunjuk Lokasi</a>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="watermark">By Angel & Thania</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-text">Model menggunakan K-Modes Clustering, TF-IDF, Sentence-BERT, dan keyword-based segmentation.</div>', unsafe_allow_html=True)
