# =========================================================
# STREAMLIT APP — DEPLOY-SAFE VERSION
# =========================================================

import sys
import os
import urllib.parse

import streamlit as st
import joblib

# ===== FIX PATH AGAR src TERDETECT SAAT DEPLOY =====
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.recommender import build_recommender, recommend

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Coffee Shop Finder Jogja",
    page_icon="☕",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.stApp { background: #FFF5E6; }

.hero {
    background: linear-gradient(135deg, #FFE4B5, #FFD8A8);
    padding: 70px 30px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 50px;
}
.hero h1 { font-size: 48px; color: #3B270C; font-weight: 900; }
.hero p { font-size: 20px; color: #4B3A26; }

.form-card {
    background-color: #FFF8F0;
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 50px;
}

.segment-card {
    background-color: #F5E8D0;
    padding: 28px;
    border-radius: 20px;
    border-left: 8px solid #4B3A26;
    margin-bottom: 35px;
}

.top1-card {
    background: linear-gradient(135deg, #795C32, #A67C52);
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 30px;
    color: white;
}

.recom-card {
    background-color: #FFF5E6;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
}

.maps-link {
    background-color: #4B3A26;
    color: white !important;
    padding: 8px 16px;
    text-decoration: none;
    border-radius: 8px;
}

.watermark {
    text-align: center;
    color: #4B3A26;
    font-size: 14px;
    margin-top: 50px;
    opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS & RECOMMENDER (CACHE)
# =========================================================
@st.cache_resource
def load_resources():
    with st.spinner("🔧 Menyiapkan sistem rekomendasi (pertama kali agak lama)..."):
        # Path folder models relatif terhadap streamlit_app.py
        base_dir = os.path.join(os.path.dirname(__file__), "models")

        kmodes = joblib.load(os.path.join(base_dir, "kmodes_model.pkl"))
        category_mappings = joblib.load(os.path.join(base_dir, "category_mappings.pkl"))
        df, tfidf, tfidf_matrix, sbert, embeddings = build_recommender()
    return kmodes, category_mappings, df, tfidf, tfidf_matrix, sbert, embeddings

# Load semua resources
kmodes, category_mappings, df, tfidf, tfidf_matrix, sbert, embeddings = load_resources()
st.success("✅ Sistem rekomendasi siap digunakan")

# =========================================================
# SEGMENT INFO
# =========================================================
segment_info = {
    0: {"name": "Instagrammable & Aesthetic",
        "desc": "Kamu menyukai coffee shop dengan desain visual yang estetik dan menarik untuk berfoto."},
    1: {"name": "Casual Coffee Drinker (Lokal)",
        "desc": "Kamu menikmati suasana santai dengan kopi yang ramah di lidah."},
    2: {"name": "Premium Coffee Enthusiast",
        "desc": "Kamu mengutamakan kualitas biji kopi dan pengalaman rasa."},
    3: {"name": "Productive Work / Study",
        "desc": "Kamu membutuhkan tempat yang nyaman untuk fokus bekerja atau belajar."}
}

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>☕ Coffee Shop Finder Jogja</h1>
    <p>Sistem rekomendasi coffee shop berbasis segmentasi pelanggan dan analisis ulasan.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# FORM INPUT
# =========================================================
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown("### Preferensi Anda")

with st.form("user_form"):
    col1, col2 = st.columns(2)

    with col1:
        tujuan = st.selectbox(
            "Tujuan Kunjungan",
            list(category_mappings["Tujuan utama Anda ke coffee shop?"].values())
        )
        faktor = st.selectbox(
            "Faktor Penentu",
            list(category_mappings["Faktor utama yang paling memengaruhi Anda dalam memilih coffee shop"].values())
        )

    with col2:
        minuman = st.selectbox(
            "Minuman Favorit",
            list(category_mappings["Jenis minuman yang paling sering Anda pesan di coffee shop"].values())
        )
        duduk = st.selectbox(
            "Tempat Duduk Favorit",
            list(category_mappings["Jenis tempat duduk favorit Anda"].values())
        )

    kebutuhan = st.text_input("Kebutuhan khusus", placeholder="wifi kencang, banyak colokan")
    area = st.text_input("Area (opsional)", placeholder="Gejayan, Jakal, UGM")

    submitted = st.form_submit_button("🔍 Cari Rekomendasi")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PROCESS & OUTPUT
# =========================================================
if submitted:
    # Encode user input
    user_input = {
        "Tujuan utama Anda ke coffee shop?": tujuan,
        "Faktor utama yang paling memengaruhi Anda dalam memilih coffee shop": faktor,
        "Jenis minuman yang paling sering Anda pesan di coffee shop": minuman,
        "Jenis tempat duduk favorit Anda": duduk
    }

    encoded = []
    for col, val in user_input.items():
        rev = {v: k for k, v in category_mappings[col].items()}
        encoded.append(rev[val])

    cluster_id = kmodes.predict([encoded])[0]
    segment = segment_info[cluster_id]

    st.markdown(f"""
    <div class="segment-card">
        <h2>{segment['name']}</h2>
        <p>{segment['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Cari rekomendasi
    with st.spinner("☕ Mencari coffee shop terbaik..."):
        results = recommend(
            df, tfidf, tfidf_matrix, sbert, embeddings,
            user_text=kebutuhan,
            segment=segment["name"],
            lokasi=area if area else None,
            top_k=5
        )

    st.markdown("### ☕ Rekomendasi Coffee Shop")

    if results.empty:
        st.warning("Tidak ditemukan coffee shop yang sesuai.")
    else:
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            query = urllib.parse.quote(f"{row['name']} {row['area']} Yogyakarta")
            maps_url = f"https://www.google.com/maps/search/?api=1&query={query}"

            if i == 1:
                st.markdown(f"""
                <div class="top1-card">
                    <h2>{row['name']} 🏆</h2>
                    <p>⭐ {row['rating']:.2f} | 📍 {row['area']}</p>
                    <p>{row['address']}</p>
                    <a href="{maps_url}" target="_blank" class="maps-link">📍 Petunjuk Lokasi</a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recom-card">
                    <h4>#{i} {row['name']}</h4>
                    <p>⭐ {row['rating']:.2f} | 📍 {row['area']}</p>
                    <p>{row['address']}</p>
                    <a href="{maps_url}" target="_blank" class="maps-link">📍 Petunjuk Lokasi</a>
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown('<div class="watermark">By Angel & Thania</div>', unsafe_allow_html=True)
st.caption(
    "Model menggunakan K-Modes Clustering, TF-IDF, Sentence-BERT, dan keyword-based segmentation."
)
