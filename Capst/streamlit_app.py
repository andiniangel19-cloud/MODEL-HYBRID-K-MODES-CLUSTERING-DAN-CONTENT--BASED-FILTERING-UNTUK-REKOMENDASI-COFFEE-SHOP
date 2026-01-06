import streamlit as st
import joblib
import urllib.parse
from src.recommender import build_recommender, recommend


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Coffee Shop Finder Jogja",
    page_icon="☕",
    layout="wide"
)

# =========================
# CUSTOM CSS MOCHA + CEDAR + PEANUT
# =========================
st.markdown("""
<style>
/* HALAMAN */
.stApp {
    background: #FFF5E6;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #FFE4B5, #FFD8A8);
    padding: 70px 30px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 50px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
}
.hero h1 {
    font-size: 48px;
    color: #3B270C;
    font-weight: 900;
}
.hero p {
    font-size: 20px;
    color: #4B3A26;
    max-width: 750px;
    margin: auto;
}

/* FORM CARD */
.form-card {
    background-color: #FFF8F0;
    padding: 40px;
    border-radius: 20px;
    border: 1px solid #EADFCB;
    box-shadow: 0 6px 25px rgba(0,0,0,0.08);
    margin-bottom: 50px;
}
/* FORM CARD TEXT */
.form-card h3,
.form-card label,
.form-card .stTextInput label,
.form-card .stSelectbox label {
    color: #3B270C !important;
    font-weight: 700;
}

/* SEGMENT CARD */
.segment-card {
    background-color: #F5E8D0;
    padding: 28px;
    border-radius: 20px;
    border-left: 8px solid #4B3A26;
    margin-bottom: 35px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.07);
    color: #3B270C;
}
.segment-card h2,
.segment-card p {
    color: #3B270C !important;
}

/* TOP 1 CARD */
.top1-card {
    background: linear-gradient(135deg, #795C32, #A67C52);
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: #FFFDF5;
}
.top1-card h2, 
.top1-card p {
    color: #FFFDF5 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
}
.top1-card:hover {
    transform: scale(1.07);
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
}
.top1-badge {
    background-color: #3B270C;
    color: #FFFDF5;
    font-size: 18px;
    font-weight: 900;
    padding: 10px 22px;
    border-radius: 30px;
}

/* TOP 2-5 CARD */
.recom-card {
    background-color: #FFF5E6;
    padding: 20px;
    border-radius: 16px;
    border-left: 6px solid #4B3A26;
    box-shadow: 0 5px 15px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    color: #3B270C;
    transition: transform 0.3s, box-shadow 0.3s;
}
.recom-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

/* RANK BADGE */
.rank-badge {
    background-color: #795C32;
    color: white;
    padding: 6px 14px;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 700;
}

/* BUTTON */
div.stButton > button:first-child {
    background-color: #795C32;
    color: white;
    border-radius: 10px;
    padding: 14px 28px;
    font-weight: 700;
    width: 100%;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #A67C52;
    transform: translateY(-2px);
}

/* MAP BUTTON */
.maps-link {
    background-color: #4B3A26;
    color: white !important;
    padding: 8px 16px;
    text-decoration: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.maps-link:hover {
    background-color: #3B270C;
    transform: translateY(-2px);
}

/* WATERMARK */
.watermark {
    text-align: center;
    color: #4B3A26;
    font-size: 14px;
    font-style: italic;
    font-weight: 500;
    margin-top: 50px;
    opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS & DATA
# =========================
@st.cache_resource
def load_resources():
    kmodes = joblib.load("models/kmodes_model.pkl")
    category_mappings = joblib.load("models/category_mappings.pkl")
    df, tfidf, tfidf_matrix, sbert, embeddings = build_recommender()
    return kmodes, category_mappings, df, tfidf, tfidf_matrix, sbert, embeddings

kmodes, category_mappings, df, tfidf, tfidf_matrix, sbert, embeddings = load_resources()

# =========================
# SEGMENT INFO (UPDATE) 
# =========================
segment_info = {
    0: {
        "name": "Instagrammable & Aesthetic",
        "desc": "Kamu menyukai coffee shop dengan desain visual yang unik dan estetik, sangat cocok untuk berfoto."
    },
    1: {
        "name": "Casual Coffee Drinker (Lokal)",
        "desc": "Kamu menikmati suasana santai dengan pilihan kopi yang ramah di lidah dan nyaman untuk ngobrol."
    },
    2: {
        "name": "Premium Coffee Enthusiast",
        "desc": "Kamu mengutamakan kualitas biji kopi, teknik seduh manual, dan pengalaman rasa yang serius."
    },
    3: {
        "name": "Productive Work / Study",
        "desc": "Kamu membutuhkan ruang yang tenang, kursi yang nyaman, dan suasana yang mendukung fokus bekerja."
    }
}

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <h1>☕ Coffee Shop Finder Jogja</h1>
    <p>Sistem rekomendasi coffee shop berbasis segmentasi pelanggan, analisis ulasan teks, dan machine learning.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# FORM
# =========================
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

    kebutuhan = st.text_input("Kebutuhan khusus", placeholder="wifi kencang, banyak colokan, tenang")
    area = st.text_input("Area (opsional)", placeholder="Gejayan, Jakal, UGM")
    submitted = st.form_submit_button("🔍 Cari Rekomendasi")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PROCESS & OUTPUT
# =========================
if submitted:
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

    with st.spinner("Menganalisis dan mencari rekomendasi terbaik..."):
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
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <h2 style="margin:0; font-weight:900;">{row['name']}</h2>
                        <div class="top1-badge">🏆 TOP 1</div>
                    </div>
                    <p>⭐ {row['rating']:.2f} | 📍 {row['area']}</p>
                    <p>{row['address']}</p>
                    <a href="{maps_url}" target="_blank" class="maps-link">📍 Petunjuk Lokasi</a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recom-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4>{row['name']}</h4>
                        <div class="rank-badge">#{i}</div>
                    </div>
                    <p>⭐ {row['rating']:.2f} | 📍 {row['area']}</p>
                    <p>{row['address']}</p>
                    <a href="{maps_url}" target="_blank" class="maps-link">📍 Petunjuk Lokasi</a>
                </div>
                """, unsafe_allow_html=True)

# =========================
# WATERMARK
# =========================
st.markdown('<div class="watermark">By Angel & Thania</div>', unsafe_allow_html=True)
st.caption(
    "Model menggunakan K-Modes Clustering, TF-IDF, Sentence-BERT, dan keyword-based segmentation."
)
