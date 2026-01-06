import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =========================
# PATH AMAN (DEPLOY SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "Coffeeshop",
    "coffee_shop_yogyakarta_reviews.csv"
)

# =========================
# IMPORT INTERNAL (WAJIB src.)
# =========================
try:
    # Saat dijalankan via streamlit / deploy
    from src.text_preprocessing import preprocess
except ModuleNotFoundError:
    # Saat dijalankan langsung (python src/recommender.py)
    from text_preprocessing import preprocess



# =========================
# SEGMENT KEYWORDS
# =========================
SEGMENT_KEYWORDS = {
    "Instagrammable & Aesthetic": [
        "estetik", "foto", "instagramable", "bagus", "view",
        "cantik", "interior", "desain", "aesthetic"
    ],
    "Casual Coffee Drinker (Lokal)": [
        "kopi susu", "enak", "mantap", "kursi kayu", "lokal", "rasa", "kopi"
    ],
    "Premium Coffee Enthusiast": [
        "latte", "cappuccino", "espresso", "premium",
        "barista", "specialty", "sofa", "relax"
    ],
    "Productive Work / Study": [
        "wifi", "nugas", "kerja", "colokan",
        "laptop", "tenang", "kondusif", "meja luas"
    ]
}


# =========================
# BUILD RECOMMENDER
# =========================
def build_recommender(csv_path=CSV_PATH):
    df = pd.read_csv(
        csv_path,
        sep=";",
        quotechar='"',
        on_bad_lines="skip",
        engine="python"
    )

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["review_text", "name"]).reset_index(drop=True)

    df["clean_review"] = df["review_text"].apply(preprocess)

    df_agg = df.groupby("name").agg({
        "clean_review": " ".join,
        "rating": "mean",
        "area": "first",
        "address": "first"
    }).reset_index()

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df_agg["clean_review"])

    sbert = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embeddings = sbert.encode(
        df_agg["clean_review"].tolist(),
        show_progress_bar=False
    )

    return df_agg, tfidf, tfidf_matrix, sbert, embeddings


# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(
    df,
    tfidf,
    tfidf_matrix,
    sbert,
    embeddings,
    user_text="",
    segment=None,
    lokasi=None,
    alpha=0.35,
    beta=0.25,
    gamma=0.20,
    delta=0.20,
    top_k=5
):
    df = df.copy()

    if lokasi:
        df = df[df["area"].str.contains(lokasi, case=False, na=False)]
        if df.empty:
            return df

        idx = df.index
        tfidf_matrix = tfidf_matrix[idx]
        embeddings = embeddings[idx]

    if user_text.strip():
        user_emb = sbert.encode([user_text])
        sim_sbert = cosine_similarity(user_emb, embeddings).flatten()

        user_tf = tfidf.transform([user_text])
        sim_tfidf = cosine_similarity(user_tf, tfidf_matrix).flatten()
    else:
        sim_sbert = np.zeros(len(df))
        sim_tfidf = np.zeros(len(df))

    rating_norm = (df["rating"] - df["rating"].min()) / (
        df["rating"].max() - df["rating"].min() + 1e-9
    )

    seg_kw = SEGMENT_KEYWORDS.get(segment, [])
    if seg_kw:
        seg_match = df["clean_review"].apply(
            lambda x: sum(1 for k in seg_kw if k in x)
        )
        seg_match = seg_match / (len(seg_kw) + 1e-9)
    else:
        seg_match = np.zeros(len(df))

    df["score"] = (
        alpha * sim_sbert +
        beta * sim_tfidf +
        gamma * rating_norm +
        delta * seg_match
    )

    return df.sort_values("score", ascending=False).head(top_k)
