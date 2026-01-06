from recommender import build_recommender, SEGMENT_KEYWORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Build recommender components
df, tfidf, tfidf_matrix, sbert, embeddings = build_recommender()

def recommend(user_text, segment, lokasi=None, top_k=5,
              alpha=0.4, beta=0.4, gamma=0.2, delta=0.1):

    # ===== SBERT Similarity =====
    user_emb = sbert.encode([user_text])
    sim_sbert = cosine_similarity(user_emb, embeddings).flatten()

    # ===== TF-IDF Similarity =====
    user_tf = tfidf.transform([user_text])
    sim_tf = cosine_similarity(user_tf, tfidf_matrix).flatten()

    # ===== Normalized Rating =====
    rating_norm = (df['rating'] - df['rating'].min()) / (
        df['rating'].max() - df['rating'].min() + 1e-9
    )

    # ===== Segment Keyword Boost (INI YANG KURANG SEBELUMNYA) =====
    seg_keywords = SEGMENT_KEYWORDS.get(segment, [])
    seg_match = df['clean_review'].apply(
        lambda x: int(any(k in x for k in seg_keywords))
    )

    # ===== Hybrid Score (IDENTIK NOTEBOOK) =====
    df['score'] = (
        alpha * sim_sbert +
        beta * sim_tf +
        gamma * rating_norm +
        delta * seg_match
    )

    # ===== Optional Location Filter =====
    if lokasi:
        df_filtered = df[df['area'].str.contains(lokasi, case=False, na=False)]
    else:
        df_filtered = df

    return df_filtered.sort_values('score', ascending=False).head(top_k)


# ====== CONTOH OUTPUT ======
if __name__ == "__main__":
    result = recommend(
        user_text="wifi cepat dan tempat tenang untuk kerja",
        segment="Pekerja Produktif",
        lokasi="Tugu Jogja",
        top_k=5
    )

    print(result[['name', 'area', 'rating', 'score']])
