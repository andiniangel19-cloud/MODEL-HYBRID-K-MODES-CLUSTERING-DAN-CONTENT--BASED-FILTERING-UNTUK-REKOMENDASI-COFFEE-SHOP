import pandas as pd
import joblib
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score, davies_bouldin_score


def normalize_category(text):
    text = str(text).strip().lower()
    text = text.replace('-', ' ')
    text = ' '.join(text.split())

    mapping = {
        # Tujuan
        'Konten / Foto-Foto': 'Konten / Foto - foto',

        # Minuman
        'Kopi susu': 'Kopi Susu',

        # Tempat duduk
        'Kursi kayu': 'Kursi Kayu'
    }

    return mapping.get(text, text.title())

def load_data():
    df_q = pd.read_excel("Dataset/Kuesioner/Kuesioner DSCP (Jawaban).xlsx")

    df_q.columns = df_q.columns.str.strip()
    df_q = df_q.drop(columns=['Timestamp', 'Domisili Sekarang (contoh:  Sleman)'])

    selected_cols = [
        "Tujuan utama Anda ke coffee shop?",
        "Faktor utama yang paling memengaruhi Anda dalam memilih coffee shop",
        "Jenis minuman yang paling sering Anda pesan di coffee shop",
        "Jenis tempat duduk favorit Anda"
    ]

    df_sel = df_q[selected_cols].dropna().reset_index(drop=True)

    for col in selected_cols:
        df_sel[col] = df_sel[col].apply(normalize_category)

    df_cat = df_sel.astype('category')
    encoded = df_cat.apply(lambda x: x.cat.codes)

    category_mappings = {
        col: dict(enumerate(df_cat[col].cat.categories))
        for col in df_cat.columns
    }

    joblib.dump(category_mappings, "models/category_mappings.pkl")

    return df_sel, encoded

def evaluate_cost(encoded, k_range=range(2, 9)):
    costs = []

    for k in k_range:
        km = KModes(n_clusters=k, init='Huang', n_init=10, random_state=42)
        km.fit(encoded)
        costs.append(km.cost_)

    plt.figure(figsize=(7,4))
    plt.plot(k_range, costs, marker='o')
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Cost")
    plt.title("Elbow Method K-Modes")
    plt.grid()
    plt.show()


def train_kmodes(encoded, k=4):
    km = KModes(n_clusters=k, init='Huang', n_init=20, random_state=42)
    labels = km.fit_predict(encoded)

    sil = silhouette_score(encoded, labels, metric='hamming')
    dbi = davies_bouldin_score(encoded, labels)

    print(f"Silhouette Score (Hamming): {sil:.4f}")
    print(f"Davies-Bouldin Index: {dbi:.4f}")
    print(f"K-Modes Cost: {km.cost_:.2f}")

    joblib.dump(km, "models/kmodes_model.pkl")

    return labels, km

def profile_clusters(df_sel, labels):
    df_clustered = df_sel.copy()
    df_clustered['cluster'] = labels

    for c in sorted(df_clustered['cluster'].unique()):
        print(f"\n{'='*10} CLUSTER {c} {'='*10}")
        cluster_data = df_clustered[df_clustered['cluster'] == c]

        for col in df_sel.columns:
            print(f"\n{col}")
            print(cluster_data[col].value_counts().head(3))

if __name__ == "__main__":
    df_sel, encoded = load_data()

    # Elbow Method
    evaluate_cost(encoded)

    # Train Final Model
    labels, km = train_kmodes(encoded, k=4)

    # Profiling Cluster
    profile_clusters(df_sel, labels)
