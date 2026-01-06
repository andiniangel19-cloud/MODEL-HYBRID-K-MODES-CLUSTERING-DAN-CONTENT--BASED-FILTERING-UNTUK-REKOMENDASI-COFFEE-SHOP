import pandas as pd
from text_preprocessing import preprocess

# Load data
df = pd.read_csv(
    "Dataset/Coffeeshop/coffee_shop_yogyakarta_reviews.csv",
    sep=';',
    quotechar='"',
    on_bad_lines='skip',
    engine='python'
)

# Ambil beberapa contoh
sample = df[['review_text']].dropna().head(5)

# Preprocessing
sample['hasil_preprocessing'] = sample['review_text'].apply(preprocess)

# Simpan ke CSV untuk dokumentasi laporan
sample.to_csv("hasil_preprocessing_contoh.csv", index=False)

print(sample)