import pandas as pd
from src.text_preprocessing import preprocess

df = pd.read_csv(
    "Dataset/Coffeeshop/coffee_shop_yogyakarta_reviews.csv",
    sep=';',
    quotechar='"',
    on_bad_lines='skip',
    engine='python'
)

df.columns = df.columns.str.strip()
df = df.dropna(subset=['review_text','name'])

df['clean_review'] = df['review_text'].apply(preprocess)

print(df[['name', 'review_text', 'clean_review']].sample(5))

