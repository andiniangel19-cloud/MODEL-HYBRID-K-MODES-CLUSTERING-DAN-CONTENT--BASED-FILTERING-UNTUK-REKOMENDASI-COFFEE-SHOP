import joblib
import pandas as pd

from recommender import build_recommender
from main import recommend

segment_mapping = {
    0: "Instagrammable & Aesthetic",
    1: "Casual Coffee Drinker (Lokal)",
    2: "Premium Coffee Enthusiast",
    3: "Productive Work / Study"
}

km = joblib.load("models/kmodes_model.pkl")
category_mappings = joblib.load("models/category_mappings.pkl")

def ask_question(question, options):
    print(f"\n{question}")
    for i, opt in enumerate(options):
        print(f"{i+1}. {opt}")

    choice = int(input("Pilih nomor: ")) - 1
    return options[choice]

def get_user_input():
    user_input = {}

    for col, categories in category_mappings.items():
        options = list(categories.values())
        answer = ask_question(col, options)
        user_input[col] = answer

    return user_input

def predict_cluster(user_input):
    encoded = []

    for col, categories in category_mappings.items():
        value = user_input[col]
        code = list(categories.values()).index(value)
        encoded.append(code)

    user_df = pd.DataFrame([encoded], columns=category_mappings.keys())
    cluster = km.predict(user_df)[0]

    return cluster

def simulate():
    print("\n===== SIMULASI SISTEM REKOMENDASI COFFEE SHOP =====")

    user_input = get_user_input()
    cluster = predict_cluster(user_input)
    segment = segment_mapping[cluster]

    print("\n====================================")
    print(f"Anda termasuk dalam SEGMENT: {segment}")
    print("====================================")

    # Input tambahan untuk recommender
    user_text = input("\nApa kebutuhan utama Anda? (contoh: wifi cepat dan tempat tenang): ")
    lokasi = input("Area yang diinginkan (kosongkan jika bebas): ")

    results = recommend(
        user_text=user_text,
        segment=segment,
        lokasi=lokasi if lokasi else None,
        top_k=5
    )

    print("\n=== REKOMENDASI COFFEE SHOP ===")
    print(results[['name', 'area', 'rating', 'score']])

if __name__ == "__main__":
    simulate()
