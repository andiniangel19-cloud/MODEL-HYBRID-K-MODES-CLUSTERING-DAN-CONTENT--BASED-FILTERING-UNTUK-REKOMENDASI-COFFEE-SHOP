import re
from langdetect import detect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = StemmerFactory().create_stemmer()
stop_id = set(stopwords.words('indonesian'))
stop_en = set(stopwords.words('english'))
all_stop = stop_id.union(stop_en)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    try:
        lang = detect(text)
    except:
        lang = 'id'

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in all_stop and len(t) > 2]

    if lang == 'id':
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
