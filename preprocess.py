import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("data/imdb_movies_2024.csv")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["Cleaned_Storyline"] = df["Storyline"].apply(clean_text)

# CREATE CLEAN FILE
df.to_csv("data/movies_clean.csv", index=False)

print("✅ movies_clean.csv created successfully")