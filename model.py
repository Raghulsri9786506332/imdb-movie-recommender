"""
NLP Model for IMDb Movie Recommendation.
Supports recommendation by movie name OR by raw storyline input.

Pipeline:
  1. Load cleaned movie data
  2. Fit TF-IDF vectorizer on all storylines
  3. Compute cosine similarity matrix
  4. recommend_by_name()   — pick a movie from the dataset
  5. recommend_by_story()  — input your own storyline text
"""

import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/movies_clean.csv"
MODEL_DIR = "data"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")


def load_data():
    """Load cleaned movie data."""
    df = pd.read_csv(DATA_PATH)
    # Drop any rows with missing storyline
    df = df.dropna(subset=["clean_storyline", "Movie Name"])
    df = df[df["clean_storyline"].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def build_model(df):
    """
    Build TF-IDF vectors and cosine similarity matrix from the dataset.
    Returns tfidf, matrix, similarity so they can be used for queries.
    Also saves them to disk for faster reload.
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["clean_storyline"])
    similarity = cosine_similarity(matrix)

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(similarity, f)

    return tfidf, matrix, similarity


def load_or_build_model(df):
    """Load precomputed model from disk, or build & save if not found."""
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            tfidf = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            similarity = pickle.load(f)
        matrix = tfidf.transform(df["clean_storyline"])
        return tfidf, matrix, similarity

    return build_model(df)


def recommend_by_name(movie_name, df, similarity):
    """
    Recommend top 5 movies by name from the dataset.
    Returns list of dicts: [{"name": ..., "story": ...}, ...]
    """
    if movie_name not in df["Movie Name"].values:
        return []

    idx = df[df["Movie Name"] == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    results = []
    for i, score in scores:
        movie = df.iloc[i]
        results.append({
            "name": movie["Movie Name"],
            "story": movie["Storyline"],
            "similarity": round(score, 4)
        })

    return results


def recommend_by_story(user_storyline, df, tfidf, similarity):
    """
    Recommend top 5 movies based on a user-provided storyline text.
    The input text is cleaned and vectorised the same way as the dataset,
    then cosine similarity is computed against all movies.

    Returns list of dicts: [{"name": ..., "story": ..., "similarity": ...}, ...]
    """
    from preprocess import clean_text

    # Clean the input text using the same pipeline
    cleaned = clean_text(user_storyline)

    if not cleaned.strip():
        return []

    # Vectorize the input
    user_vec = tfidf.transform([cleaned])

    # Compute similarity against all movies
    scores = cosine_similarity(user_vec, tfidf.transform(df["clean_storyline"]))[0]

    # Get top 5
    top_indices = scores.argsort()[::-1][:5]

    results = []
    for idx in top_indices:
        results.append({
            "name": df.iloc[idx]["Movie Name"],
            "story": df.iloc[idx]["Storyline"],
            "similarity": round(float(scores[idx]), 4)
        })

    return results


def get_movie_list(df):
    """Return sorted list of unique movie names for dropdown."""
    return sorted(df["Movie Name"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data and building model...")
    df = load_data()
    tfidf, matrix, sim = load_or_build_model(df)
    print(f"Model ready. {len(df)} movies in dataset.")

    # Test recommend_by_name
    sample = df["Movie Name"].iloc[0] if len(df) > 0 else None
    if sample:
        print(f"\nRecommendations for '{sample}':")
        for r in recommend_by_name(sample, df, sim):
            print(f"  - {r['name']} ({r['similarity']})")

    # Test recommend_by_story
    print("\nRecommendations for sample storyline:")
    for r in recommend_by_story("A young hero discovers a hidden world", df, tfidf, sim):
        print(f"  - {r['name']} ({r['similarity']})")
