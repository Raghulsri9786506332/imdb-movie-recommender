"""
IMDb Movie Recommendation System — Streamlit App

Users can either:
  1. Type/paste a storyline and get recommendations (primary use case)
  2. Select a movie from a dropdown to see similar movies (fallback)
"""

import streamlit as st
import pandas as pd
from model import (
    load_data,
    load_or_build_model,
    recommend_by_name,
    recommend_by_story,
    get_movie_list,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data and model (cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    df = load_data()
    tfidf, matrix, similarity = load_or_build_model(df)
    return df, tfidf, matrix, similarity


df, tfidf, _, similarity = load_resources()
movie_list = get_movie_list(df)


# ---------------------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------------------
st.title("🎬 IMDb Movie Recommendation System (2024)")
st.markdown(
    """
    Get **top 5 movie recommendations** based on storyline similarity.  
    Enter a plot description below, or pick a movie to find similar ones.
    """
)

# ---------------------------------------------------------------------------
# Input method tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["✍️ Search by Storyline", "🎯 Pick a Movie"])

# ===================== TAB 1: Search by Storyline =====================
with tab1:
    st.subheader("Describe a movie plot")

    user_storyline = st.text_area(
        "Type or paste a movie storyline / plot description:",
        height=150,
        placeholder=(
            "e.g. A young wizard begins his journey at a magical school "
            "where he makes friends and enemies, facing dark forces..."
        ),
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_clicked = st.button("🔍 Recommend", type="primary", use_container_width=True)

    if search_clicked:
        if not user_storyline.strip():
            st.warning("Please enter a storyline first.")
        else:
            with st.spinner("Finding similar movies..."):
                results = recommend_by_story(user_storyline, df, tfidf, similarity)

            if not results:
                st.error("Could not find any similar movies. Try a different storyline.")
            else:
                st.success(f"Top {len(results)} recommendations based on your storyline:")
                for i, r in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"### {i}. 🎥 {r['name']}")
                        st.markdown(f"**Similarity:** {r['similarity'] * 100:.1f}%")
                        st.write(r['story'])
                        st.divider()

# ===================== TAB 2: Pick a Movie =====================
with tab2:
    st.subheader("Find movies similar to a known title")

    selected_movie = st.selectbox(
        "Choose a movie from the 2024 list:",
        movie_list,
        index=None,
        placeholder="Select a movie...",
    )

    if st.button("Get Similar Movies", type="primary", key="btn_movie"):
        if not selected_movie:
            st.warning("Please select a movie first.")
        else:
            results = recommend_by_name(selected_movie, df, similarity)

            if not results:
                st.warning(f"No similar movies found for '{selected_movie}'.")
            else:
                st.success(f"Movies similar to **{selected_movie}**:")
                for i, r in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"### {i}. 🎥 {r['name']}")
                        st.markdown(f"**Similarity:** {r['similarity'] * 100:.1f}%")
                        st.write(r['story'])
                        st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.header("📊 Dataset Info")
st.sidebar.metric("Movies in dataset", len(df))
st.sidebar.metric("Movies with storylines", df["Storyline"].notna().sum())
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with ❤️ using **Streamlit**, **Scikit-learn**, and **Selenium**"
)
