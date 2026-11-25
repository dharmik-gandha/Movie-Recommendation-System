
import streamlit as st
import pandas as pd
import pickle
import requests

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #101014;
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ff4b5c;
    margin-bottom: 0;
}

.subtitle {
    color: #b5b5bd;
    font-size: 17px;
    margin-bottom: 25px;
}

.movie-card {
    background-color: #1c1c24;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #30303a;
}

.movie-title {
    color: white;
    font-size: 17px;
    font-weight: bold;
    min-height: 45px;
}

.movie-info {
    color: #b5b5bd;
    font-size: 14px;
}

.stButton > button {
    background-color: #ff4b5c;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #e63b4c;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_data
def load_data():

    with open("model/movie_data.pkl", "rb") as file:
        movies = pickle.load(file)

    with open("model/similarity.pkl", "rb") as file:
        similarity = pickle.load(file)

    return movies, similarity


movies, similarity = load_data()


# ---------------- TMDB POSTER FUNCTION ----------------

@st.cache_data(ttl=86400)
def get_movie_poster(movie_id):

    try:
        api_key = st.secrets["TMDB_API_KEY"]

        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"

        response = requests.get(
            url,
            params={"api_key": api_key},
            timeout=10
        )

        if response.status_code == 200:

            data = response.json()
            poster_path = data.get("poster_path")

            if poster_path:
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

    except Exception:
        pass

    return None


# ---------------- RECOMMENDATION FUNCTION ----------------

def recommend(movie_name):

    matches = movies[
        movies["title"].str.lower() == movie_name.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    distances = list(enumerate(similarity[movie_index]))

    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    recommended_movies = []

    for index, score in distances:
        recommended_movies.append(movies.iloc[index])

    return pd.DataFrame(recommended_movies)


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover movies you will love. '
    'Find your next favourite film.</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- SEARCH SECTION ----------------

st.subheader("🔎 Find Your Next Movie")

movie_list = sorted(movies["title"].dropna().unique())

selected_movie = st.selectbox(
    "Search or select a movie",
    movie_list,
    index=None,
    placeholder="Type a movie name..."
)

if st.button("Find Similar Movies"):

    if selected_movie:

        st.session_state["selected_movie"] = selected_movie

        st.session_state["recommendations"] = recommend(
            selected_movie
        )

    else:
        st.warning("Please select a movie first.")


# ---------------- SELECTED MOVIE ----------------

if "selected_movie" in st.session_state:

    selected = st.session_state["selected_movie"]

    movie_details = movies[
        movies["title"] == selected
    ].iloc[0]

    st.divider()

    st.subheader("🎥 Selected Movie")

    col1, col2 = st.columns([1, 3])

    with col1:

        poster = get_movie_poster(movie_details["id"])

        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.info("Poster not available")

    with col2:

        st.markdown(f"### {movie_details['title']}")

        rating = movie_details["vote_average"]

        st.write(f"⭐ Rating: {rating}/10")

        release = movie_details["release_date"]

        if pd.notna(release) and str(release).strip():
            st.write(f"📅 Release Date: {release}")

        st.write("**Overview:**")

        st.write(
            movie_details["overview"]
            if movie_details["overview"]
            else "Overview not available."
        )


    # ---------------- RECOMMENDATIONS ----------------

    st.divider()

    st.subheader("✨ Recommended Movies For You")

    recommendations = st.session_state["recommendations"]

    if not recommendations.empty:

        cols = st.columns(5)

        for i, (_, movie) in enumerate(
            recommendations.iterrows()
        ):

            with cols[i % 5]:

                poster = get_movie_poster(movie["id"])

                st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:220px; display:flex; '
                        'align-items:center; justify-content:center; '
                        'font-size:45px;">🎬</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f'<div class="movie-title">{movie["title"]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="movie-info">⭐ {movie["vote_average"]}/10</div>',
                    unsafe_allow_html=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("No recommendations found.")


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center style='color:#888;'>"
    "CineMatch | Powered by Machine Learning | Movie data and images by TMDB"
    "</center>",
    unsafe_allow_html=True
)
