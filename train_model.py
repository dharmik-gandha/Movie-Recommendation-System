
import pandas as pd
import ast
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

print("Datasets loaded successfully!")

# Merge datasets
# Merge datasets
credits.rename(columns={"movie_id": "id"}, inplace=True)

# Remove duplicate title column from credits
credits.drop(columns=["title"], inplace=True)

movies = movies.merge(credits, on="id")
print("Available columns:")
print(movies.columns.tolist())
# Select required columns
movies = movies[
    ["id", "title", "overview", "genres",
     "keywords", "cast", "crew",
     "release_date", "vote_average"]
]
# Convert JSON-like data into readable text
def convert_data(value):
    try:
        return " ".join(
            item["name"] for item in ast.literal_eval(value)
        )
    except:
        return ""

def get_director(value):
    try:
        crew = ast.literal_eval(value)
        for person in crew:
            if person["job"] == "Director":
                return person["name"]
        return ""
    except:
        return ""

# Process features
movies["genres"] = movies["genres"].apply(convert_data)
movies["keywords"] = movies["keywords"].apply(convert_data)

movies["cast"] = movies["cast"].apply(
    lambda x: " ".join(
        item["name"] for item in ast.literal_eval(x)[:3]
    )
)

movies["director"] = movies["crew"].apply(get_director)

movies["overview"] = movies["overview"].fillna("")

# Combine features
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["director"]
)

# Remove duplicate movies
movies.drop_duplicates(subset="title", inplace=True)
movies.reset_index(drop=True, inplace=True)

# Create TF-IDF model
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

vectors = tfidf.fit_transform(movies["tags"])

# Calculate similarity
similarity = cosine_similarity(vectors)

# Save model data
os.makedirs("model", exist_ok=True)

with open("model/movie_data.pkl", "wb") as file:
    pickle.dump(movies, file)

with open("model/similarity.pkl", "wb") as file:
    pickle.dump(similarity, file)

print("Movie Recommendation Model trained successfully!")
print("Total movies:", len(movies))
print("Model files saved in model folder.")
