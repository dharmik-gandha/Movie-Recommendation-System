
# 🎬 CineMatch – Movie Recommendation System

CineMatch is a Machine Learning-based movie recommendation website that suggests similar movies based on movie content such as genres, keywords, cast, director, and overview.

## 🚀 Features

- Content-based movie recommendations
- TF-IDF Vectorization
- Cosine Similarity
- Movie search functionality
- Movie posters using TMDB API
- Movie ratings and release dates
- Interactive Streamlit web interface

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- TMDB API

## 📂 Project Structure

```text
Movie-Recommendation-System/
├── dataset/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── model/
├── .streamlit/
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Movie-Recommendation-System.git
```

2. Open the project folder:

```bash
cd Movie-Recommendation-System
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Generate the recommendation model:

```bash
python train_model.py
```

5. Create `.streamlit/secrets.toml` and add your TMDB API key:

```toml
TMDB_API_KEY = "YOUR_API_KEY"
```

6. Run the application:

```bash
streamlit run app.py
```

## 🤖 Machine Learning Approach

CineMatch uses TF-IDF Vectorization to convert movie information into numerical features and Cosine Similarity to identify movies with similar content.

## 🎞️ Dataset

TMDB 5000 Movies Dataset:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## 🖼️ Movie Data and Images

Movie poster images and related movie information are retrieved using the TMDB API.

This product uses the TMDB API but is not endorsed or certified by TMDB.

## 👨‍💻 Author

Dharmik Gandha
