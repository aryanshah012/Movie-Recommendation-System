from flask import Flask, render_template, request
import os
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


movies = pickle.load(open("movie_list.pkl", "rb"))
vectors = pickle.load(open("movie_vectors.pkl", "rb"))  # sparse matrix, ~1.6MB

API_KEY = "6debf3ee8c45d785666185ca0e5bd059"



def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()

    if data.get("poster_path"):
        return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
    return ""


def recommend(movie_name):
    index = movies[movies["title"] == movie_name].index[0]

    # Compute similarity of this one movie against all others on the fly.
    # This avoids ever loading/storing the full NxN similarity matrix.
    distances = cosine_similarity(vectors[index], vectors).flatten()

    recommendations = []

    for i in sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]:
        movie = movies.iloc[i[0]]

        recommendations.append({
            "title": movie["title"],
            "poster": fetch_poster(movie["movie_id"])
        })

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    selected_movie = ""
    recommendations = []
    error = None

    if request.method == "POST":
        selected_movie = request.form.get("movie", "").strip()
        if selected_movie:
            matched = movies[movies["title"] == selected_movie]
            if not matched.empty:
                recommendations = recommend(selected_movie)
            else:
                error = "Movie not found. Please type a valid title."
        else:
            error = "Please enter a movie name."

    return render_template(
        "index.html",
        movie_list=movies["title"].tolist(),
        recommendations=recommendations,
        selected_movie=selected_movie,
        error=error,
    )


if __name__ == "__main__":
    print("Starting Flask...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
