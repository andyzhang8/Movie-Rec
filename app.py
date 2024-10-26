from flask import Flask, request, jsonify, render_template
import pandas as pd
from movie_recommender import parallel_processing, perform_clustering, recommend_movies

app = Flask(__name__)

print("Starting Flask app...")

try:
    movies, genre_features, numeric_features = parallel_processing()
    clusters = perform_clustering(genre_features)  # Cluster based on genre features
    print("Data loaded and clustered successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")

@app.route('/')
def index():
    print("Index route accessed")
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie_title')
    
    if not movie_title:
        print("No movie title provided.")
        return jsonify({'success': False, 'message': 'No movie title provided.'}), 400

    try:
        print(f"Recommendation request received for: {movie_title}")
        recommendations, referenced_movie = recommend_movies(
            movies, movie_title, clusters, genre_features, numeric_features, genre_weight=0.7, numeric_weight=0.3
        )
        
        if recommendations.empty:
            print(f"No recommendations found for: {movie_title}")
            return jsonify({'success': False, 'message': 'Movie not found.'}), 404
        recommendations = recommendations.sort_values(by=['averageRating', 'similarity'], ascending=[False, False])

        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify({
            'success': True,
            'referencedMovie': referenced_movie,
            'recommendations': recommendations_list
        }), 200

    except Exception as e:
        print(f"Internal Server Error for '{movie_title}': {e}")
        return jsonify({'success': False, 'message': 'Internal server error.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
