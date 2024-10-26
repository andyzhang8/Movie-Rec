'''
Idea: Use feature extraction, combine text and numeric features of movies and use TF-IDF vectorization to create comprehensive feature matrix. 
Use KMeans clustering to narrow recommendations. Calculate cosine similarity between searched movie and other movies in that cluster
'''
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import gzip
import os

DATASET_DIR = './datasets/'
BASICS_FILE = 'title.basics.tsv.gz'
RATINGS_FILE = 'title.ratings.tsv.gz'

def optimize_dataframe(df):
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def load_dataset(filename, filter_col=None, filter_value=None):
    """
    Loads a dataset file from the dataset directory, optionally filtering it.
    """
    file_path = os.path.join(DATASET_DIR, filename)
    print(f"Loading file: {file_path}")
    
    with gzip.open(file_path) as f:
        df_iter = pd.read_csv(f, sep='\t', low_memory=False, chunksize=100000)
        if filter_col:
            filtered_df = pd.concat(chunk[chunk[filter_col] == filter_value] for chunk in df_iter)
        else:
            filtered_df = pd.concat(df_iter)
            
    return optimize_dataframe(filtered_df)

# Load Datasets
def load_datasets():
    print("Loading IMDb datasets...")
    basics = load_dataset(BASICS_FILE, filter_col='titleType', filter_value='movie')
    ratings = load_dataset(RATINGS_FILE)
    return basics, ratings

# Preprocess and Clean Data
def preprocess_data(basics, ratings):
    """
    Merge basics and ratings, clean and preprocess the data.
    """
    print("Preprocessing data...")
    movies = basics[['tconst', 'primaryTitle', 'startYear', 'genres', 'runtimeMinutes']]
    
    # Data cleaning
    movies['startYear'] = pd.to_numeric(movies['startYear'], errors='coerce')
    movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors='coerce').fillna(0)
    movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.split(','))
    
    # Merge datasets
    movies = movies.merge(ratings[['tconst', 'averageRating']], on='tconst', how='left')
    movies['averageRating'] = movies['averageRating'].fillna(0)

    return movies

# Feature Extraction
def extract_features(movies):
    print("Extracting features...")
    # Extract genre metadata using TF-IDF
    movies['metadata'] = movies['genres'].apply(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    genre_features = vectorizer.fit_transform(movies['metadata'])

    numeric_features = csr_matrix(movies[['averageRating', 'runtimeMinutes']].fillna(0).values)
    return genre_features, numeric_features


# Clustering
def perform_clustering(features, n_clusters=20):
    print("Clustering movies...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(features)
    return clusters

def recommend_movies(movies, movie_title, clusters, genre_features, numeric_features, genre_weight=0.7, numeric_weight=0.3):
    try:
        movies['primaryTitle'] = movies['primaryTitle'].fillna('')

        print(f"Searching for movie title: {movie_title}")
        movie_idx = movies[movies['primaryTitle'].str.contains(movie_title, case=False, na=False)].index[0]
        referenced_movie = movies.iloc[movie_idx]

        referenced_movie_details = {
            'primaryTitle': str(referenced_movie['primaryTitle']),
            'averageRating': float(referenced_movie['averageRating']) if pd.notna(referenced_movie['averageRating']) else None,
            'startYear': int(referenced_movie['startYear']) if pd.notna(referenced_movie['startYear']) else None,
            'runtimeMinutes': int(referenced_movie['runtimeMinutes']) if pd.notna(referenced_movie['runtimeMinutes']) else None,
            'genres': ', '.join(referenced_movie['genres']) if isinstance(referenced_movie['genres'], list) else str(referenced_movie['genres'])
        }
    except IndexError:
        print(f"Movie '{movie_title}' not found.")
        return pd.DataFrame(), None

    try:
        # Identify the cluster
        movie_cluster = clusters[movie_idx]
        cluster_movies = movies[clusters == movie_cluster].copy()

        # Genre similarity
        movie_genre_feature = genre_features[movie_idx]
        genre_similarity = cosine_similarity(movie_genre_feature, genre_features[clusters == movie_cluster]).flatten()

        # Numeric similarity
        movie_numeric_feature = numeric_features[movie_idx].reshape(1, -1)
        numeric_similarity = cosine_similarity(movie_numeric_feature, numeric_features[clusters == movie_cluster]).flatten()
        combined_similarity = (genre_similarity * genre_weight) + (numeric_similarity * numeric_weight)

        cluster_movies['similarity'] = combined_similarity
        recommendations = cluster_movies.sort_values(by=['similarity', 'averageRating'], ascending=[False, False])
        recommendations = recommendations[recommendations['tconst'] != referenced_movie['tconst']]
        recommendations['genres'] = recommendations['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

        return recommendations.head(10)[['primaryTitle', 'averageRating', 'similarity', 'startYear', 'runtimeMinutes', 'genres']], referenced_movie_details
    except Exception as e:
        print(f"Error during recommendation computation: {e}")
        return pd.DataFrame(), None






# Main Processing Pipeline
def parallel_processing():
    basics, ratings = load_datasets()
    movies = preprocess_data(basics, ratings)
    genre_features, numeric_features = extract_features(movies)
    return movies, genre_features, numeric_features

if __name__ == "__main__":
    print("Starting movie recommendation system...")
    movies, genre_features, numeric_features = parallel_processing()
    clusters = perform_clustering(genre_features)  # Cluster using genre features for better grouping
    # Testing
    movie_title = "Star Wars"
    recommendations, referenced_movie = recommend_movies(movies, movie_title, clusters, genre_features, numeric_features)

    if not recommendations.empty:
        print(f"Top 10 movies similar to '{movie_title}':")
        print(recommendations)

