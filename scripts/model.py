import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="movielens100k",
        user="postgres",
        password="siddhesh",
        host="localhost",
        port="5432"
    )

def load_data(conn):
    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    movies = pd.read_sql("SELECT * FROM movies", conn)
    return ratings, movies

def get_user_ratings(user_id, ratings, movies):
    user_rated = ratings[ratings["user_id"] == user_id]
    return user_rated.merge(movies, on="movie_id")[["title", "rating"]]

def recommend_movies(user_id, ratings, movies, top_n=5):
    user_item_matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    if user_id not in similarity_df.index:
        return pd.DataFrame()

    similar_users = similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    weighted_scores = np.dot(similar_users.values, user_item_matrix.loc[similar_users.index])
    scores = pd.Series(weighted_scores, index=user_item_matrix.columns)

    already_rated = user_item_matrix.loc[user_id]
    scores = scores[already_rated == 0].sort_values(ascending=False).head(top_n)

    return movies[movies["movie_id"].isin(scores.index)][["title"]]
