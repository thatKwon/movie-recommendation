from typing import DefaultDict

import pandas as pd
import numpy as np
import pickle
import json
from collections import defaultdict

from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CfSystem:
    def __init__(self):
        self.top_k = None

        self.ratings_df = self.load_ratings()
        self.ratings_df = self.normalize_ratings(self.ratings_df)

        self.user_ratings = self.build_user_ratings_lookup(self.ratings_df)     # 영화별 유저 별점
        self.movie_ratings = self.build_movie_ratings_lookup(self.ratings_df)   # 유저별 영화 별점

        self.user_means = self.build_user_means_lookup(self.ratings_df)         # 유저의 평균 별점
        self.movie_means = self.build_movie_means_lookup(self.ratings_df)       # 영화의 평균 별점

        self.user_sim_top = None                                                # 유저 유사도 dict
        self.movie_sim_top = None                                               # 영화 유사도 dict

    @staticmethod
    def save_pickle(obj, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    @staticmethod
    def load_pickle(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def initialize(self, top_k=10, use_pickle=False):
        self.top_k = top_k

        if use_pickle:
            self.user_sim_top = self.load_pickle("cf_user_sim_top.pkl")

            self.movie_sim_top = self.load_pickle("cf_movie_sim_top.pkl")
        else:
            self.user_sim_top = self.get_user_sim(self.ratings_df, self.top_k)
            self.save_pickle(self.user_sim_top, "cf_user_sim_top.pkl")

            self.movie_sim_top = self.get_movie_sim(self.ratings_df, self.top_k)
            self.save_pickle(self.movie_sim_top, "cf_movie_sim_top.pkl")


    @staticmethod
    def load_ratings(filepath="../../data/ratings.ndjson", limit=None) -> DataFrame:
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                obj = json.loads(line)
                records.append({
                    'user': int(obj['user']),
                    'movie': int(obj['movie']),
                    'rate': float(obj['rate'])
                })
        return pd.DataFrame(records, columns=['user', 'movie', 'rate'])

    @staticmethod
    def normalize_ratings(df) -> DataFrame:
        df = df.groupby(['user', 'movie'], as_index=False)['rate'].mean()
        df['rate_mean'] = df.groupby('user')['rate'].transform('mean')
        df['rate_centered'] = df['rate'] - df['rate_mean']

        # df['rate_mean_movie'] = df.groupby('movie')['rate'].transform('mean')
        # df['rate_centered_movie'] = df['rate'] - df['rate_mean_movie']
        return df

    @staticmethod
    def build_user_means_lookup(df) -> dict:
        return df.groupby('user')['rate'].mean().to_dict()
    @staticmethod
    def build_movie_means_lookup(df) -> dict:
        return df.groupby('movie')['rate'].mean().to_dict()

    @staticmethod
    def build_user_ratings_lookup(df) -> DefaultDict[int, dict[int, float]]:
        ratings = defaultdict(dict)
        for row in df[['user', 'movie', 'rate', 'rate_centered']].itertuples(index=False):
            ratings[row.movie][row.user] = row.rate_centered
        return ratings
    @staticmethod
    def build_movie_ratings_lookup(df) -> DefaultDict[int, dict[int, float]]:
        ratings = defaultdict(dict)
        for row in df[['user', 'movie', 'rate', 'rate_centered']].itertuples(index=False):
            ratings[row.user][row.movie] = row.rate_centered
        return ratings

    @staticmethod
    def get_user_sim(df, top_k) -> dict:
        ratings_mat = df.pivot(index='user', columns='movie', values='rate_centered').fillna(0.0)
        user_ids = ratings_mat.index.to_numpy()
        sparse_ratings_mat = csr_matrix(ratings_mat.values)

        sim_sparse = cosine_similarity(sparse_ratings_mat, dense_output=False)

        sim_top = {}
        for i, uid in enumerate(user_ids):
            row = sim_sparse[i].toarray().flatten()
            row[i] = -1

            top_i = np.argpartition(row, -top_k)[-top_k:]
            top_i = top_i[np.argsort(-row[top_i])]

            sim_top[uid] = {
                "neighbors": user_ids[top_i],
                "sims": row[top_i]
            }

        return sim_top

    @staticmethod
    def get_movie_sim(df, top_k) -> dict:
        ratings_mat = df.pivot(index='movie', columns='user', values='rate_centered').fillna(0.0)
        movie_ids = ratings_mat.index.to_numpy()
        sparse_ratings_mat = csr_matrix(ratings_mat.values)

        sim_sparse = cosine_similarity(sparse_ratings_mat, dense_output=False)

        sim_top = {}
        for i, mid in enumerate(movie_ids):
            row = sim_sparse[i].toarray().flatten()
            row[i] = -1  # 자기 자신 제외

            top_i = np.argpartition(row, -top_k)[-top_k:]
            top_i = top_i[np.argsort(-row[top_i])]

            sim_top[mid] = {
                "neighbors": movie_ids[top_i],
                "sims": row[top_i]
            }
        return sim_top

    @staticmethod
    def get_user_predicted_rating(user_id, movie_id, sim_top, user_means, ratings) -> float:
        if movie_id not in ratings:
            return user_means[user_id]

        neighbors = sim_top[user_id]['neighbors']
        sims = sim_top[user_id]['sims']

        movie_ratings = ratings[movie_id]
        numerator = 0.0
        denominator = 0.0

        for other_user, sim in zip(neighbors, sims):
            r = movie_ratings.get(other_user)
            if r is None:
                continue
            numerator += sim * r
            denominator += abs(sim)

        if denominator == 0.0:
            return user_means[user_id]

        predicted_rating = user_means[user_id] + numerator / denominator
        return max(1.0, min(10.0, predicted_rating))

    @staticmethod
    def get_movie_predicted_rating(user_id, movie_id, sim_top, user_means, movie_means, ratings) -> float:
        if movie_id not in sim_top:
            return user_means[user_id]

        neighbors = sim_top[movie_id]['neighbors']
        sims = sim_top[movie_id]['sims']

        user_ratings = ratings[user_id]
        numerator = 0.0
        denominator = 0.0

        for other_movie, sim in zip(neighbors, sims):
            r = user_ratings.get(other_movie)
            if r is None:
                continue
            numerator += sim * r
            denominator += abs(sim)

        if denominator == 0.0:
            return user_means[user_id]

        predicted_rating = user_means[user_id] + numerator / denominator
        return predicted_rating
        # return max(1.0, min(10.0, predicted_rating))


    def get_recommendations(self, user_id, top_n=10, movie_limit=300) -> str:
        if user_id not in self.user_sim_top:
            return json.dumps({
                "user_id": user_id,
                "recommendations": []
            })

        neighbors = self.user_sim_top[user_id]['neighbors']

        watched_by_neighbors = self.ratings_df[self.ratings_df['user'].isin(neighbors)]
        movie_count = watched_by_neighbors['movie'].value_counts()
        candidate_movies = movie_count.head(movie_limit).index.tolist()

        watched_by_user = set(self.ratings_df[self.ratings_df['user'] == user_id]['movie'])
        candidate_movies = [m for m in candidate_movies if m not in watched_by_user]

        recommendations = []
        for movie_id in candidate_movies:
            predicted_rating = self.get_user_predicted_rating(user_id, movie_id, self.user_sim_top, self.user_means,
                                                              self.user_ratings)
            recommendations.append((movie_id, predicted_rating))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:top_n]

        # jsonfy
        recommendations_json = json.dumps({
            "user_id": user_id,
            "recommendations": [
                {"movie_id": mid, "predicted_rating": round(score, 1)}
                for mid, score in recommendations
            ]
        })

        return recommendations_json

    def get_movie_recommendations(self, user_id, top_n=10) -> str:
        if user_id not in self.user_sim_top:
            return json.dumps({
                "user_id": user_id,
                "recommendations": []
            })

        watched_by_user = set(self.ratings_df[self.ratings_df['user'] == user_id]['movie'])

        movie_rating = self.movie_ratings[user_id]
        liked_by_user = [m for m, r in movie_rating.items() if r > 0]

        candidate_movies = set()
        for m in liked_by_user:
            if m in self.movie_sim_top:
                candidate_movies.update(self.movie_sim_top[m]['neighbors'])
        candidate_movies = [m for m in candidate_movies if m not in watched_by_user]

        movie_count = self.ratings_df['movie'].value_counts()

        recommendations = []
        for movie_id in candidate_movies:
            if movie_count.get(movie_id, 0) < 20:
                continue

            predicted_rating = self.get_movie_predicted_rating(user_id, movie_id, self.movie_sim_top, self.user_means, self.movie_means, self.movie_ratings)
            recommendations.append((movie_id, predicted_rating))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        recommendations = recommendations[:top_n]

        return json.dumps({
            "user_id": user_id,
            "recommendations": [
                {"movie_id": int(mid), "predicted_rating": float(round(score, 1))}
                for mid, score in recommendations
            ]
        })


if __name__ == '__main__':
    USE_PICKLE = True
    TOP_K = 50
    TOP_N = 10

    cf = CfSystem()
    cf.initialize(top_k=TOP_K, use_pickle=USE_PICKLE)

    def load_movies(filepath="../../data/movies.ndjson", limit=None):
        movies = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                obj = json.loads(line)
                movie_id = int(obj['_id'])
                movies[movie_id] = obj
        return movies

    def display_movie_info(movies_df, recommendations_json):
        data = json.loads(recommendations_json)
        user_id = data.get("user_id")
        recommendations = data.get("recommendations", [])

        print(f"User ID: {user_id}")
        print(f"{'Movie ID':<10} | {'Title':<40} | {'Predicted Rating'}")
        print("-" * 70)

        for item in recommendations:
            mid = item["movie_id"]
            rating = item["predicted_rating"]
            title = movies_df.get(mid, {}).get("title", "Unknown")
            print(f"{mid:<10} | {title:<40} | {rating}")
        print("=" * 70)

    movies_df = load_movies()

    while True:
        user_input = input("Enter user id (q to quit): ")
        if user_input == 'q':
            print("Exiting...")
            break

        user_id = int(user_input)
        recommendations_json = cf.get_recommendations(user_id, TOP_N)
        display_movie_info(movies_df, recommendations_json)

        # recommendations_json = cf.get_movie_recommendations(user_id)
        # display_movie_info(movies_df, recommendations_json)