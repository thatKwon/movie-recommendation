import pandas as pd
import numpy as np
import pickle
import json
from collections import defaultdict
from math import sqrt
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

def load_ndjson(filepath, limit=None):
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

def load_movies_ndjson(filepath, limit=None):
    movie_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            obj = json.loads(line)
            movie_id = int(obj['_id'])
            movie_dict[movie_id] = obj
    return movie_dict

def display_movie_info(movie_dict, recommendations):
    print(f"{'Movie ID':<10} | {'Title':<40} | {'Pred Rating'}")
    print("-" * 70)

    for item in recommendations:  # item은 dict
        mid = item["movie_id"]
        rating = item["predicted_rating"]

        info = movie_dict.get(mid, {})
        title = info.get("title", "Unknown")

        print(f"{mid:<10} | {title:<40} | {rating}")


def get_user_means(df):
    return df.groupby('user')['rate'].mean().to_dict()

def preprocess_df(df):
    df = df.groupby(['user', 'movie'], as_index=False)['rate'].mean()
    df['rate_mean'] = df.groupby('user')['rate'].transform('mean')
    df['rate_centered'] = df['rate'] - df['rate_mean']
    return df

def get_movie_ratings_lookup(df):
    movie_ratings = defaultdict(dict)
    for row in df[['user', 'movie', 'rate', 'rate_centered']].itertuples(index=False):
        movie_ratings[row.movie][row.user] = row.rate_centered
    return movie_ratings

def get_user_sim_topk(df, top_k=50):
    user_movie_mat = df.pivot(index='user', columns='movie', values='rate_centered').fillna(0.0)
    user_ids = user_movie_mat.index.to_numpy()
    sparse_mat = csr_matrix(user_movie_mat.values)

    user_sim_sparse = cosine_similarity(sparse_mat, dense_output=False)

    sim_topk = {}
    for idx, uid in enumerate(user_ids):
        row = user_sim_sparse[idx].toarray().flatten()
        row[idx] = -1

        topk_idx = np.argpartition(row, -top_k)[-top_k:]
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]

        sim_topk[uid] = {
            "neighbors": user_ids[topk_idx],
            "sims": row[topk_idx]
        }

    return sim_topk

def predict_rating_user_topk(user_id, movie_id, df, sim_topk, user_means, movie_ratings):
    neighbors = sim_topk[user_id]['neighbors']
    sims = sim_topk[user_id]['sims']

    if movie_id not in movie_ratings:
        return user_means[user_id]

    movie_dict = movie_ratings[movie_id]

    numerator = 0.0
    denominator = 0.0

    for other_user, sim in zip(neighbors, sims):
        r = movie_dict.get(other_user)
        if r is None:
            continue
        numerator += sim * r
        denominator += abs(sim)

    if denominator == 0.0:
        return user_means[user_id]

    pred = user_means[user_id] + numerator / denominator
    return max(1.0, min(10.0, pred))

def recommend_movies_topk(user_id, df, sim_topk, user_means, movie_ratings, top_k=50, top_n=10, movie_limit=300):
    neighbors = sim_topk[user_id]['neighbors']

    watched_by_neighbors = df[df['user'].isin(neighbors)]
    movie_count = watched_by_neighbors['movie'].value_counts()

    candidate_movies = movie_count.head(movie_limit).index.tolist()

    watched_by_user = set(df[df['user'] == user_id]['movie'])
    candidate_movies = [m for m in candidate_movies if m not in watched_by_user]

    recommendations = []
    for movie_id in candidate_movies:
        pred = predict_rating_user_topk(user_id, movie_id, df, sim_topk, user_means, movie_ratings)
        recommendations.append((movie_id, pred))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

USE_PICKLE = True
TOP_K = 50
TOP_N = 10

if __name__ == '__main__':
    # init
    filepath_ratings = "ratings.ndjson"

    df = load_ndjson(filepath_ratings, 1000000)
    df = preprocess_df(df)

    movies_info = "movies.ndjson"
    movie_dict = load_movies_ndjson(movies_info)

    if USE_PICKLE:
        user_means = load_pickle("cf_user_means.pkl")

        user_sim_topk = load_pickle("cf_user_sim_topk.pkl")
    else:
        user_means = get_user_means(df)
        save_pickle(user_means, "cf_user_means.pkl")

        user_sim_topk = get_user_sim_topk(df, TOP_K)
        save_pickle(user_sim_topk, "cf_user_sim_topk.pkl")

    movie_ratings = get_movie_ratings_lookup(df)

    # display
    while True:
        user_input = input("Enter user id (q to quit): ")
        if user_input == 'q':
            print("Exiting...")
            break

        user_id = int(user_input)
        if user_id not in user_sim_topk:
            print(f"User {user_id} not found...")
            continue

        print(f"Top {TOP_K} neighbors:")
        for nb, sim in zip(user_sim_topk[user_id]['neighbors'][:TOP_N], user_sim_topk[user_id]['sims'][:TOP_N]):
            print(f"  user {nb}: sim={sim:.4f}")

        print("===")

        recommendations = recommend_movies_topk(user_id, df, user_sim_topk, user_means, movie_ratings, TOP_K, TOP_N)
        print(f"Recommendations:")
        for mid, score in recommendations:
            print(f"  movie {mid}: predicted={score:.3f}")
        print("===")

        # json
        json_result = json.dumps({
            "user_id": user_id,
            "recommendations": [
                {"movie_id": mid, "predicted_rating": round(score, 1)}
                for mid, score in recommendations
            ]
        })
        # print(json_result)
        print("===")

        recommendations_dict = [
            {"movie_id": mid, "predicted_rating": round(score, 1)}
            for mid, score in recommendations
        ]
        display_movie_info(movie_dict, recommendations_dict)
        print("===")