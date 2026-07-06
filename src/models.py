import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
# pyrefly: ignore [missing-import]
from implicit.als import AlternatingLeastSquares
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class PopularityRecommender:
    """
    Recommends products by general popularity (total interaction score or sales).
    Supports geographic state-level filtering.
    """
    def __init__(self):
        self.global_popular = []
        self.state_popular = {}

    def fit(self, df):
        print("Fitting PopularityRecommender...")
        # Global popularity
        global_agg = df.groupby("product_id")["interaction_score"].sum().sort_values(ascending=False)
        self.global_popular = global_agg.index.tolist()

        # State-level popularity
        state_groups = df.groupby(["customer_state", "product_id"])["interaction_score"].sum().reset_index()
        for state in state_groups["customer_state"].unique():
            state_df = state_groups[state_groups["customer_state"] == state]
            state_df = state_df.sort_values(by="interaction_score", ascending=False)
            self.state_popular[state] = state_df["product_id"].tolist()

    def recommend(self, state=None, n=10):
        if state and state in self.state_popular:
            state_list = self.state_popular[state]
            if len(state_list) >= n:
                return state_list[:n]
            # fallback with global popular items if state list is short
            fallback_list = state_list + [x for x in self.global_popular if x not in state_list]
            return fallback_list[:n]
        return self.global_popular[:n]


class ALSRecommender:
    """
    Alternating Least Squares Collaborative Filtering recommender for warm users.
    """
    def __init__(self, factors=100, regularization=0.1, iterations=20):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = None

    def fit(self, interaction_df):
        print("Fitting ALSRecommender...")
        # Create sparse user-product interaction matrix
        # Matrix shape: (num_users, num_products)
        num_users = interaction_df["user_id_enc"].max() + 1
        num_products = interaction_df["product_id_enc"].max() + 1

        self.interaction_matrix = csr_matrix(
            (interaction_df["interaction_score"], 
             (interaction_df["user_id_enc"], interaction_df["product_id_enc"])),
            shape=(num_users, num_products)
        )

        # Train implicit ALS model
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42
        )
        self.model.fit(self.interaction_matrix)

    def recommend(self, user_id_enc, n=10):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        
        # recommend returns (ids, scores)
        ids, _ = self.model.recommend(
            user_id_enc, 
            self.interaction_matrix[user_id_enc], 
            N=n
        )
        return list(ids)


class ContentRecommender:
    """
    Content-Based recommender using product category, review score, and price bucket.
    """
    def __init__(self):
        self.vectorizer = None
        self.nn_model = None
        self.product_vectors = None
        self.product_id_to_row = {}
        self.row_to_product_id = {}

    def _get_price_bucket(self, price):
        if price < 50:
            return "Low"
        elif price <= 200:
            return "Mid"
        else:
            return "High"

    def fit(self, product_profiles):
        print("Fitting ContentRecommender...")
        profiles = product_profiles.copy()
        
        # Build text description column
        profiles["price_bucket"] = profiles["avg_price"].apply(self._get_price_bucket)
        profiles["text"] = (
            profiles["category"].fillna("") + " " + 
            profiles["price_bucket"] + " " + 
            profiles["avg_review_score"].round(1).astype(str)
        )

        self.row_to_product_id = profiles["product_id"].to_dict()
        self.product_id_to_row = {v: k for k, v in self.row_to_product_id.items()}

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.product_vectors = self.vectorizer.fit_transform(profiles["text"])

        self.nn_model = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
        self.nn_model.fit(self.product_vectors)

    def recommend_similar(self, product_id, n=10):
        if product_id not in self.product_id_to_row:
            return []
        
        row_idx = self.product_id_to_row[product_id]
        query_vector = self.product_vectors[row_idx]
        
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=n+1)
        
        # Flatten indices and discard the first one (which is the query product itself)
        sim_indices = indices.flatten()[1:]
        
        similar_ids = [self.row_to_product_id[idx] for idx in sim_indices]
        return similar_ids


def save_models(save_dir, popularity_rec, als_rec, content_rec, user_encoder, product_encoder, user_profiles, product_profiles, user_history=None):
    """
    Serializes models and encoders to disk.
    """
    print(f"Saving models to directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(popularity_rec, os.path.join(save_dir, "popularity_rec.joblib"))
    joblib.dump(als_rec, os.path.join(save_dir, "als_rec.joblib"))
    joblib.dump(content_rec, os.path.join(save_dir, "content_rec.joblib"))
    joblib.dump(user_encoder, os.path.join(save_dir, "user_encoder.joblib"))
    joblib.dump(product_encoder, os.path.join(save_dir, "product_encoder.joblib"))
    joblib.dump(user_profiles, os.path.join(save_dir, "user_profiles.joblib"))
    joblib.dump(product_profiles, os.path.join(save_dir, "product_profiles.joblib"))
    
    if user_history is not None:
        joblib.dump(user_history, os.path.join(save_dir, "user_history.joblib"))
        
    print("Saving completed successfully.")


def load_models(save_dir):
    """
    Deserializes models and encoders from disk.
    """
    print(f"Loading models from directory: {save_dir}")
    popularity_rec = joblib.load(os.path.join(save_dir, "popularity_rec.joblib"))
    als_rec = joblib.load(os.path.join(save_dir, "als_rec.joblib"))
    content_rec = joblib.load(os.path.join(save_dir, "content_rec.joblib"))
    user_encoder = joblib.load(os.path.join(save_dir, "user_encoder.joblib"))
    product_encoder = joblib.load(os.path.join(save_dir, "product_encoder.joblib"))
    user_profiles = joblib.load(os.path.join(save_dir, "user_profiles.joblib"))
    product_profiles = joblib.load(os.path.join(save_dir, "product_profiles.joblib"))
    
    return popularity_rec, als_rec, content_rec, user_encoder, product_encoder, user_profiles, product_profiles

