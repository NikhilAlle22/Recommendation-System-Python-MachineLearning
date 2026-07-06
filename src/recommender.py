import os
import joblib
import pandas as pd
from src.models import load_models

class RecommenderSystem:
    """
    Orchestration layer that integrates all models and implements segment-based routing
    and business rule filtering (e.g. deduplication of historically purchased items).
    """
    def __init__(self, models_dir):
        self.models_dir = models_dir
        
        # Load serialized components
        (self.popularity_rec, 
         self.als_rec, 
         self.content_rec, 
         self.user_encoder, 
         self.product_encoder, 
         self.user_profiles, 
         self.product_profiles) = load_models(models_dir)
        
        # Load user history mapping for filtering
        history_path = os.path.join(models_dir, "user_history.joblib")
        if os.path.exists(history_path):
            self.user_history = joblib.load(history_path)
        else:
            self.user_history = {}
            print("Warning: user_history.joblib not found. Purchase filtering is disabled.")
            
        # Convert product profiles to a quick-lookup dataframe indexed by product_id
        self.product_db = self.product_profiles.set_index("product_id")

    def _enrich_metadata(self, product_ids):
        """
        Retrieves pricing, rating, and category metadata for a list of product IDs.
        """
        enriched = []
        for pid in product_ids:
            if pid in self.product_db.index:
                row = self.product_db.loc[pid]
                enriched.append({
                    "product_id": pid,
                    "category": str(row["category"]),
                    "price": float(row["avg_price"]),
                    "review_score": float(row["avg_review_score"]),
                    "popularity_sales": int(row["total_purchases"])
                })
            else:
                enriched.append({
                    "product_id": pid,
                    "category": "unknown",
                    "price": 0.0,
                    "review_score": 0.0,
                    "popularity_sales": 0
                })
        return enriched

    def recommend_for_user(self, user_id, state=None, n=10):
        """
        Determines user type (Cold Start vs. Warm) and returns top-n recommended products
        enriched with product catalog metadata.
        """
        # 1. Cold Start: User has no history or is not found in user encoder
        if user_id not in self.user_encoder.classes_:
            print(f"Cold-start detected for user '{user_id}'. Using popularity model.")
            recommendations = self.popularity_rec.recommend(state=state, n=n)
            return self._enrich_metadata(recommendations)

        # 2. Warm User: Retrieve collaborative candidates from ALS
        print(f"Warm-user detected for user '{user_id}'. Using Implicit ALS.")
        
        # Encode user string ID to internal integer code
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Query 2x the requested candidates to allow filtering out already purchased products
        candidates_raw = self.als_rec.recommend(user_idx, n=n * 2)
        
        # Decode candidates back to string product IDs
        candidates = self.product_encoder.inverse_transform(candidates_raw).tolist()

        # Filter out items already purchased by the user
        history = self.user_history.get(user_id, set())
        recommendations = [item for item in candidates if item not in history]
        
        # If filtering left us with fewer than n recommendations, fill with popularity items
        if len(recommendations) < n:
            popularity_fill = self.popularity_rec.recommend(state=state, n=n)
            for item in popularity_fill:
                if item not in recommendations and item not in history:
                    recommendations.append(item)
                if len(recommendations) >= n:
                    break

        return self._enrich_metadata(recommendations[:n])

    def recommend_similar_products(self, product_id, n=10):
        """
        Retrieves content-similar items for a given product.
        """
        recommendations = self.content_rec.recommend_similar(product_id, n=n)
        return self._enrich_metadata(recommendations)
