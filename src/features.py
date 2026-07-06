import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    """
    Computes implicit interaction scores and creates unified user and product profiles,
    and returns encoders to map user and product IDs to contiguous integer codes.
    """
    print("Preparing features & interaction scores...")

    # 1. Implicit Feedback calculation
    # Base purchase score
    purchase_weight = 5
    df["purchase_weight"] = purchase_weight

    # Review score boost
    df["review_weight"] = np.where(df["review_score"] >= 4.0, 2, 0)

    # Order value boost (payment value > global mean AOV)
    global_mean_aov = df["payment_value"].mean()
    df["value_weight"] = np.where(df["payment_value"] > global_mean_aov, 1, 0)

    # Repeat purchase boost (Group by user AND product to check if a specific user bought the same product multiple times)
    user_product_counts = df.groupby(["user_id", "product_id"]).size().reset_index(name="user_product_count")
    user_product_counts["repeat_weight"] = np.where(user_product_counts["user_product_count"] > 1, 3, 0)

    # Merge repeat purchase boost back
    df = df.merge(user_product_counts[["user_id", "product_id", "repeat_weight"]], on=["user_id", "product_id"], how="left")

    # Combine into unified interaction score
    df["interaction_score"] = df["purchase_weight"] + df["review_weight"] + df["value_weight"] + df["repeat_weight"]

    # 2. Extract final interaction table (taking maximum interaction score if duplicates exist)
    interaction_df = df.groupby(["user_id", "product_id"])["interaction_score"].max().reset_index()

    # 3. Create unified label encoders for users and products
    print("Fitting unified ID encoders...")
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    interaction_df["user_id_enc"] = user_encoder.fit_transform(interaction_df["user_id"])
    interaction_df["product_id_enc"] = product_encoder.fit_transform(interaction_df["product_id"])

    # 4. Generate User Profiles
    print("Generating user profiles...")
    user_purchases = df.groupby("user_id").size().reset_index(name="total_purchases")
    user_aov = df.groupby("user_id")["payment_value"].mean().reset_index(name="avg_order_value")
    
    # Favorite category (highest purchase count per category)
    fav_categories = df.groupby(["user_id", "category"]).size().reset_index(name="cat_count")
    fav_categories = fav_categories.sort_values(["user_id", "cat_count"], ascending=[True, False])
    fav_categories = fav_categories.drop_duplicates(subset="user_id").rename(columns={"category": "favorite_category"})

    # Active days span
    df["order_time"] = pd.to_datetime(df["order_time"])
    user_active_days = df.groupby("user_id")["order_time"].agg(lambda x: (x.max() - x.min()).days + 1).reset_index(name="active_days")

    user_profiles = (user_purchases
                     .merge(user_aov, on="user_id", how="left")
                     .merge(fav_categories[["user_id", "favorite_category"]], on="user_id", how="left")
                     .merge(user_active_days, on="user_id", how="left")
                     )

    # 5. Generate Product Profiles
    print("Generating product profiles...")
    product_purchases = df.groupby("product_id").size().reset_index(name="total_purchases")
    product_price = df.groupby("product_id")["price"].mean().reset_index(name="avg_price")
    product_review = df.groupby("product_id")["review_score"].mean().reset_index(name="avg_review_score")
    product_category = df[["product_id", "category"]].drop_duplicates(subset="product_id")

    product_profiles = (product_purchases
                        .merge(product_price, on="product_id", how="left")
                        .merge(product_review, on="product_id", how="left")
                        .merge(product_category, on="product_id", how="left")
                        )

    return df, interaction_df, user_encoder, product_encoder, user_profiles, product_profiles
