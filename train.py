import os
from src.data_loader import load_and_clean_data
from src.features import prepare_features
from src.models import PopularityRecommender, ALSRecommender, ContentRecommender, save_models

def main():
    # Define directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(base_dir, "Data", "raw")
    models_dir = os.path.join(base_dir, "models")
    
    # Check if raw data files are present (they are gitignored in server deployments)
    required_file = os.path.join(raw_data_dir, "olist_customers_dataset.csv")
    if not os.path.exists(required_file):
        print(f"\n[NOTICE] Raw data file '{required_file}' not found.")
        print("Skipping model training pipeline because raw datasets are not present (typical on cloud deployments).")
        print("Existing pre-trained models in the 'models/' folder will be used by the API.\n")
        return
    
    print("--- STEP 1: Ingesting & Cleaning Data ---")
    master_df = load_and_clean_data(raw_data_dir)
    
    print("\n--- STEP 2: Feature Engineering & Profiling ---")
    (master_df,
     interaction_df, 
     user_encoder, 
     product_encoder, 
     user_profiles, 
     product_profiles) = prepare_features(master_df)

    print("\n--- STEP 3: Initializing & Fitting Models ---")
    
    # 1. Popularity Recommender
    pop_rec = PopularityRecommender()
    pop_rec.fit(master_df)
    
    # 2. Collaborative Filtering (ALS) Recommender
    als_rec = ALSRecommender(factors=100, regularization=0.1, iterations=20)
    als_rec.fit(interaction_df)
    
    # 3. Content-Based Recommender
    content_rec = ContentRecommender()
    content_rec.fit(product_profiles)
    
    # 4. Extract user history (list of product IDs purchased per customer unique ID)
    print("Extracting user transaction history maps...")
    user_history_df = master_df.groupby("user_id")["product_id"].apply(set).reset_index()
    user_history_dict = dict(zip(user_history_df["user_id"], user_history_df["product_id"]))

    print("\n--- STEP 4: Serializing Models to Disk ---")
    save_models(
        save_dir=models_dir,
        popularity_rec=pop_rec,
        als_rec=als_rec,
        content_rec=content_rec,
        user_encoder=user_encoder,
        product_encoder=product_encoder,
        user_profiles=user_profiles,
        product_profiles=product_profiles,
        user_history=user_history_dict
    )
    
    print("\nTraining completed successfully! Binary files are saved in the 'models/' folder.")

if __name__ == "__main__":
    main()
