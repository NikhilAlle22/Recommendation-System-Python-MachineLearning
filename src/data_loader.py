import pandas as pd
import os

def load_and_clean_data(raw_data_dir):
    """
    Loads raw Olist CSV datasets, performs filtering, joins them, 
    and returns a cleaned master DataFrame.
    """
    print("Loading raw datasets...")
    
    # Load required CSV datasets
    customer_data = pd.read_csv(os.path.join(raw_data_dir, "olist_customers_dataset.csv"))
    order_data = pd.read_csv(os.path.join(raw_data_dir, "olist_orders_dataset.csv"))
    product_data = pd.read_csv(os.path.join(raw_data_dir, "olist_products_dataset.csv"))
    order_items_data = pd.read_csv(os.path.join(raw_data_dir, "olist_order_items_dataset.csv"))
    order_payments_data = pd.read_csv(os.path.join(raw_data_dir, "olist_order_payments_dataset.csv"))
    order_review_data = pd.read_csv(os.path.join(raw_data_dir, "olist_order_reviews_dataset.csv"))
    product_name_translation = pd.read_csv(os.path.join(raw_data_dir, "product_category_name_translation.csv"))

    # 1. Only delivered orders represent completed user-product interactions
    print("Cleaning order table...")
    order_cleaned = order_data[order_data["order_status"] == "delivered"].copy()
    
    # Rename timestamps to standard columns
    order_cleaned = order_cleaned.rename(columns={"order_purchase_timestamp": "order_time"})

    # 2. Aggregate payment value per order (resolves duplicate items/installments per order)
    print("Aggregating payments...")
    payment_agg = order_payments_data.groupby("order_id")["payment_value"].sum().reset_index()

    # 3. Rename customer_unique_id to user_id to avoid confusion with transaction-level customer_id
    customer_data = customer_data.rename(columns={"customer_unique_id": "user_id"})

    # 4. Join datasets
    print("Performing table joins...")
    # Merge orders with customer info
    merged = order_cleaned.merge(customer_data, on="customer_id", how="inner")
    
    # Merge with order items
    merged = merged.merge(order_items_data, on="order_id", how="inner")
    
    # Merge with product info
    merged = merged.merge(product_data, on="product_id", how="inner")
    
    # Merge with translated category names
    merged = merged.merge(product_name_translation, on="product_category_name", how="left")
    
    # Standardize translated category names and handle missing categories
    merged["category"] = merged["product_category_name_english"].fillna("other")
    
    # Merge with aggregated payments
    merged = merged.merge(payment_agg, on="order_id", how="left")
    
    # Merge with reviews (taking mean review score if an order has duplicate reviews)
    review_agg = order_review_data.groupby("order_id")["review_score"].mean().reset_index()
    merged = merged.merge(review_agg, on="order_id", how="left")
    
    # Fallback review score to average if missing (or defaults)
    merged["review_score"] = merged["review_score"].fillna(3.0)
    merged["payment_value"] = merged["payment_value"].fillna(0.0)

    print(f"Data loading and cleaning finished. Master DataFrame shape: {merged.shape}")
    return merged
