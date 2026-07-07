import os
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Query
from src.recommender import RecommenderSystem

from fastapi.responses import RedirectResponse

app = FastAPI(
    title="E-Commerce Recommendation System API",
    description="Serving personalized and content-based recommendations for Olist e-commerce dataset.",
    version="1.0.0"
)

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")

# Global reference to Recommender System
recommender = None

@app.on_event("startup")
def startup_event():
    global recommender
    # Resolve the models directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(os.path.join(models_dir, "popularity_rec.joblib")):
        print(f"Error: Pre-trained models not found in {models_dir}. Please run 'python train.py' first.")
        # We don't raise exception here to allow startup, but will return 503 on queries
        return
        
    print("Loading models into memory...")
    recommender = RecommenderSystem(models_dir)
    print("Recommender system loaded successfully.")

@app.get("/recommend", tags=["Recommendations"])
def get_user_recommendations(
    user_id: str = Query(..., description="Unique customer ID (customer_unique_id)"),
    state: str = Query(None, description="Optional customer state code (e.g. SP, RJ) for cold-start filtering"),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations to return")
):
    """
    Returns personalized recommendations for a user. If the user is new (cold start),
    popularity-based fallback suggestions are served, filtered by customer state if provided.
    """
    global recommender
    if recommender is None:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service is currently unavailable. Ensure models have been trained."
        )
    
    try:
        recommendations = recommender.recommend_for_user(user_id=user_id, state=state, n=n)
        return {
            "status": "success",
            "user_id": user_id,
            "results_count": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/recommend/similar", tags=["Recommendations"])
def get_similar_products(
    product_id: str = Query(..., description="Product ID to find similar items for"),
    n: int = Query(10, ge=1, le=50, description="Number of similar products to return")
):
    """
    Returns content-based similar items for a product details page.
    """
    global recommender
    if recommender is None:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service is currently unavailable. Ensure models have been trained."
        )
    
    try:
        similar_items = recommender.recommend_similar_products(product_id=product_id, n=n)
        if not similar_items:
            raise HTTPException(status_code=404, detail=f"Product ID '{product_id}' not found in catalog.")
        return {
            "status": "success",
            "product_id": product_id,
            "results_count": len(similar_items),
            "recommendations": similar_items
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health", tags=["Utilities"])
def health_check():
    """
    Simple service health check.
    """
    global recommender
    return {
        "status": "healthy",
        "models_loaded": recommender is not None
    }
