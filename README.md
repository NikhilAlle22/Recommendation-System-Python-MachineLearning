# 🛍️ Hybrid E-Commerce Recommendation System

An end-to-end, production-ready Hybrid Product Recommendation System built on the **Olist Brazilian E-Commerce Public Dataset**. 

This system is designed with a production-first mindset: it segments users into **Warm** (returning) and **Cold-Start** (new) segments, applies regional/demographic fallback heuristics, exposes recommendations through a high-performance **FastAPI REST API**, and provides a stunning, interactive client-side **Streamlit Dashboard** for visualization.

---

## 🚀 Live Demo

*   **Streamlit Interactive Dashboard**: [https://recommendation-dashboard.onrender.com/](https://recommendation-dashboard.onrender.com/)
*   **FastAPI REST API Documentation (Swagger)**: [https://recommendation-system-python-3kq1.onrender.com/docs](https://recommendation-system-python-3kq1.onrender.com/docs)

---

## 📊 Business Goals & Success Metrics

### Business Objective
Increase **customer retention**, **repeat purchases**, and **Average Order Value (AOV)** by serving personalized and contextually relevant product suggestions during active customer journeys.

### Success Metrics
*   **Offline (ML) Metrics:** Precision@K, Recall@K, and Normalized Discounted Cumulative Gain (NDCG) to validate retrieval quality.
*   **Online (Business) Metrics:** Click-Through Rate (CTR), Conversion Rate, Average Order Value (AOV), and Repeat Purchase Rate.

---

## 🏗️ System Architecture & Workflow

The system is split into distinct pipelines for data ingestion, training, serving, and dashboard visualization.

```mermaid
graph TD
    subgraph Offline Pipeline (Batch Training)
        A[Olist CSV Raw Data] --> B[Data Loader & Cleaning]
        B --> C[Implicit Interaction Scoring]
        C --> D[Feature Engineering & Profiling]
        D --> E[Model Ingestion & Fitting]
        E --> F[Serialization to /models]
    end

    subgraph Online Serving (Real-time REST API)
        F --> G[FastAPI Service]
        G --> H[Endpoint: /recommend]
        G --> I[Endpoint: /recommend/similar]
        G --> J[Endpoint: /health]
    end

    subgraph Presentation Layer (User Interface)
        H --> K[Streamlit Interactive Hub]
        I --> K
    end
    
    style Offline Pipeline fill:#f5f7ff,stroke:#3b5998,stroke-width:2px;
    style Online Serving fill:#f0fff4,stroke:#38a169,stroke-width:2px;
    style Presentation Layer fill:#fffaf0,stroke:#dd6b20,stroke-width:2px;
```

---

## 🤖 Recommendation Strategies by Segment

To provide a robust recommendation experience, the system routes requests dynamically based on the customer's profile:

| Model Component | Type | Technique | Target User / Scenario | Features |
| :--- | :--- | :--- | :--- | :--- |
| **PopularityRecommender** | Heuristic | Weighted Interaction Sum + State-level Filtering | **Cold-Start Users** (new / unrecognized) | Fallbacks are personalized by geographic state (e.g., SP, RJ) if provided. |
| **ALSRecommender** | Collaborative Filtering | Alternating Least Squares (Implicit Feedback) | **Warm Users** (returning customers with purchase history) | Utilizes implicit feedback signals (purchases, ratings, repeat purchase count). |
| **ContentRecommender** | Content-Based | TF-IDF Text Vectorization + Cosine Similarity (K-NN) | **Similar Products** (Product detail pages cross-selling) | Matches products using English translated categories, price buckets, and rating profiles. |

---

## 📁 Project Directory Structure

```text
E-commerce_Recommendation/
│
├── Data/
│   ├── raw/                 # Raw Olist CSV datasets (from Kaggle)
│   └── cleaned/             # Intermediate processed data
│
├── api/
│   └── app.py               # FastAPI application backend (REST endpoints)
│
├── dashboards/
│   └── dashboard.py         # Streamlit dashboard client with custom CSS
│
├── models/                  # Serialized trained model binaries (.joblib)
│   ├── popularity_rec.joblib
│   ├── als_rec.joblib
│   ├── content_rec.joblib
│   ├── user_encoder.joblib
│   ├── product_encoder.joblib
│   ├── user_profiles.joblib
│   ├── product_profiles.joblib
│   └── user_history.joblib
│
├── notebooks/               # Jupyter notebooks for EDA and prototyping
│   └── 01_data_cleaning.ipynb
│
├── src/                     # Core Python modules package
│   ├── __init__.py
│   ├── data_loader.py       # Data cleaning, parsing, and merging logic
│   ├── features.py          # Feature creation & implicit interaction scoring
│   ├── models.py            # Custom ML models (ALS, Content-Based, Popularity)
│   └── recommender.py       # Production router and enrichment layer
│
├── train.py                 # Master pipeline training & evaluation script
├── requirements.txt         # Project dependencies (pinned to working versions)
├── api.Dockerfile           # Container configuration for backend API
├── dashboard.Dockerfile     # Container configuration for frontend UI
├── docker-compose.yml       # Docker orchestrator for multi-container run
└── README.md                # Project documentation (this file)
```

---

## ⚡ Setup & Quickstart

Follow these steps to run the recommendation pipeline locally.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. Extract the Olist datasets under `Data/raw/`. The raw directory should contain:
*   `olist_customers_dataset.csv`
*   `olist_orders_dataset.csv`
*   `olist_products_dataset.csv`
*   `olist_order_items_dataset.csv`
*   `olist_order_payments_dataset.csv`
*   `olist_order_reviews_dataset.csv`
*   `product_category_name_translation.csv`

### 2. Environment Configuration
Create a virtual environment and install the required dependencies:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Models
Run the training pipeline script. This processes the raw CSV tables, engineers features, trains the popularity/ALS/content models, and serializes them into the `models/` directory:
```bash
python train.py
```

### 4. Launch the FastAPI Serving Layer
Run the FastAPI application server in reload mode:
```bash
uvicorn api.app:app --reload
```
Once started, the interactive OpenAPI documentation is available at **http://127.0.0.1:8000/docs**.

### 5. Launch the Streamlit Frontend Hub
Run the Streamlit dashboard:
```bash
streamlit run dashboards/dashboard.py
```
This launches a browser page where you can simulate returning or cold-start users, filter popularity metrics by Brazil states, search for similar products, and inspect pricing/rating distributions.

### 6. Alternative: Run via Docker Compose (Containerized Run)
If you have Docker installed, you can build and run both the API and Streamlit Dashboard simultaneously using a single command:
```bash
# Build images and start services in the background
docker compose up --build -d
```
* Once running, the FastAPI service will be accessible at **http://localhost:8000** and the Streamlit Dashboard will be accessible at **http://localhost:8501**.
* To stop the services:
  ```bash
  docker compose down
  ```

---

## 🌐 API Endpoint Reference

### 1. Get Recommendations
* **Endpoint:** `GET /recommend`
* **Description:** Retrieves personalized product recommendations for a customer. If the user is new (cold start), popularity-based suggestions are served instead.
* **Query Parameters:**
  * `user_id` (string, required): Customer Unique ID.
  * `state` (string, optional): State code (e.g., `SP`, `RJ`) for geographic filtering under cold start.
  * `n` (integer, optional, default=10): Number of recommendations (1-50).
* **Sample Response:**
  ```json
  {
    "status": "success",
    "user_id": "0000366f3b9a7992bf8c76cfdf3221e2",
    "results_count": 10,
    "recommendations": [
      {
        "product_id": "53b36df67ebb7c41585e8d54d6772e08",
        "category": "watches_gifts",
        "price": 115.9,
        "review_score": 4.2,
        "popularity_sales": 23
      }
    ]
  }
  ```

### 2. Get Similar Products
* **Endpoint:** `GET /recommend/similar`
* **Description:** Retrieves content-similar items for product details pages.
* **Query Parameters:**
  * `product_id` (string, required): Product ID to find similar items for.
  * `n` (integer, optional, default=10): Number of recommendations (1-50).

### 3. Service Health Check
* **Endpoint:** `GET /health`
* **Description:** Returns the operational status of the service and confirms if model artifacts are successfully loaded into memory.

---

## 💡 Key Design Highlights
*   **Implicit Feedback Engine:** In the absence of star-ratings, we engineer an *interaction score* combining base purchases, positive reviews (score $\ge$ 4.0), high average order values, and repeat transactions.
*   **Dynamic Deduplication:** Warm users receive suggestions from ALS but have their historical purchase list filtered out in real-time, preventing redundant recommendations.
*   **Fast cold-starts:** When a new user lands on the website, they are not served empty pages. They instantly receive regional top-selling products filtered by their geographical location.
