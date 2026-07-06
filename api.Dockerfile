# Use a lightweight Python base image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling numerical libraries (scipy/implicit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and model binaries
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose FastAPI default port
EXPOSE 8000

# Start Gunicorn with Uvicorn workers for production scale
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
