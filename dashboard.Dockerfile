FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dashboard scripts
COPY dashboards/ ./dashboards/

# Expose Streamlit default port
EXPOSE 8501

# Streamlit-specific configurations for production
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "dashboards/dashboard.py"]
