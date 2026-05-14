FROM python:3.11-slim

WORKDIR /app

# Build deps: libgomp for OpenMP-based gradient boosting, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Application code
COPY config/ config/
COPY src/ src/
COPY dashboard/ dashboard/
COPY api/ api/

# Trained models + demo data (small enough to ship in the image)
COPY models/ models/
COPY data/processed/trips_hourly.parquet data/processed/
COPY data/processed/trips_featured.parquet data/processed/
COPY data/raw/taxi_zone_lookup.csv data/raw/
COPY data/raw/weather/ data/raw/weather/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
