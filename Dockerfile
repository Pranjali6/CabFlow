FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY config/ config/
COPY src/ src/
COPY dashboard/ dashboard/
COPY api/ api/
COPY data/demo/ data/demo/

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
