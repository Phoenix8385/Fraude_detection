FROM python:3.11.15-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.lock .
RUN pip install -r requirements-api.lock
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY models ./models
ENV PYTHONPATH=/app/src:/app
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
 CMD curl -fs http://localhost:${PORT:-8000}/health || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
