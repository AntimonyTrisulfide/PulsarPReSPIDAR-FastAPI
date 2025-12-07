# syntax = docker/dockerfile:1.5

FROM python:3.11-slim AS base

# Ensure deterministic Python behavior and install curl for blob downloads
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying the rest of the app for better layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy FastAPI source code
COPY . /app

# Copy pre-downloaded model artifacts (downloaded in CI before docker build)
ENV MODEL_DIR=/models
RUN mkdir -p "${MODEL_DIR}"
COPY models/ "${MODEL_DIR}/"


FROM base AS final
ENV MODEL_DIR=/models
WORKDIR /app

EXPOSE 8001

CMD ["uvicorn","app.app:app","--host","0.0.0.0","--port","8001"]