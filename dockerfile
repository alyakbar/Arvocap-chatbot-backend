# Enable BuildKit features
# syntax=docker/dockerfile:1.4

# Use Python 3.11 slim image as base
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    gcc \
    libc6-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    cmake \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for FAISS optimization
ENV FAISS_ENABLE_GPU=0 \
    FAISS_ENABLE_AVX2=1 \
    FAISS_ENABLE_SSE4=1 \
    BLAS_LIBS="-lopenblas" \
    FAISS_OPT_LEVEL=3

# Create necessary directories
RUN mkdir -p logs data/vector_db data/models

# Copy requirements first for better caching
COPY requirements.txt .

# Install base dependencies
RUN pip install --upgrade pip && \
    pip install wheel setuptools

# Install heavy ML dependencies first for better caching
RUN pip install torch==2.5.0 faiss-cpu==1.12.0 transformers==4.41.2 sentence-transformers==5.1.0

# Install remaining dependencies
RUN pip install -r requirements.txt && \
    pip install langdetect

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set default command
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
