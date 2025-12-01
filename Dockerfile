# Use Python 3.11 as base image (matches project requirements)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Build and install vendored PyVRP (so pyvrp._pyvrp exists)
RUN pip install --no-cache-dir ./solver/pyvrp

# Set Python path to include the project root and src directory
ENV PYTHONPATH=/app:/app/src

# Default entry point: src/master/main.py
ENTRYPOINT ["python", "-m", "master.benchmark"]

