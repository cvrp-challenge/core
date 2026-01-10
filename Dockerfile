# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    git \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')" && \
    python -c "from sklearn.cluster import AgglomerativeClustering, KMeans; from sklearn.preprocessing import StandardScaler; print('sklearn imports successful')"

# Copy PyVRP first and install it (for better Docker layer caching)
# This layer will only rebuild if solver/pyvrp/ changes
COPY solver/pyvrp ./solver/pyvrp
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ./solver/pyvrp

# Copy the rest of the project (includes gurobi.lic if present)
COPY . .

# Set Python path to include the project root and src directory
ENV PYTHONPATH=/app:/app/src

# Set Gurobi license file to use the workspace tokenserver license
ENV GRB_LICENSE_FILE=/app/gurobi.lic

# Default entry point: src/master/main.py
ENTRYPOINT ["python", "-m", "master.benchmark_drsci"]

