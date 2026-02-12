# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies (Python build + Java for AILS2)
# Uses Docker BuildKit cache mounts so apt packages are reused across builds
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
    openjdk-21-jre-headless \
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

# Build COBRA library (required by filo1)
# This layer will only rebuild if solver/cobra/ changes
COPY solver/cobra ./solver/cobra
RUN cd solver/cobra && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# Build FILO1 (depends on cobra)
# This layer will only rebuild if solver/filo1/ changes
COPY solver/filo1 ./solver/filo1
RUN cd solver/filo1 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=OFF && \
    make -j$(nproc)

# Build FILO2 (standalone)
# This layer will only rebuild if solver/filo2/ changes
COPY solver/filo2 ./solver/filo2
RUN cd solver/filo2 && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=ON && \
    make -j$(nproc)

# Copy the rest of the project (includes gurobi.lic if present)
COPY . .

# Set Python path to include the project root and src directory
ENV PYTHONPATH=/app:/app/src

# Set Gurobi license file to use the workspace tokenserver license
ENV GRB_LICENSE_FILE=/app/gurobi.lic

# Default entry point: run final_benchmark.py
ENTRYPOINT ["python", "-m", "master.challenge_runner"]

