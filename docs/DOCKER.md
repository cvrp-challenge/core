# Docker Guide

This guide covers the essential Docker workflow for this project, from building images to deploying on a server.

## Table of Contents

1. [Creating Docker Images](#1-creating-docker-images)
2. [Testing Images Locally](#2-testing-images-locally)
3. [Pushing Images to Docker Hub](#3-pushing-images-to-docker-hub)
4. [Tagging Images](#4-tagging-images)
5. [Docker Login on Server](#5-docker-login-on-server)
6. [Pulling Images on Server](#6-pulling-images-on-server)
7. [Running Images on Server](#7-running-images-on-server)
8. [Understanding the Dockerfile](#8-understanding-the-dockerfile)
9. [Understanding .dockerignore](#9-understanding-dockerignore)

---

## 1. Creating Docker Images

Build a Docker image from the Dockerfile:

```bash
docker build -t core-solver:latest .
```

**What this does:**
- `-t core-solver:latest` - Names the image "core-solver" with tag "latest"
- `.` - Uses current directory as build context

**View your built image:**
```bash
docker images core-solver
```

---

## 2. Testing Images Locally

### Test with interactive shell

```bash
docker run -it --rm core-solver:latest /bin/bash
```

This opens a shell inside the container so you can explore and test.

### Run with default entrypoint

```bash
docker run --rm core-solver:latest
```

This runs the default command defined in the Dockerfile.

### Run with volume mounts

Mount local directories to access data:

```bash
docker run --rm \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/output:/app/output \
  core-solver:latest
```

### View logs from running container

```bash
# List running containers
docker ps

# View logs
docker logs <container_id>
```

---

## 3. Pushing Images to Docker Hub

### Login to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted.

### Tag image with your username

```bash
docker tag core-solver:latest your-dockerhub-username/core-solver:latest
```

### Push to Docker Hub

```bash
docker push your-dockerhub-username/core-solver:latest
```

### Verify

Check your repository at: `https://hub.docker.com/r/your-dockerhub-username/core-solver`

---

## 4. Tagging Images

### Create version tags

```bash
# Tag specific version
docker tag core-solver:latest your-dockerhub-username/core-solver:v1.0.0

# Tag and push both
docker push your-dockerhub-username/core-solver:v1.0.0
docker push your-dockerhub-username/core-solver:latest
```

**Best practice:** Use semantic versioning (v1.0.0, v1.0.1, etc.) for releases and `latest` for the most recent stable version.

---

## 5. Docker Login on Server

### SSH to your server

```bash
ssh user@your-server.com
```

### Install Docker (if needed)

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Allow running without sudo (optional)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

### Login to Docker Hub

```bash
docker login
```

Enter your credentials when prompted.

---

## 6. Pulling Images on Server

### Pull your image

```bash
docker pull your-dockerhub-username/core-solver:latest
```

### Verify download

```bash
docker images your-dockerhub-username/core-solver
```

---

## 7. Running Images on Server

### Run in background

```bash
docker run -d \
  --name core-solver-instance \
  --restart unless-stopped \
  -v /path/on/server/instances:/app/instances:ro \
  -v /path/on/server/output:/app/output \
  your-dockerhub-username/core-solver:latest
```

**Options explained:**
- `-d` - Run in background (detached)
- `--name` - Give container a name
- `--restart unless-stopped` - Auto-restart if it crashes
- `-v` - Mount volumes (`:ro` = read-only)

### Run with resource limits

```bash
docker run -d \
  --name core-solver-instance \
  --restart unless-stopped \
  --memory="4g" \
  --cpus="4.0" \
  -v /path/on/server/instances:/app/instances:ro \
  -v /path/on/server/output:/app/output \
  your-dockerhub-username/core-solver:latest
```

### Run with Gurobi Token Server (Network Configuration)

If you're using a Gurobi token server license, the container needs network access to reach the token server. Use `--network host` to allow the container to access the host's network:

```bash
docker run --rm --network host \
  -v /tmp/drsci_benchmark:/app/output \
  zorroy/cvrp:2.1.1 /app/output --max_workers 10
```

**Why this is needed:**
- Gurobi token server licenses require the client to be on the same network subnet as the server
- Docker containers by default use an isolated network namespace
- `--network host` makes the container use the host's network stack, allowing it to reach the token server

**Platform compatibility:**
- ✅ **Native Linux servers**: `--network host` works perfectly. This is the recommended setup for university servers or cloud VMs running Linux.
- ⚠️ **Docker Desktop on Windows/WSL2**: `--network host` has limitations and may not work as expected. The container may still appear on a different subnet than the token server. In this case, you may need to:
  - Run the code directly on the host (outside Docker)
  - Use a different license type (node-locked instead of token server)
  - Configure Docker networking with custom bridge networks (advanced)

### Monitor running container

```bash
# View running containers
docker ps

# View logs (real-time)
docker logs -f core-solver-instance

# View resource usage
docker stats core-solver-instance
```

### Stop and restart

```bash
# Stop container
docker stop core-solver-instance

# Start stopped container
docker start core-solver-instance

# Remove container
docker rm core-solver-instance
```

---

## 8. Understanding the Dockerfile

The Dockerfile defines how the image is built. Here's what each section does:

### Base Image
```dockerfile
FROM python:3.11-slim
```
Starts with minimal Python 3.11 image from Docker Hub.

### Working Directory
```dockerfile
WORKDIR /app
```
Sets `/app` as the working directory inside the container.

### System Dependencies
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*
```
Installs system tools needed to build C++ extensions (PyVRP requires these).

### Python Dependencies
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
```
Installs Python packages. Copied separately first for better caching (if requirements.txt doesn't change, this layer is reused).

### Copy Project Files
```dockerfile
COPY . .
```
Copies all project files (respecting .dockerignore).

### Build PyVRP
```dockerfile
RUN pip install --no-cache-dir ./solver/pyvrp
```
Builds and installs PyVRP solver from source.

### Environment Setup
```dockerfile
ENV PYTHONPATH=/app:/app/src
```
Allows Python to import modules from project root and src directory.

### Default Command
```dockerfile
ENTRYPOINT ["python", "-m", "master.benchmark"]
```
Defines what runs when the container starts. Can be overridden with `docker run ... python -m other.module`.

---

## 9. Understanding .dockerignore

The `.dockerignore` file tells Docker which files to exclude when building the image. This makes builds faster and images smaller.

### What's excluded and why:

**Python artifacts:**
```
__pycache__/
*.pyc
*.so
venv/
```
Runtime files that will be regenerated inside the container.

**IDE files:**
```
.vscode/
.idea/
```
Not needed in production, contains local settings.

**Git files:**
```
.git/
.gitignore
```
Version control metadata isn't needed in the image.

**Build artifacts:**
```
build/
*.o
*.a
```
Compiled files that will be rebuilt for the container's OS.

**Output files:**
```
*.log
*.out
results/
```
Generated at runtime, should be written to mounted volumes.

**Large unnecessary files:**
```
instances/**/*.png
```
Visualization files are large and slow down builds. Instance data (.vrp files) are still included.

**Selective solver inclusion:**
```
solver/
!solver/pyvrp/
!solver/pyvrp/**
```
Only PyVRP is needed, other solvers are excluded to reduce image size.

### Why this matters:
- **Faster builds** - Less data to transfer to Docker
- **Smaller images** - Only essential files included
- **Better caching** - Fewer file changes mean better cache reuse
- **Security** - Prevents accidentally including credentials or secrets

---

## Complete Workflow Example

### Local Development to Server Deployment

**On your local machine:**
```bash
# 1. Build image
docker build -t core-solver:v1.0.0 .

# 2. Test locally
docker run --rm \
  -v $(pwd)/instances:/app/instances:ro \
  -v $(pwd)/output:/app/output \
  core-solver:v1.0.0

# 3. Tag for Docker Hub
docker tag core-solver:v1.0.0 yourusername/core-solver:v1.0.0
docker tag core-solver:v1.0.0 yourusername/core-solver:latest

# 4. Push to Docker Hub
docker login
docker push yourusername/core-solver:v1.0.0
docker push yourusername/core-solver:latest
```

**On your server:**
```bash
# 1. Login and pull
docker login
docker pull yourusername/core-solver:latest

# 2. Stop old version (if running)
docker stop core-solver-prod
docker rm core-solver-prod

# 3. Run new version
docker run -d \
  --name core-solver-prod \
  --restart unless-stopped \
  -v /data/instances:/app/instances:ro \
  -v /data/output:/app/output \
  --memory="8g" \
  --cpus="8.0" \
  yourusername/core-solver:latest

# 4. Check logs
docker logs -f core-solver-prod
```

---

## Useful Commands

### Cleanup
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove everything unused
docker system prune
```

### Troubleshooting
```bash
# View container logs
docker logs <container-name>

# Get shell in running container
docker exec -it <container-name> /bin/bash

# Check container details
docker inspect <container-name>
```

---

**Last Updated:** December 2024
