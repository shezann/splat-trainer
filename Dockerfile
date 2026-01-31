# Deco 3DGS Training Server
# Build: docker build -t deco-splat-trainer .
# Run: docker run --gpus all -p 8000:8000 -v ./data:/workspace/data -v ./outputs:/workspace/outputs deco-splat-trainer

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/workspace/data
ENV OUTPUT_DIR=/workspace/outputs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace/splat-trainer

# Copy requirements first for caching
COPY requirements.txt .

# Initialize uv project and install dependencies
RUN uv init --name splat-trainer && \
    uv python install 3.11 && \
    uv add fastapi uvicorn[standard] python-multipart aiofiles pillow numpy plyfile pydantic python-dotenv trimesh && \
    uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY main.py config.py ./
COPY models/ ./models/
COPY services/ ./services/
COPY routes/ ./routes/

# Create data directories
RUN mkdir -p /workspace/data /workspace/outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
