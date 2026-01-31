#!/bin/bash
#
# Deco 3DGS Training Server - Vast.ai Setup Script
#
# Run this script after SSHing into your Vast.ai instance:
#   bash setup_gpu.sh
#
# Prerequisites:
#   - Vast.ai instance with CUDA 12.x image
#   - RTX 3090/4090 or better GPU
#   - Port 8000 open in Vast.ai config

set -e

echo "=========================================="
echo "  Deco 3DGS Training Server Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on GPU instance
echo -e "${YELLOW}Checking GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
echo -e "${GREEN}GPU detected!${NC}"

# Create workspace
echo -e "${YELLOW}Setting up workspace...${NC}"
mkdir -p /workspace/splat-trainer
mkdir -p /workspace/data
mkdir -p /workspace/outputs

cd /workspace/splat-trainer

# Install uv (fast Python package manager)
echo -e "${YELLOW}Installing uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi

# Initialize Python project
echo -e "${YELLOW}Initializing Python environment...${NC}"
uv init --name splat-trainer 2>/dev/null || true
uv python install 3.11

# Install base dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
uv add fastapi uvicorn[standard] python-multipart aiofiles pillow numpy plyfile pydantic python-dotenv trimesh

# Install PyTorch with CUDA support
echo -e "${YELLOW}Installing PyTorch with CUDA...${NC}"
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install gsplat (if available via pip)
echo -e "${YELLOW}Installing gsplat...${NC}"
uv add gsplat 2>/dev/null || echo "gsplat not available via pip, will use bundled training"

# Clone OpenSplat as alternative backend
echo -e "${YELLOW}Setting up OpenSplat...${NC}"
if [ ! -d "/workspace/opensplat" ]; then
    cd /workspace
    git clone https://github.com/pierotofy/OpenSplat.git opensplat
    cd opensplat
    pip install -r requirements.txt
    pip install .
    cd /workspace/splat-trainer
else
    echo "OpenSplat already installed"
fi

# Copy server files (if not already present)
echo -e "${YELLOW}Setting up server files...${NC}"
if [ ! -f "main.py" ]; then
    echo "Please copy the server files to /workspace/splat-trainer/"
    echo "Required files:"
    echo "  - main.py"
    echo "  - config.py"
    echo "  - models/"
    echo "  - services/"
    echo "  - routes/"
fi

# Create systemd service (optional)
echo -e "${YELLOW}Creating startup script...${NC}"
cat > /workspace/splat-trainer/start.sh << 'EOF'
#!/bin/bash
cd /workspace/splat-trainer
export DATA_DIR=/workspace/data
export OUTPUT_DIR=/workspace/outputs
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
EOF
chmod +x /workspace/splat-trainer/start.sh

# Test PyTorch CUDA
echo -e "${YELLOW}Testing CUDA...${NC}"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To start the server:"
echo "  cd /workspace/splat-trainer"
echo "  ./start.sh"
echo ""
echo "Or manually:"
echo "  uv run uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "API will be available at: http://<your-instance-ip>:8000"
echo ""
echo "Don't forget to open port 8000 in Vast.ai!"
