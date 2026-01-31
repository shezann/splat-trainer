#!/bin/bash
#
# Deploy splat-trainer to a Vast.ai instance
#
# Usage:
#   ./deploy.sh <ssh-command>
#   ./deploy.sh "ssh -p 12345 root@123.45.67.89"

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <ssh-command>"
    echo "Example: $0 'ssh -p 12345 root@123.45.67.89'"
    exit 1
fi

SSH_CMD="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Deploying from: $PROJECT_DIR"
echo "SSH command: $SSH_CMD"

# Extract host and port from SSH command
SSH_ARGS=$(echo "$SSH_CMD" | sed 's/ssh //')

# Create remote directory
echo "Creating remote directory..."
$SSH_CMD "mkdir -p /workspace/splat-trainer/{models,services,routes,scripts}"

# Copy Python files
echo "Copying Python files..."
scp $SSH_ARGS "$PROJECT_DIR/main.py" /workspace/splat-trainer/
scp $SSH_ARGS "$PROJECT_DIR/config.py" /workspace/splat-trainer/
scp $SSH_ARGS "$PROJECT_DIR/requirements.txt" /workspace/splat-trainer/

# Copy modules
echo "Copying modules..."
scp -r $SSH_ARGS "$PROJECT_DIR/models/"* /workspace/splat-trainer/models/
scp -r $SSH_ARGS "$PROJECT_DIR/services/"* /workspace/splat-trainer/services/
scp -r $SSH_ARGS "$PROJECT_DIR/routes/"* /workspace/splat-trainer/routes/
scp -r $SSH_ARGS "$PROJECT_DIR/scripts/"* /workspace/splat-trainer/scripts/

# Run setup script
echo "Running setup script..."
$SSH_CMD "cd /workspace/splat-trainer && bash scripts/setup_gpu.sh"

echo ""
echo "Deployment complete!"
echo "Start the server with: $SSH_CMD './workspace/splat-trainer/start.sh'"
