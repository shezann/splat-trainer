"""
Configuration settings for the training server.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent

# Use environment variables or fall back to local directories
_default_data = BASE_DIR / "data"
_default_output = BASE_DIR / "outputs"

DATA_DIR = Path(os.getenv("DATA_DIR", str(_default_data)))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_default_output)))

# Create directories if they don't exist (only if writable)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Fall back to local directories if workspace not writable
    DATA_DIR = _default_data
    OUTPUT_DIR = _default_output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training defaults
DEFAULT_ITERATIONS = 30000
DEFAULT_TARGET_GAUSSIANS = 500000
CHECKPOINT_ITERATIONS = [5000, 15000, 25000, 30000]

# Upload settings
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".zip", ".jpg", ".jpeg", ".png", ".json", ".ply", ".glb"}

# Server settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Quality presets
QUALITY_PRESETS = {
    "fast": {
        "iterations": 10000,
        "target_gaussians": 200000,
        "sh_degree": 2,
    },
    "balanced": {
        "iterations": 30000,
        "target_gaussians": 500000,
        "sh_degree": 3,
    },
    "high": {
        "iterations": 50000,
        "target_gaussians": 1000000,
        "sh_degree": 3,
    },
}
