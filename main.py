"""
Deco 3DGS Training API Server

FastAPI server for training 3D Gaussian Splatting models on GPU.
Designed to run on Vast.ai with CUDA-enabled GPUs.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uuid
import time
import setup_build_env  # noqa: F401 - configures ninja + MSVC for gsplat

from routes import jobs, upload
from services.job_manager import job_manager

# Configure logging with timestamps and structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Deco 3DGS Training Server...")
    job_manager.start_worker()
    yield
    logger.info("Shutting down...")
    job_manager.stop_worker()


app = FastAPI(
    title="Deco 3DGS Training API",
    description="Train 3D Gaussian Splatting models from iOS scan data",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for iOS app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    """Add request ID and log all requests with timing."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Add request ID to state for use in route handlers
    request.state.request_id = request_id

    # Log incoming request
    logger.info(f"[{request_id}] --> {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000

        # Log response with timing
        logger.info(
            f"[{request_id}] <-- {response.status_code} "
            f"({duration_ms:.1f}ms)"
        )

        # Add request ID to response headers for client debugging
        response.headers["X-Request-ID"] = request_id

        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[{request_id}] !!! Error after {duration_ms:.1f}ms: {str(e)}"
        )
        raise


# Include routers
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(upload.router, tags=["upload"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Deco 3DGS Training API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None

    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        gpu_memory = f"{props.total_memory / 1024**3:.1f} GB"

    return {
        "status": "healthy",
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "memory": gpu_memory,
        },
        "jobs": {
            "pending": len(job_manager.pending_jobs),
            "active": job_manager.current_job is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
