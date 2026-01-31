"""
Job management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse
from typing import Optional, List

from models.schemas import (
    JobStatus,
    TrainingConfig,
    TrainingJob,
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
    CheckpointInfo,
    CheckpointListResponse,
    TrainingJobResponse,
)
from services.job_manager import job_manager
from services.storage import storage_service

router = APIRouter()


@router.post("", response_model=JobCreateResponse, status_code=201)
async def create_job(request: JobCreateRequest = None):
    """
    Create a new training job.

    Returns job ID and upload URL for sending training data.
    """
    config = request.to_config() if request else TrainingConfig()
    scan_id = request.scan_id if request else None

    job = job_manager.create_job(
        scan_id=scan_id or "unnamed",
        config=config,
    )

    # Build full upload URL
    upload_url = f"/upload/{job.id}"

    return JobCreateResponse(
        jobId=job.id,
        uploadURL=upload_url,
    )


@router.get("")
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
):
    """List all training jobs - returns iOS JobListResponse format."""
    jobs = job_manager.list_jobs(status=status, limit=limit)
    return {
        "jobs": [TrainingJobResponse.from_job(job) for job in jobs]
    }


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """
    Get job status and progress.

    This endpoint is polled by the iOS app for progress updates.
    Returns JobStatusResponse with nested 'job' field.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse.from_job(job)


@router.get("/{job_id}/full", response_model=TrainingJob)
async def get_job_full(job_id: str):
    """Get full job details including configuration."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Cannot cancel job (not found or already completed)",
        )

    return {"status": "cancelled", "job_id": job_id}


@router.get("/{job_id}/checkpoints")
async def list_checkpoints(job_id: str):
    """List available checkpoints for a job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    checkpoints = storage_service.get_checkpoints(job_id)

    # Match iOS CheckpointInfo format
    checkpoint_infos = [
        {
            "iteration": cp["iteration"],
            "gaussianCount": 0,  # Not available from PLY without parsing
            "downloadURL": f"/jobs/{job_id}/checkpoints/{cp['iteration']}",
            "createdAt": cp["created_at"].isoformat(),
        }
        for cp in checkpoints
    ]

    return {"checkpoints": checkpoint_infos}


@router.get("/{job_id}/checkpoints/{iteration}")
async def download_checkpoint(job_id: str, iteration: int):
    """
    Download a specific checkpoint.

    Returns CheckpointDownloadResponse for iOS client, then redirect or file.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = storage_service.get_job_output_dir(job_id)
    checkpoint_path = output_dir / f"point_cloud_{iteration}.ply"

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    file_size = checkpoint_path.stat().st_size

    # Return download info matching iOS CheckpointDownloadResponse
    return {
        "downloadURL": f"/download/{job_id}/checkpoint/{iteration}",
        "gaussianCount": 0,
        "fileSize": file_size,
    }
