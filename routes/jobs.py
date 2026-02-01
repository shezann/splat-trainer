"""
Job management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, List
import asyncio
import json
import logging

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

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/downloads")
async def list_available_downloads(
    limit: int = Query(50, ge=1, le=100, description="Max results"),
):
    """
    List all completed training jobs with their download URLs.

    This endpoint allows any client to browse and download available splats
    without needing to track individual job IDs.
    """
    # Get all completed jobs
    jobs = job_manager.list_jobs(status=JobStatus.completed, limit=limit)

    downloads = []
    for job in jobs:
        result_path = storage_service.get_final_result(job.id)
        if result_path and result_path.exists():
            file_size = result_path.stat().st_size
            downloads.append({
                "jobId": job.id,
                "scanId": job.scan_id,
                "name": job.scan_id or job.id,
                "downloadURL": f"/download/{job.id}",
                "fileSize": file_size,
                "fileSizeMB": round(file_size / 1024 / 1024, 2),
                "completedAt": job.updated_at.isoformat() if job.updated_at else None,
                "createdAt": job.created_at.isoformat() if job.created_at else None,
                "quality": job.config.quality_preset if job.config else "balanced",
            })

    return {
        "downloads": downloads,
        "count": len(downloads),
    }


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


@router.get("/{job_id}/stream")
async def stream_job_status(job_id: str):
    """
    Stream real-time job updates via Server-Sent Events (SSE).

    This endpoint provides live progress updates without polling.
    The client receives events every 2 seconds until the job reaches
    a terminal state (completed, failed, or cancelled).
    """
    logger.info(f"SSE stream started for job {job_id}")

    async def event_generator():
        try:
            while True:
                job = job_manager.get_job(job_id)

                if job is None:
                    error_data = json.dumps({"error": "Job not found"})
                    yield f"event: error\ndata: {error_data}\n\n"
                    logger.warning(f"SSE: Job {job_id} not found")
                    break

                # Build status data
                data = {
                    "status": job.status.value,
                    "progress": job.progress,
                    "iteration": job.current_iteration,
                    "total": job.total_iterations,
                    "loss": job.current_loss,
                    "gaussians": job.num_gaussians,
                    "errorMessage": job.error_message,
                    "resultURL": job.result_url,
                }
                json_data = json.dumps(data)
                yield f"data: {json_data}\n\n"

                # Check for terminal state
                if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
                    logger.info(f"SSE: Job {job_id} reached terminal state: {job.status.value}")
                    break

                # Wait before next update
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.info(f"SSE stream cancelled for job {job_id}")
            raise
        except Exception as e:
            logger.error(f"SSE error for job {job_id}: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


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
