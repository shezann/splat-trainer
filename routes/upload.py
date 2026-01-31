"""
File upload and download endpoints.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse

from models.schemas import JobStatus, UploadResponse
from services.job_manager import job_manager
from services.storage import storage_service
from config import MAX_UPLOAD_SIZE

logger = logging.getLogger(__name__)

router = APIRouter()


@router.put("/upload/{job_id}", response_model=UploadResponse)
async def upload_training_data(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks = None,
):
    """
    Upload training data (ZIP file) for a job.

    Accepts raw bytes with Content-Type: application/zip (or multipart/form-data).

    The ZIP should contain:
    - transforms.json: Camera transforms in NeRF format
    - images/: Directory of training images
    - (optional) mesh.ply or mesh.glb: Initial mesh for Gaussian seeding
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in (JobStatus.pending, JobStatus.uploading):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot upload to job with status: {job.status}",
        )

    # Update status to uploading
    job_manager.update_job_status(job_id, JobStatus.uploading)

    # Read raw bytes from request body
    try:
        content = await request.body()
    except Exception as e:
        job_manager.update_job_status(job_id, JobStatus.failed, error_message=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to read request body: {e}")

    # Check size
    if len(content) > MAX_UPLOAD_SIZE:
        job_manager.update_job_status(
            job_id,
            JobStatus.failed,
            error_message="File too large",
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB",
        )

    # Get filename from Content-Disposition header or default
    content_disposition = request.headers.get("Content-Disposition", "")
    filename = "upload.zip"
    if "filename=" in content_disposition:
        # Parse filename from header (e.g., 'attachment; filename="data.zip"')
        try:
            filename = content_disposition.split("filename=")[1].strip('"').strip("'")
        except (IndexError, AttributeError):
            pass  # Keep default filename
    file_path = await storage_service.save_upload(job_id, content, filename)

    # Extract if ZIP
    if filename.endswith(".zip"):
        try:
            result = await storage_service.extract_zip(job_id, file_path)

            if not result["validation"]["valid"]:
                errors = result["validation"]["errors"]
                job_manager.update_job_status(
                    job_id,
                    JobStatus.failed,
                    error_message=f"Invalid data: {', '.join(errors)}",
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid training data: {errors}",
                )

            files_count = len(result["files"])
            stats = result["validation"]["stats"]
            logger.info(
                f"Job {job_id}: Extracted {files_count} files, "
                f"{stats.get('num_images', 0)} images, "
                f"{stats.get('num_frames', 0)} frames"
            )

        except HTTPException:
            raise
        except Exception as e:
            job_manager.update_job_status(
                job_id,
                JobStatus.failed,
                error_message=str(e),
            )
            raise HTTPException(status_code=400, detail=f"Failed to extract ZIP: {e}")

        # Queue the job for training
        job_manager.queue_job(job_id)

        return UploadResponse(
            job_id=job_id,
            status=JobStatus.queued,
            message="Data uploaded and queued for training",
            files_received=files_count,
        )
    else:
        # Single file upload (for incremental uploads)
        return UploadResponse(
            job_id=job_id,
            status=JobStatus.uploading,
            message=f"File {filename} received",
            files_received=1,
        )


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """
    Download the final training result (PLY file).
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}",
        )

    result_path = storage_service.get_final_result(job_id)
    if result_path is None or not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=result_path,
        filename=f"{job_id}_gaussians.ply",
        media_type="application/octet-stream",
    )


@router.get("/download/{job_id}/checkpoint/{iteration}")
async def download_checkpoint_file(job_id: str, iteration: int):
    """
    Download a specific checkpoint PLY file.
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = storage_service.get_job_output_dir(job_id)
    checkpoint_path = output_dir / f"point_cloud_{iteration}.ply"

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return FileResponse(
        path=checkpoint_path,
        filename=f"{job_id}_checkpoint_{iteration}.ply",
        media_type="application/octet-stream",
    )


@router.delete("/jobs/{job_id}/data")
async def cleanup_job_data(job_id: str, keep_output: bool = True):
    """
    Clean up job data to free disk space.

    Args:
        keep_output: If True, keeps the output PLY files
    """
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in (JobStatus.training, JobStatus.uploading):
        raise HTTPException(
            status_code=400,
            detail="Cannot cleanup active job",
        )

    storage_service.cleanup_job(job_id, keep_output=keep_output)

    return {
        "status": "cleaned",
        "job_id": job_id,
        "kept_output": keep_output,
    }


@router.get("/storage/stats")
async def storage_stats():
    """Get storage usage statistics."""
    usage = storage_service.get_disk_usage()

    return {
        "data_mb": usage["data_bytes"] / 1024 / 1024,
        "output_mb": usage["output_bytes"] / 1024 / 1024,
        "total_mb": usage["total_bytes"] / 1024 / 1024,
    }
