"""
Job manager for training queue.
Handles job lifecycle, queuing, and status tracking.
"""

import asyncio
import uuid
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, List
from collections import OrderedDict

from models.schemas import JobStatus, TrainingConfig, TrainingJob
from services.storage import storage_service
from services.training import GaussianSplatTrainer, TrainingProgress

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages training jobs with a single-worker queue.
    Jobs are processed sequentially (one GPU training at a time).
    """

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = OrderedDict()
        self.pending_jobs: List[str] = []
        self.current_job: Optional[str] = None
        self.trainer: Optional[GaussianSplatTrainer] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

    def create_job(
        self,
        scan_id: str,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingJob:
        """Create a new training job."""
        job_id = str(uuid.uuid4())[:8]

        if config is None:
            config = TrainingConfig()

        now = datetime.utcnow()
        job = TrainingJob(
            id=job_id,
            scan_id=scan_id,
            status=JobStatus.pending,
            progress=0.0,
            config=config,
            created_at=now,
            updated_at=now,
            total_iterations=config.iterations,
        )

        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} for scan {scan_id}")

        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        current_iteration: Optional[int] = None,
        result_url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TrainingJob]:
        """Update job status and progress."""
        job = self.jobs.get(job_id)
        if job is None:
            return None

        job.status = status
        job.updated_at = datetime.utcnow()

        if progress is not None:
            job.progress = progress
        if current_iteration is not None:
            job.current_iteration = current_iteration
        if result_url is not None:
            job.result_url = result_url
        if error_message is not None:
            job.error_message = error_message

        if status == JobStatus.training and job.started_at is None:
            job.started_at = datetime.utcnow()
        elif status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
            job.completed_at = datetime.utcnow()

        logger.info(f"Job {job_id}: {status.value} ({job.progress:.1%})")
        return job

    def queue_job(self, job_id: str) -> bool:
        """Add a job to the training queue."""
        job = self.jobs.get(job_id)
        if job is None:
            return False

        if job_id not in self.pending_jobs:
            self.pending_jobs.append(job_id)
            self.update_job_status(job_id, JobStatus.queued)
            logger.info(f"Queued job {job_id}, position {len(self.pending_jobs)}")

        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if job is None:
            return False

        # Remove from queue if pending
        if job_id in self.pending_jobs:
            self.pending_jobs.remove(job_id)
            self.update_job_status(job_id, JobStatus.cancelled)
            return True

        # Cancel if currently training
        if self.current_job == job_id and self.trainer:
            self.trainer.cancel()
            self.update_job_status(job_id, JobStatus.cancelled)
            return True

        # Can't cancel completed/failed jobs
        if job.status in (JobStatus.completed, JobStatus.failed):
            return False

        self.update_job_status(job_id, JobStatus.cancelled)
        return True

    def start_worker(self):
        """Start the background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Started job worker thread")

    def stop_worker(self):
        """Stop the background worker."""
        self._stop_event.set()
        if self.trainer:
            self.trainer.cancel()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Stopped job worker")

    def _run_worker(self):
        """Worker thread that processes jobs."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._worker_loop())
        finally:
            self._loop.close()

    async def _worker_loop(self):
        """Main worker loop that processes queued jobs."""
        logger.info("Job worker started")

        while not self._stop_event.is_set():
            # Check for pending jobs
            if not self.pending_jobs:
                await asyncio.sleep(1)
                continue

            # Get next job
            job_id = self.pending_jobs.pop(0)
            job = self.jobs.get(job_id)

            if job is None or job.status == JobStatus.cancelled:
                continue

            # Process job
            await self._process_job(job_id)

    async def _process_job(self, job_id: str):
        """Process a single training job."""
        job = self.jobs.get(job_id)
        if job is None:
            return

        self.current_job = job_id
        self.trainer = GaussianSplatTrainer()

        logger.info(f"Starting training for job {job_id}")
        self.update_job_status(job_id, JobStatus.training)

        try:
            # Get paths
            data_dir = storage_service.get_job_data_dir(job_id) / "extracted"
            output_dir = storage_service.get_job_output_dir(job_id)

            # Progress callback
            def on_progress(progress: TrainingProgress):
                pct = progress.iteration / progress.total_iterations
                self.update_job_status(
                    job_id,
                    JobStatus.training,
                    progress=pct,
                    current_iteration=progress.iteration,
                )

            # Run training
            result = await self.trainer.train(
                data_dir=data_dir,
                output_dir=output_dir,
                iterations=job.config.iterations,
                quality_preset=job.config.quality_preset,
                use_mesh_init=job.config.use_mesh_init,
                progress_callback=on_progress,
            )

            if result["success"]:
                # Generate download URL
                result_url = f"/download/{job_id}"
                self.update_job_status(
                    job_id,
                    JobStatus.completed,
                    progress=1.0,
                    current_iteration=job.config.iterations,
                    result_url=result_url,
                )
                logger.info(f"Job {job_id} completed successfully")
            else:
                self.update_job_status(
                    job_id,
                    JobStatus.failed,
                    error_message=result.get("error", "Unknown error"),
                )
                logger.error(f"Job {job_id} failed: {result.get('error')}")

        except Exception as e:
            logger.exception(f"Error processing job {job_id}")
            self.update_job_status(
                job_id,
                JobStatus.failed,
                error_message=str(e),
            )

        finally:
            self.current_job = None
            self.trainer = None


# Singleton instance
job_manager = JobManager()
