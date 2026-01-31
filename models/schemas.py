"""
Pydantic models for the API.
These match the expected format from the iOS CloudTrainingService.
"""

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, List


class JobStatus(str, Enum):
    """Training job status values."""
    pending = "pending"
    uploading = "uploading"
    queued = "queued"
    processing = "processing"  # Alias for training
    training = "training"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TrainingConfig(BaseModel):
    """Configuration for a training job."""
    iterations: int = Field(default=30000, ge=1000, le=100000)
    target_gaussians: int = Field(default=500000, ge=10000, le=5000000)
    quality_preset: str = Field(default="balanced")
    use_mesh_init: bool = Field(default=True)
    sh_degree: int = Field(default=3, ge=0, le=4)


class JobCreateRequest(BaseModel):
    """Request to create a new training job."""
    scan_id: Optional[str] = None
    iterations: Optional[int] = None
    targetGaussians: Optional[int] = None
    qualityPreset: Optional[str] = None
    useMeshInit: Optional[bool] = None

    def to_config(self) -> TrainingConfig:
        """Convert request to TrainingConfig."""
        config = TrainingConfig()
        if self.iterations:
            config.iterations = self.iterations
        if self.targetGaussians:
            config.target_gaussians = self.targetGaussians
        if self.qualityPreset:
            config.quality_preset = self.qualityPreset
        if self.useMeshInit is not None:
            config.use_mesh_init = self.useMeshInit
        return config


class JobCreateResponse(BaseModel):
    """Response after creating a job - matches iOS CreateJobResponse."""
    jobId: str
    uploadURL: str


class TrainingJob(BaseModel):
    """Full training job details."""
    id: str
    scan_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    config: TrainingConfig
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CheckpointInfo(BaseModel):
    """Information about a training checkpoint."""
    iteration: int
    file_size: int
    download_url: str
    created_at: datetime
    num_gaussians: Optional[int] = None
    psnr: Optional[float] = None


class CheckpointListResponse(BaseModel):
    """List of available checkpoints for a job."""
    job_id: str
    checkpoints: List[CheckpointInfo]


class UploadResponse(BaseModel):
    """Response after uploading data."""
    job_id: str
    status: JobStatus
    message: str
    files_received: int


class TrainingJobResponse(BaseModel):
    """Training job response matching iOS TrainingJob model."""
    id: str
    status: JobStatus
    progress: float
    currentIteration: Optional[int] = None
    message: Optional[str] = None
    createdAt: str
    updatedAt: str
    resultURL: Optional[str] = None
    errorMessage: Optional[str] = None
    estimatedTimeRemaining: Optional[float] = None
    availableCheckpoints: Optional[List[CheckpointInfo]] = None

    @classmethod
    def from_job(cls, job: TrainingJob) -> "TrainingJobResponse":
        return cls(
            id=job.id,
            status=job.status,
            progress=job.progress,
            currentIteration=job.current_iteration,
            message=f"Iteration {job.current_iteration}/{job.total_iterations}" if job.current_iteration else None,
            createdAt=job.created_at.isoformat(),
            updatedAt=job.updated_at.isoformat(),
            resultURL=job.result_url,
            errorMessage=job.error_message,
            estimatedTimeRemaining=None,
            availableCheckpoints=None,
        )


class JobStatusResponse(BaseModel):
    """Response wrapper matching iOS JobStatusResponse."""
    job: TrainingJobResponse

    @classmethod
    def from_job(cls, job: TrainingJob) -> "JobStatusResponse":
        return cls(job=TrainingJobResponse.from_job(job))
