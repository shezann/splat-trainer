"""
Storage service for managing training data and outputs.
"""

import os
import shutil
import zipfile
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from config import DATA_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


class StorageService:
    """Handles file storage for training jobs."""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR

    def get_job_data_dir(self, job_id: str) -> Path:
        """Get the data directory for a job."""
        path = self.data_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_job_output_dir(self, job_id: str) -> Path:
        """Get the output directory for a job."""
        path = self.output_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_upload(self, job_id: str, content: bytes, filename: str) -> Path:
        """Save uploaded file to job data directory."""
        job_dir = self.get_job_data_dir(job_id)
        file_path = job_dir / filename

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved {filename} ({len(content)} bytes) for job {job_id}")
        return file_path

    async def extract_zip(self, job_id: str, zip_path: Path) -> Dict[str, Any]:
        """Extract a ZIP file and validate contents."""
        job_dir = self.get_job_data_dir(job_id)
        extract_dir = job_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        files_extracted = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Security: prevent path traversal
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue

                # Check extension
                ext = Path(member).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS and not member.endswith("/"):
                    logger.warning(f"Skipping file with disallowed extension: {member}")
                    continue

                zf.extract(member, extract_dir)
                files_extracted.append(member)

        logger.info(f"Extracted {len(files_extracted)} files for job {job_id}")

        # Validate required files
        validation = self._validate_extracted_data(extract_dir)

        return {
            "extract_dir": extract_dir,
            "files": files_extracted,
            "validation": validation,
        }

    def _validate_extracted_data(self, extract_dir: Path) -> Dict[str, Any]:
        """Validate extracted training data (NeRF or COLMAP format)."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        # Find the actual data directory (might be nested)
        data_dir = self._find_data_dir(extract_dir)
        if data_dir is None:
            result["valid"] = False
            result["errors"].append("Could not find valid data directory")
            return result

        # Detect format
        has_colmap = (data_dir / "sparse" / "0" / "cameras.bin").exists()
        has_nerf = (data_dir / "transforms.json").exists()

        if has_colmap:
            result["stats"]["format"] = "colmap"
        elif has_nerf:
            result["stats"]["format"] = "nerf"
        else:
            result["valid"] = False
            result["errors"].append(
                "Missing training data: need transforms.json (NeRF) "
                "or sparse/0/cameras.bin (COLMAP)"
            )

        # Check for transforms.json (NeRF only)
        if has_nerf and not has_colmap:
            transforms_path = data_dir / "transforms.json"
            try:
                with open(transforms_path) as f:
                    transforms = json.load(f)
                result["stats"]["num_frames"] = len(transforms.get("frames", []))
            except json.JSONDecodeError:
                result["valid"] = False
                result["errors"].append("Invalid transforms.json format")

        # Check for images
        image_exts = {".jpg", ".jpeg", ".png"}
        images = list(data_dir.rglob("*"))
        images = [f for f in images if f.suffix.lower() in image_exts]
        result["stats"]["num_images"] = len(images)

        if len(images) == 0:
            result["valid"] = False
            result["errors"].append("No images found")
        elif len(images) < 10:
            result["warnings"].append(f"Only {len(images)} images found, recommend 20+")

        # Check for optional mesh
        mesh_files = list(data_dir.rglob("*.ply")) + list(data_dir.rglob("*.glb"))
        result["stats"]["has_mesh"] = len(mesh_files) > 0

        return result

    def _find_data_dir(self, extract_dir: Path) -> Optional[Path]:
        """Find the actual data directory within extracted files."""
        # Check if transforms.json or COLMAP data is directly in extract_dir
        if (extract_dir / "transforms.json").exists():
            return extract_dir
        if (extract_dir / "sparse" / "0" / "cameras.bin").exists():
            return extract_dir

        # Check one level deep
        for subdir in extract_dir.iterdir():
            if subdir.is_dir():
                if (subdir / "transforms.json").exists():
                    return subdir
                if (subdir / "sparse" / "0" / "cameras.bin").exists():
                    return subdir

        # Check for images directory pattern
        for subdir in extract_dir.rglob("images"):
            if subdir.is_dir():
                parent = subdir.parent
                if (parent / "transforms.json").exists():
                    return parent
                if (parent / "sparse" / "0" / "cameras.bin").exists():
                    return parent

        return None

    def get_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """List available checkpoints for a job."""
        output_dir = self.get_job_output_dir(job_id)
        checkpoints = []

        # Look for PLY files with iteration numbers
        for ply_file in output_dir.glob("*.ply"):
            # Parse iteration from filename (e.g., "point_cloud_10000.ply")
            name = ply_file.stem
            try:
                if "_" in name:
                    iteration = int(name.split("_")[-1])
                else:
                    iteration = 0

                stat = ply_file.stat()
                checkpoints.append({
                    "iteration": iteration,
                    "path": ply_file,
                    "file_size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime),
                })
            except (ValueError, IndexError):
                continue

        return sorted(checkpoints, key=lambda x: x["iteration"])

    def get_mesh_result(self, job_id: str) -> Optional[Path]:
        """Get the room mesh GLB (from iOS LiDAR/ARKit data) for a job."""
        output_dir = self.get_job_output_dir(job_id)
        mesh_path = output_dir / "mesh.glb"
        if mesh_path.exists():
            return mesh_path
        return None

    def get_final_result(self, job_id: str) -> Optional[Path]:
        """Get the final result for a job, preferring GLB over PLY."""
        output_dir = self.get_job_output_dir(job_id)

        # Look for final output, preferring GLB
        candidates = [
            output_dir / "point_cloud.glb",
            output_dir / "point_cloud.ply",
            output_dir / "final.ply",
            output_dir / "splat.ply",
        ]

        for path in candidates:
            if path.exists():
                return path

        # Fall back to highest iteration checkpoint
        checkpoints = self.get_checkpoints(job_id)
        if checkpoints:
            return checkpoints[-1]["path"]

        return None

    def cleanup_job(self, job_id: str, keep_output: bool = True) -> None:
        """Clean up job data, optionally keeping outputs."""
        # Remove input data
        job_data_dir = self.data_dir / job_id
        if job_data_dir.exists():
            shutil.rmtree(job_data_dir)
            logger.info(f"Cleaned up data for job {job_id}")

        # Optionally remove outputs
        if not keep_output:
            job_output_dir = self.output_dir / job_id
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
                logger.info(f"Cleaned up outputs for job {job_id}")

    def get_disk_usage(self) -> Dict[str, int]:
        """Get disk usage statistics."""

        def dir_size(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        return {
            "data_bytes": dir_size(self.data_dir),
            "output_bytes": dir_size(self.output_dir),
            "total_bytes": dir_size(self.data_dir) + dir_size(self.output_dir),
        }


# Singleton instance
storage_service = StorageService()
