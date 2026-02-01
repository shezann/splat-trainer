"""
3D Gaussian Splatting training service using gsplat.
Implements real differentiable Gaussian splatting for scene reconstruction.
"""

import asyncio
import logging
import math
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

import numpy as np
import torch

from config import CHECKPOINT_ITERATIONS, QUALITY_PRESETS

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Training progress information."""
    iteration: int
    total_iterations: int
    loss: Optional[float] = None
    psnr: Optional[float] = None
    num_gaussians: Optional[int] = None
    elapsed_seconds: Optional[float] = None


@dataclass
class GaussianParams:
    """Container for Gaussian splat parameters."""
    means: torch.Tensor       # (N, 3) positions
    scales: torch.Tensor      # (N, 3) log-scales
    quats: torch.Tensor       # (N, 4) quaternions (wxyz)
    opacities: torch.Tensor   # (N,) logit-opacities
    sh0: torch.Tensor         # (N, 3) DC spherical harmonics (color)
    sh_rest: Optional[torch.Tensor] = None  # (N, K, 3) higher-order SH

    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_params_list(self) -> List[torch.Tensor]:
        """Get list of all parameter tensors."""
        params = [self.means, self.scales, self.quats, self.opacities, self.sh0]
        if self.sh_rest is not None:
            params.append(self.sh_rest)
        return params


class GaussianSplatTrainer:
    """
    3D Gaussian Splatting trainer using gsplat for differentiable rendering.
    """

    def __init__(self):
        self.cancelled = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def train(
        self,
        data_dir: Path,
        output_dir: Path,
        iterations: int = 30000,
        quality_preset: str = "balanced",
        use_mesh_init: bool = True,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run 3D Gaussian Splatting training.

        Args:
            data_dir: Directory containing transforms.json and images
            output_dir: Directory for output files
            iterations: Number of training iterations
            quality_preset: Quality preset (fast, balanced, high)
            use_mesh_init: Whether to use mesh for initialization
            progress_callback: Callback for progress updates

        Returns:
            Dict with training results
        """
        self.cancelled = False
        output_dir.mkdir(parents=True, exist_ok=True)

        preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])

        # Find transforms.json
        transforms_path = self._find_transforms(data_dir)
        if transforms_path is None:
            raise ValueError(f"No transforms.json found in {data_dir}")

        actual_data_dir = transforms_path.parent

        logger.info(f"Starting gsplat training: {actual_data_dir} -> {output_dir}")
        logger.info(f"Device: {self.device}, iterations: {iterations}, preset: {quality_preset}")

        result = await self._train_gsplat(
            actual_data_dir,
            output_dir,
            iterations,
            preset,
            progress_callback,
        )

        return result

    def _find_transforms(self, data_dir: Path) -> Optional[Path]:
        """Find transforms.json in the data directory."""
        if (data_dir / "transforms.json").exists():
            return data_dir / "transforms.json"

        for path in data_dir.rglob("transforms.json"):
            return path

        return None

    async def _train_gsplat(
        self,
        data_dir: Path,
        output_dir: Path,
        iterations: int,
        preset: Dict[str, Any],
        progress_callback: Optional[Callable[[TrainingProgress], None]],
    ) -> Dict[str, Any]:
        """
        Train Gaussians using gsplat library.
        """
        from services.camera_utils import load_training_data
        from services.loss_utils import combined_loss, compute_psnr

        result = {
            "success": False,
            "iterations_completed": 0,
            "final_loss": None,
            "final_psnr": None,
            "num_gaussians": 0,
            "output_path": None,
            "error": None,
        }

        start_time = time.time()

        try:
            # Import gsplat
            try:
                from gsplat.rendering import rasterization
                from gsplat.strategy import DefaultStrategy
            except ImportError as e:
                logger.error(f"gsplat not available: {e}")
                raise ImportError(
                    "gsplat library not installed. Install with: pip install gsplat"
                ) from e

            # Load training data
            logger.info("Loading training data...")
            data = load_training_data(data_dir, device=self.device, image_scale=0.5)

            viewmats = data["viewmats"]  # (N, 4, 4)
            Ks = data["Ks"]              # (N, 3, 3)
            images = data["images"]       # (N, H, W, 3)
            init_points = data["initial_points"]
            width = data["width"]
            height = data["height"]

            num_images = len(images)
            logger.info(f"Loaded {num_images} images at {width}x{height}")
            logger.info(f"Initial points: {len(init_points)}")

            # Initialize Gaussian parameters
            params = self._init_gaussians(init_points, preset.get("sh_degree", 3))
            logger.info(f"Initialized {params.num_gaussians()} Gaussians")

            # Setup optimizer with per-parameter learning rates
            optimizer = self._create_optimizer(params)

            # Setup densification strategy
            strategy = DefaultStrategy(
                refine_start_iter=500,
                refine_stop_iter=int(iterations * 0.5),
                refine_every=100,
                prune_opa=0.005,
                grow_grad2d=0.0002,
                grow_scale3d=0.01,
                prune_scale3d=0.1,
                pause_refine_after_reset=0,
                absgrad=True,  # Must match rasterization absgrad parameter
                verbose=True,
            )
            strategy_state = strategy.initialize_state()

            # Training loop
            logger.info(f"Starting training for {iterations} iterations")

            for step in range(1, iterations + 1):
                if self.cancelled:
                    result["error"] = "Training cancelled"
                    return result

                # Pre-backward step for densification
                strategy.step_pre_backward(
                    params=self._params_dict(params),
                    optimizers={"default": optimizer},
                    state=strategy_state,
                    step=step,
                    info={},
                )

                # Random camera selection
                cam_idx = random.randint(0, num_images - 1)

                # Get camera parameters
                viewmat = viewmats[cam_idx:cam_idx + 1]  # (1, 4, 4)
                K = Ks[cam_idx:cam_idx + 1]              # (1, 3, 3)
                gt_image = images[cam_idx]               # (H, W, 3)

                # Render - absgrad=True required for DefaultStrategy densification
                renders, alphas, info = rasterization(
                    means=params.means,
                    quats=params.quats / params.quats.norm(dim=-1, keepdim=True),
                    scales=torch.exp(params.scales),
                    opacities=torch.sigmoid(params.opacities),
                    colors=self._sh_to_rgb(params, viewmat),
                    viewmats=viewmat,
                    Ks=K,
                    width=width,
                    height=height,
                    packed=False,
                    render_mode="RGB",
                    absgrad=True,
                )

                # renders shape: (1, H, W, 3)
                pred_image = renders[0]

                # Compute loss
                loss = combined_loss(pred_image, gt_image, lambda_l1=0.8, lambda_ssim=0.2)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(params.get_params_list(), max_norm=1.0)

                optimizer.step()

                # Post-backward step for densification
                strategy.step_post_backward(
                    params=self._params_dict(params),
                    optimizers={"default": optimizer},
                    state=strategy_state,
                    step=step,
                    info=info,
                    packed=False,
                )

                # Update params if densification changed them
                params = self._update_params_from_dict(params, self._params_dict(params))

                # Progress reporting
                if step % 100 == 0 or step == 1:
                    psnr = compute_psnr(pred_image.detach(), gt_image)
                    elapsed = time.time() - start_time

                    progress = TrainingProgress(
                        iteration=step,
                        total_iterations=iterations,
                        loss=loss.item(),
                        psnr=psnr,
                        num_gaussians=params.num_gaussians(),
                        elapsed_seconds=elapsed,
                    )

                    logger.info(
                        f"Step {step}/{iterations}: loss={loss.item():.4f}, "
                        f"PSNR={psnr:.2f}dB, gaussians={params.num_gaussians()}"
                    )

                    if progress_callback:
                        progress_callback(progress)

                    result["iterations_completed"] = step
                    result["final_loss"] = loss.item()
                    result["final_psnr"] = psnr
                    result["num_gaussians"] = params.num_gaussians()

                # Save checkpoints
                if step in CHECKPOINT_ITERATIONS:
                    self._log_system_status()
                    checkpoint_path = output_dir / f"point_cloud_{step}.ply"
                    if self._save_ply(checkpoint_path, params):
                        logger.info(f"Saved checkpoint: {checkpoint_path}")
                    else:
                        logger.warning(f"Checkpoint save failed: {checkpoint_path}")

                # Yield to event loop periodically
                if step % 10 == 0:
                    await asyncio.sleep(0)

            # Save final result
            self._log_system_status()
            final_path = output_dir / "point_cloud.ply"
            if self._save_ply(final_path, params):
                logger.info(f"Saved final result: {final_path}")
                result["success"] = True
                result["output_path"] = str(final_path)
            else:
                logger.error("Failed to save final PLY - output validation failed")
                result["success"] = False
                result["error"] = "Failed to save output file - validation failed"

        except Exception as e:
            logger.exception(f"Training error: {e}")
            result["error"] = str(e)

        return result

    def _init_gaussians(
        self,
        points: torch.Tensor,
        sh_degree: int = 3,
    ) -> GaussianParams:
        """Initialize Gaussian parameters from point cloud."""
        num_points = len(points)

        # Positions
        means = points.clone().requires_grad_(True)

        # Scales (log-space, initialized small)
        scales = torch.zeros(num_points, 3, device=self.device)
        scales.fill_(-3.0)  # exp(-3) ≈ 0.05
        scales = scales.requires_grad_(True)

        # Rotations (quaternions, initialized to identity)
        quats = torch.zeros(num_points, 4, device=self.device)
        quats[:, 0] = 1.0  # w=1, x=y=z=0
        quats = quats.requires_grad_(True)

        # Opacities (logit-space, initialized to ~0.1)
        opacities = torch.zeros(num_points, device=self.device)
        opacities.fill_(-2.2)  # sigmoid(-2.2) ≈ 0.1
        opacities = opacities.requires_grad_(True)

        # Spherical harmonics DC component (RGB color)
        sh0 = torch.zeros(num_points, 3, device=self.device)
        sh0.fill_(0.5)  # Initialize to gray
        sh0 = sh0.requires_grad_(True)

        # Higher order SH (optional)
        sh_rest = None
        if sh_degree > 0:
            num_sh_coeffs = (sh_degree + 1) ** 2 - 1
            sh_rest = torch.zeros(num_points, num_sh_coeffs, 3, device=self.device)
            sh_rest = sh_rest.requires_grad_(True)

        return GaussianParams(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            sh_rest=sh_rest,
        )

    def _create_optimizer(self, params: GaussianParams) -> torch.optim.Optimizer:
        """Create Adam optimizer with per-parameter learning rates."""
        param_groups = [
            {"params": [params.means], "lr": 0.00016, "name": "means"},
            {"params": [params.scales], "lr": 0.005, "name": "scales"},
            {"params": [params.quats], "lr": 0.001, "name": "quats"},
            {"params": [params.opacities], "lr": 0.05, "name": "opacities"},
            {"params": [params.sh0], "lr": 0.0025, "name": "sh0"},
        ]

        if params.sh_rest is not None:
            param_groups.append({
                "params": [params.sh_rest],
                "lr": 0.0025 / 20,
                "name": "sh_rest"
            })

        return torch.optim.Adam(param_groups)

    def _params_dict(self, params: GaussianParams) -> Dict[str, torch.Tensor]:
        """Convert params to dict for gsplat strategy."""
        d = {
            "means": params.means,
            "scales": params.scales,
            "quats": params.quats,
            "opacities": params.opacities,
            "sh0": params.sh0,
        }
        if params.sh_rest is not None:
            d["sh_rest"] = params.sh_rest
        return d

    def _update_params_from_dict(
        self,
        params: GaussianParams,
        d: Dict[str, torch.Tensor]
    ) -> GaussianParams:
        """Update params from dict (after densification)."""
        return GaussianParams(
            means=d["means"],
            scales=d["scales"],
            quats=d["quats"],
            opacities=d["opacities"],
            sh0=d["sh0"],
            sh_rest=d.get("sh_rest"),
        )

    def _sh_to_rgb(
        self,
        params: GaussianParams,
        viewmat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert spherical harmonics to RGB colors.
        For now, just return the DC component (view-independent).
        Full SH evaluation would compute view-dependent colors.
        """
        # Simple: just use DC component
        # Full implementation would evaluate SH with view directions
        colors = params.sh0

        # Clamp to valid range
        colors = torch.clamp(colors, 0.0, 1.0)

        return colors

    def _save_ply(self, path: Path, params: GaussianParams) -> bool:
        """Save Gaussians as PLY file in 3DGS format with atomic write.

        Returns:
            True if save and validation succeeded, False otherwise.
        """
        from plyfile import PlyData, PlyElement

        num_points = params.num_gaussians()

        if num_points == 0:
            logger.error("Cannot save PLY: no Gaussians to save")
            return False

        # Detach and move to CPU
        means = params.means.detach().cpu().numpy()
        scales = params.scales.detach().cpu().numpy()
        quats = params.quats.detach().cpu().numpy()
        opacities = params.opacities.detach().cpu().numpy()
        sh0 = params.sh0.detach().cpu().numpy()

        # Normalize quaternions
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # Build dtype for PLY
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]

        # Add higher-order SH if present
        if params.sh_rest is not None:
            sh_rest = params.sh_rest.detach().cpu().numpy()
            num_sh = sh_rest.shape[1]
            for i in range(num_sh):
                for c in range(3):
                    dtype.append((f"f_rest_{i * 3 + c}", "f4"))

        dtype.extend([
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ])

        # Create structured array
        elements = np.zeros(num_points, dtype=dtype)

        # Positions
        elements["x"] = means[:, 0]
        elements["y"] = means[:, 1]
        elements["z"] = means[:, 2]

        # Normals (unused, but required by format)
        elements["nx"] = 0
        elements["ny"] = 0
        elements["nz"] = 0

        # DC spherical harmonics (color)
        # Convert from [0,1] to SH coefficient space
        C0 = 0.28209479177387814
        elements["f_dc_0"] = (sh0[:, 0] - 0.5) / C0
        elements["f_dc_1"] = (sh0[:, 1] - 0.5) / C0
        elements["f_dc_2"] = (sh0[:, 2] - 0.5) / C0

        # Higher-order SH
        if params.sh_rest is not None:
            sh_rest = params.sh_rest.detach().cpu().numpy()
            for i in range(sh_rest.shape[1]):
                for c in range(3):
                    elements[f"f_rest_{i * 3 + c}"] = sh_rest[:, i, c]

        # Opacity (convert from logit to log-space for PLY format)
        # PLY stores inverse-sigmoid, viewer applies sigmoid
        elements["opacity"] = opacities

        # Scales (already in log-space)
        elements["scale_0"] = scales[:, 0]
        elements["scale_1"] = scales[:, 1]
        elements["scale_2"] = scales[:, 2]

        # Rotations (quaternion wxyz)
        elements["rot_0"] = quats[:, 0]
        elements["rot_1"] = quats[:, 1]
        elements["rot_2"] = quats[:, 2]
        elements["rot_3"] = quats[:, 3]

        # Write to temp file first for atomic operation
        temp_path = path.with_suffix('.ply.tmp')
        try:
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(str(temp_path))

            # Validate the temp file before renaming
            if not self._validate_ply(temp_path, num_points):
                temp_path.unlink(missing_ok=True)
                return False

            # Atomic rename (move temp to final path)
            shutil.move(str(temp_path), str(path))
            logger.info(f"Saved PLY: {path} ({num_points} Gaussians)")
            return True

        except Exception as e:
            logger.exception(f"Failed to save PLY: {e}")
            temp_path.unlink(missing_ok=True)
            return False

    def _validate_ply(self, path: Path, expected_vertices: int) -> bool:
        """Verify PLY file is valid and complete.

        Args:
            path: Path to the PLY file.
            expected_vertices: Expected number of vertices.

        Returns:
            True if valid, False otherwise.
        """
        try:
            if not path.exists():
                logger.error("PLY validation failed: file doesn't exist")
                return False

            file_size = path.stat().st_size
            # Estimate: ~62 bytes per vertex for standard 3DGS format + ~500 byte header
            # With SH, can be up to ~250 bytes per vertex
            expected_min_size = expected_vertices * 50 + 500  # Conservative minimum

            if file_size < expected_min_size * 0.9:  # Allow 10% tolerance
                logger.error(
                    f"PLY validation failed: file size {file_size} < expected minimum {expected_min_size}"
                )
                return False

            # Try to read back and verify vertex count
            from plyfile import PlyData
            plydata = PlyData.read(str(path))
            actual_vertices = len(plydata["vertex"])

            if actual_vertices != expected_vertices:
                logger.error(
                    f"PLY validation failed: {actual_vertices} vertices != expected {expected_vertices}"
                )
                return False

            logger.info(f"PLY validated: {actual_vertices} vertices, {file_size} bytes")
            return True

        except Exception as e:
            logger.exception(f"PLY validation error: {e}")
            return False

    def _log_system_status(self):
        """Log memory and GPU status for diagnostics."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(
                    f"GPU Memory: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved, {total:.2f}GB total"
                )

            # Log disk space
            disk = shutil.disk_usage("/")
            free_gb = disk.free / 1024**3
            logger.info(f"Disk space: {free_gb:.2f}GB free")
        except Exception as e:
            logger.warning(f"Could not log system status: {e}")

    def cancel(self):
        """Cancel the current training."""
        self.cancelled = True
        logger.info("Training cancellation requested")
