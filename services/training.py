"""
3D Gaussian Splatting training service using gsplat.
Implements real differentiable Gaussian splatting for scene reconstruction.
"""

import asyncio
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
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
    sh0: torch.Tensor         # (N, 3) SH degree-0 coefficients (RGB channels)
    sh_rest: Optional[torch.Tensor] = None  # (N, K, 3) higher-order SH

    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_params_list(self) -> List[torch.Tensor]:
        """Get list of all parameter tensors."""
        params = [self.means, self.scales, self.quats, self.opacities, self.sh0]
        if self.sh_rest is not None:
            params.append(self.sh_rest)
        return params


def _get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_mult: float = 1.0,
    lr_delay_steps: int = 0,
    max_steps: int = 1000000,
):
    """Continuous learning rate decay function (from Plenoxels/JaxNeRF).

    Returns lr_init at step=0, lr_final at step=max_steps, with log-linear
    interpolation in between.  Optionally applies a cosine warmup ramp
    controlled by lr_delay_steps / lr_delay_mult.
    """
    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(step / lr_delay_steps, 1.0)
            )
        else:
            delay_rate = 1.0
        t = min(step / max_steps, 1.0)
        log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


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
        output_format: str = "both",
    ) -> Dict[str, Any]:
        """
        Run 3D Gaussian Splatting training.

        Args:
            data_dir: Directory containing training data (COLMAP or NeRF)
            output_dir: Directory for output files
            iterations: Number of training iterations
            quality_preset: Quality preset (fast, balanced, high)
            use_mesh_init: Whether to use mesh for initialization
            progress_callback: Callback for progress updates
            output_format: Output format - "ply", "glb", or "both"

        Returns:
            Dict with training results
        """
        self.cancelled = False
        output_dir.mkdir(parents=True, exist_ok=True)

        preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["balanced"])

        # Find training data source (COLMAP or NeRF)
        actual_data_dir = self._find_data_source(data_dir)
        if actual_data_dir is None:
            raise ValueError(f"No training data found in {data_dir}")

        logger.info(f"Starting gsplat training: {actual_data_dir} -> {output_dir}")
        logger.info(f"Device: {self.device}, iterations: {iterations}, preset: {quality_preset}")

        result = await self._train_gsplat(
            actual_data_dir,
            output_dir,
            iterations,
            preset,
            progress_callback,
            output_format,
        )

        return result

    def _find_data_source(self, data_dir: Path) -> Optional[Path]:
        """Find training data directory, supporting both COLMAP and NeRF formats."""
        # Check for COLMAP format
        if (data_dir / "sparse" / "0" / "cameras.bin").exists():
            return data_dir

        # Check for NeRF transforms.json
        if (data_dir / "transforms.json").exists():
            return data_dir

        # Search one level deep
        for path in data_dir.rglob("transforms.json"):
            return path.parent

        for subdir in data_dir.iterdir():
            if subdir.is_dir() and (subdir / "sparse" / "0" / "cameras.bin").exists():
                return subdir

        return None

    async def _train_gsplat(
        self,
        data_dir: Path,
        output_dir: Path,
        iterations: int,
        preset: Dict[str, Any],
        progress_callback: Optional[Callable[[TrainingProgress], None]],
        output_format: str = "both",
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
            # Export iOS LiDAR mesh to GLB before training starts.
            # mesh_hint.ply is produced by ARKit scene reconstruction and is a
            # real triangle mesh of the room — completely separate from the
            # Gaussian splat we train. Export it immediately so the caller can
            # download a clean room mesh even if training is still running.
            try:
                from services.mesh_utils import export_ios_mesh_glb
                mesh_glb = export_ios_mesh_glb(data_dir, output_dir)
                if mesh_glb:
                    result["mesh_output_path"] = str(mesh_glb)
                    logger.info(f"Room mesh GLB ready: {mesh_glb}")
            except Exception as mesh_exc:
                logger.warning(f"Mesh export skipped: {mesh_exc}")

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
            image_scale = float(os.environ.get("SPLAT_IMAGE_SCALE", "0.5"))
            data = load_training_data(data_dir, device=self.device, image_scale=image_scale)

            viewmats = data["viewmats"]  # (N, 4, 4)
            Ks = data["Ks"]              # (N, 3, 3)
            images = data["images"]       # (N, H, W, 3)
            init_points = data["initial_points"]
            init_colors = data.get("initial_colors")
            scene_scale = float(data.get("scene_scale", 1.0))
            width = data["width"]
            height = data["height"]

            num_images = len(images)
            logger.info(f"Loaded {num_images} images at {width}x{height}")
            logger.info(f"Initial points: {len(init_points)}")
            logger.info(f"Estimated scene scale: {scene_scale:.4f}")

            # Initialize Gaussian parameters
            max_sh_degree = int(preset.get("sh_degree", 3))
            params = self._init_gaussians(
                init_points,
                sh_degree=max_sh_degree,
                init_colors=init_colors,
                scene_scale=scene_scale,
            )
            # Keep a persistent params dict so strategy-driven tensor replacement
            # (densify/prune) survives across iterations.
            strategy_params = self._params_dict(params)
            logger.info(f"Initialized {params.num_gaussians()} Gaussians")
            logger.info(
                f"Initial positions: min={params.means.min().item():.4f}, "
                f"max={params.means.max().item():.4f}, "
                f"mean={params.means.mean().item():.4f}"
            )

            # Setup one optimizer per trainable parameter (required by gsplat strategy)
            optimizers = self._create_optimizers(
                params, scene_scale=scene_scale, max_steps=iterations
            )

            # Setup densification strategy
            # Match reference behavior more closely: late start + bounded window.
            if iterations >= 1500:
                refine_start_iter = 500
            else:
                refine_start_iter = max(50, int(iterations * 0.33))
            refine_stop_iter = min(
                15000,
                max(refine_start_iter + 1, int(iterations * 0.9)),
            )
            reset_every = min(3000, max(200, int(iterations * 0.33)))
            logger.info(
                "Densification: start=%d stop=%d every=%d reset_every=%d",
                refine_start_iter,
                refine_stop_iter,
                100,
                reset_every,
            )
            strategy = DefaultStrategy(
                refine_start_iter=refine_start_iter,
                refine_stop_iter=refine_stop_iter,
                refine_every=100,
                reset_every=reset_every,
                prune_opa=0.005,
                grow_grad2d=0.0002,
                grow_scale3d=0.01,
                prune_scale3d=0.1,
                pause_refine_after_reset=min(num_images, 300),
                absgrad=False,
                verbose=True,
            )
            strategy.check_sanity(strategy_params, optimizers)
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)

            # Training loop
            logger.info(f"Starting training for {iterations} iterations")

            # Epoch-based camera cycling: shuffle and pop, ensuring all
            # cameras are seen once before any is repeated (matches reference).
            cam_indices_pool: List[int] = []

            for step in range(1, iterations + 1):
                if self.cancelled:
                    result["error"] = "Training cancelled"
                    return result

                # Update position learning rate (exponential decay)
                self._update_learning_rate(optimizers, step)

                # Epoch-based camera selection (no replacement within epoch)
                if not cam_indices_pool:
                    cam_indices_pool = list(range(num_images))
                    random.shuffle(cam_indices_pool)
                cam_idx = cam_indices_pool.pop()

                # Get camera parameters
                viewmat = viewmats[cam_idx:cam_idx + 1]  # (1, 4, 4)
                K = Ks[cam_idx:cam_idx + 1]              # (1, 3, 3)
                gt_image = images[cam_idx]               # (H, W, 3)

                sh_coeffs, active_sh_degree = self._build_sh_coeffs(
                    params, max_sh_degree, step
                )

                # Render
                renders, alphas, info = rasterization(
                    means=params.means,
                    quats=(
                        params.quats
                        / params.quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    ),
                    scales=torch.exp(params.scales),
                    opacities=torch.sigmoid(params.opacities),
                    colors=sh_coeffs,
                    sh_degree=active_sh_degree,
                    viewmats=viewmat,
                    Ks=K,
                    width=width,
                    height=height,
                    packed=False,
                    render_mode="RGB",
                    absgrad=False,
                )

                # renders shape: (1, H, W, 3)
                # Clamp to [0,1] before loss (matches reference, prevents
                # spurious gradients from out-of-range values).
                pred_image = renders[0].clamp(0, 1)

                # Pre-backward step for densification - must be after rasterization
                # to have means2d available, but before loss.backward()
                strategy.step_pre_backward(
                    params=strategy_params,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                )

                # Compute reconstruction loss
                loss = combined_loss(pred_image, gt_image, lambda_l1=0.8, lambda_ssim=0.2)

                # Scale anisotropy regularization — the primary cause of shard/needle
                # artifacts in iOS room scans.  Without this, 50%+ of Gaussians can
                # become razor-thin discs (aspect ratio >10,000) after densification
                # stops, since nothing prevents the optimizer from flattening them.
                #
                # Penalty: mean difference between max and min log-scale per Gaussian.
                # A perfect sphere has diff=0; a razor disc has a very large diff.
                # Weight 0.01 keeps it well below the reconstruction loss while still
                # strongly discouraging degenerate shapes.
                scales = strategy_params["scales"]
                scale_aniso_reg = (
                    scales.max(dim=1).values - scales.min(dim=1).values
                ).mean()
                loss = loss + 0.01 * scale_aniso_reg

                # Backward
                for opt in optimizers.values():
                    opt.zero_grad()
                loss.backward()

                # NOTE: No gradient clipping -- the reference 3DGS never clips
                # gradients.  Clipping suppresses the viewspace position gradient
                # signal that the densification strategy relies on for clone/split
                # decisions.

                for opt in optimizers.values():
                    opt.step()

                # Post-backward step for densification
                strategy.step_post_backward(
                    params=strategy_params,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                    packed=False,
                )

                # Update params if densification changed them
                params = self._update_params_from_dict(params, strategy_params)

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
                    if output_format in ("ply", "both"):
                        checkpoint_path = output_dir / f"point_cloud_{step}.ply"
                        if self._save_ply(checkpoint_path, params):
                            logger.info(f"Saved PLY checkpoint: {checkpoint_path}")
                        else:
                            logger.warning(f"PLY checkpoint save failed: {checkpoint_path}")
                    if output_format in ("glb", "both"):
                        glb_checkpoint_path = output_dir / f"point_cloud_{step}.glb"
                        self._save_glb(glb_checkpoint_path, params)
                        logger.info(f"Saved GLB checkpoint: {glb_checkpoint_path}")

                # Yield to event loop periodically
                if step % 10 == 0:
                    await asyncio.sleep(0)

            # Save final result
            self._log_system_status()
            logger.info(
                f"Final positions: min={params.means.min().item():.4f}, "
                f"max={params.means.max().item():.4f}"
            )
            logger.info(
                f"Final scales: min={params.scales.min().item():.4f}, "
                f"max={params.scales.max().item():.4f}"
            )

            save_success = False

            # Save PLY
            if output_format in ("ply", "both"):
                final_ply_path = output_dir / "point_cloud.ply"
                if self._save_ply(final_ply_path, params):
                    logger.info(f"Saved final PLY: {final_ply_path}")
                    result["output_path"] = str(final_ply_path)
                    save_success = True
                else:
                    logger.error("Failed to save final PLY - validation failed")

            # Save glTF
            if output_format in ("glb", "both"):
                final_glb_path = output_dir / "point_cloud.glb"
                try:
                    self._save_glb(final_glb_path, params)
                    logger.info(f"Saved final GLB: {final_glb_path}")
                    result["gltf_output_path"] = str(final_glb_path)
                    # If PLY wasn't saved or failed, use GLB as primary output
                    if result.get("output_path") is None:
                        result["output_path"] = str(final_glb_path)
                    save_success = True
                except Exception as e:
                    logger.exception(f"Failed to save final GLB: {e}")

            if save_success:
                result["success"] = True
            else:
                result["success"] = False
                result["error"] = "Failed to save output files"

        except Exception as e:
            logger.exception(f"Training error: {e}")
            result["error"] = str(e)

        return result

    @staticmethod
    def _knn_scale_init(points: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Compute per-point initial scale from k-nearest-neighbor distances.

        Matches the reference 3DGS implementation: each Gaussian's initial
        scale is set to the mean distance to its k nearest neighbors,
        giving data-driven sizes proportional to local point density.
        """
        from torch import cdist

        # For very large point clouds, compute in chunks to avoid OOM
        n = points.shape[0]
        if n <= 50000:
            dists = cdist(points, points)  # (N, N)
            # Set self-distance to inf so it's not selected as a neighbor
            dists.fill_diagonal_(float("inf"))
            knn_dists, _ = dists.topk(k, dim=1, largest=False)  # (N, k)
            mean_dist = knn_dists.mean(dim=1)  # (N,)
        else:
            # Chunked computation for large point clouds
            chunk_size = 4096
            mean_dists = []
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk = points[i:end]  # (C, 3)
                d = cdist(chunk, points)  # (C, N)
                # Zero out self-distances
                for j in range(end - i):
                    d[j, i + j] = float("inf")
                knn_d, _ = d.topk(k, dim=1, largest=False)
                mean_dists.append(knn_d.mean(dim=1))
            mean_dist = torch.cat(mean_dists)

        # Clamp to avoid degenerate zero-size Gaussians
        mean_dist = torch.clamp_min(mean_dist, 1e-7)
        # log(sqrt(dist^2)) = log(dist); reference uses log(sqrt(dist2))
        log_scale = torch.log(mean_dist)
        return log_scale

    def _init_gaussians(
        self,
        points: torch.Tensor,
        sh_degree: int = 3,
        init_colors: Optional[torch.Tensor] = None,
        scene_scale: float = 1.0,
    ) -> GaussianParams:
        """Initialize Gaussian parameters from point cloud."""
        num_points = len(points)

        # Positions
        means = points.clone().requires_grad_(True)

        # Scales (log-space) - use KNN-based per-point adaptive sizing
        # matching the reference 3DGS implementation (distCUDA2).
        if num_points >= 4:
            log_scale = self._knn_scale_init(points, k=min(3, num_points - 1))
            scales = log_scale.unsqueeze(1).expand(-1, 3).clone()
            logger.info(
                "KNN-based init scales: min=%.4f, max=%.4f, median=%.4f (log-space)",
                scales[:, 0].min().item(),
                scales[:, 0].max().item(),
                scales[:, 0].median().item(),
            )
        else:
            base_scale = max(scene_scale * 0.01, 1e-4)
            scales = torch.full((num_points, 3), math.log(base_scale), device=self.device)
        scales = scales.requires_grad_(True)

        # Rotations (quaternions, initialized to identity)
        quats = torch.zeros(num_points, 4, device=self.device)
        quats[:, 0] = 1.0  # w=1, x=y=z=0
        quats = quats.requires_grad_(True)

        # Opacities (logit-space, initialized to ~0.1)
        opacities = torch.zeros(num_points, device=self.device)
        opacities.fill_(-2.2)  # sigmoid(-2.2) ~= 0.1
        opacities = opacities.requires_grad_(True)

        # SH degree-0 coefficients (RGB channels).
        # RGB2SH: coeff = (rgb - 0.5) / C0
        C0 = 0.28209479177387814
        sh0 = torch.zeros(num_points, 3, device=self.device)
        if init_colors is not None and len(init_colors) == num_points:
            init_colors = torch.clamp(init_colors.to(self.device), 0.0, 1.0)
            sh0 = (init_colors - 0.5) / C0
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

    def _create_optimizers(
        self,
        params: GaussianParams,
        scene_scale: float = 1.0,
        max_steps: int = 30000,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Create one Adam optimizer per trainable tensor for gsplat strategy.

        Matches reference 3DGS learning rates and Adam eps=1e-15.
        Position LR is scaled by scene spatial extent.
        """
        spatial_lr_scale = scene_scale
        position_lr_init = 0.00016 * spatial_lr_scale

        optimizers: Dict[str, torch.optim.Optimizer] = {
            "means": torch.optim.Adam([params.means], lr=position_lr_init, eps=1e-15),
            "scales": torch.optim.Adam([params.scales], lr=0.005, eps=1e-15),
            "quats": torch.optim.Adam([params.quats], lr=0.001, eps=1e-15),
            "opacities": torch.optim.Adam([params.opacities], lr=0.025, eps=1e-15),
            "sh0": torch.optim.Adam([params.sh0], lr=0.0025, eps=1e-15),
        }

        if params.sh_rest is not None:
            optimizers["sh_rest"] = torch.optim.Adam(
                [params.sh_rest],
                lr=0.0025 / 20,
                eps=1e-15,
            )

        # Build exponential LR decay for position (matches reference exactly)
        self._position_lr_fn = _get_expon_lr_func(
            lr_init=position_lr_init,
            lr_final=0.0000016 * spatial_lr_scale,
            lr_delay_mult=0.01,
            max_steps=max_steps,
        )

        logger.info(
            "Optimizer LRs: position=%.6f*spatial(%.3f), opacity=0.025, "
            "scaling=0.005, rotation=0.001, sh0=0.0025, eps=1e-15",
            0.00016, spatial_lr_scale,
        )

        return optimizers

    def _update_learning_rate(
        self, optimizers: Dict[str, torch.optim.Optimizer], step: int
    ) -> None:
        """Update position learning rate using exponential decay schedule."""
        lr = self._position_lr_fn(step)
        for param_group in optimizers["means"].param_groups:
            param_group["lr"] = lr

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

    def _build_sh_coeffs(
        self,
        params: GaussianParams,
        max_sh_degree: int,
        step: int,
    ) -> tuple[torch.Tensor, int]:
        """Build SH coefficient tensor and active SH degree for gsplat."""
        active_sh_degree = min(max_sh_degree, step // 1000)
        if params.sh_rest is None:
            return params.sh0.unsqueeze(1), 0
        coeffs = torch.cat([params.sh0.unsqueeze(1), params.sh_rest], dim=1)
        return coeffs, active_sh_degree

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
        quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)

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
            # 3DGS PLY stores SH rest in channel-major order:
            # [R_all_coeffs..., G_all_coeffs..., B_all_coeffs...]
            for c in range(3):
                for i in range(num_sh):
                    dtype.append((f"f_rest_{c * num_sh + i}", "f4"))

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

        # SH degree-0 coefficients
        elements["f_dc_0"] = sh0[:, 0]
        elements["f_dc_1"] = sh0[:, 1]
        elements["f_dc_2"] = sh0[:, 2]

        # Higher-order SH
        if params.sh_rest is not None:
            sh_rest = params.sh_rest.detach().cpu().numpy()
            num_sh = sh_rest.shape[1]
            for c in range(3):
                for i in range(num_sh):
                    elements[f"f_rest_{c * num_sh + i}"] = sh_rest[:, i, c]

        # PLY stores opacity in inverse-sigmoid (logit) space.
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

    def _save_glb(self, path: Path, params: GaussianParams) -> None:
        """Save Gaussians as GLB with KHR_gaussian_splatting extension."""
        from services.gltf_export import save_gltf

        save_gltf(
            path=path,
            means=params.means,
            scales=params.scales,
            quats=params.quats,
            opacities=params.opacities,
            sh0=params.sh0,
            sh_rest=params.sh_rest,
        )

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
