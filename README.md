# Splat Trainer

A 3D Gaussian Splatting training server implementing the method described in
["3D Gaussian Splatting for Real-Time Radiance Field Rendering"](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
(Kerbl et al., SIGGRAPH 2023). Built on [gsplat](https://github.com/nerfstudio-project/gsplat)
for differentiable rasterization and exposed as both a CLI tool and a FastAPI service.

## Paper Implementation Details

The training pipeline in `services/training.py` follows the original 3DGS paper
as closely as possible. Below is a section-by-section mapping from paper to code.

### Initialization (Paper Section 4)

The paper initializes Gaussians from a Structure-from-Motion (SfM) point cloud
produced by COLMAP. Each point becomes a Gaussian with:

| Parameter | Paper | Code (`_init_gaussians`) |
|-----------|-------|--------------------------|
| Position | SfM point xyz | `means = points.clone()` |
| Scale | Mean distance to 3 nearest neighbors (isotropic) | `_knn_scale_init(points, k=3)` stored in log-space |
| Rotation | Identity quaternion | `quats[:, 0] = 1.0` (wxyz) |
| Opacity | Low initial value (~0.1) | `sigmoid(-2.2) = 0.1` in logit-space |
| Color (SH DC) | SfM point color converted to SH | `(rgb - 0.5) / C0` where `C0 = 0.28209` |
| Color (SH rest) | Zeros for higher-order bands | `sh_rest = zeros(N, (deg+1)^2-1, 3)` |

The KNN-based scale initialization (`_knn_scale_init`) matches the reference
`distCUDA2` kernel: it computes per-point mean distance to 3 nearest neighbors,
giving data-driven Gaussian sizes proportional to local point density.
For large point clouds (>50K), distances are computed in chunks to avoid OOM.

### Loss Function (Paper Section 5)

The paper uses a combined L1 + D-SSIM loss:

```
L = (1 - lambda) * L1 + lambda * L_D-SSIM
```

with `lambda = 0.2`. Implemented in `services/loss_utils.py`:

- **`l1_loss`**: Mean absolute error between predicted and ground-truth images.
- **`compute_ssim`**: Structural Similarity Index using an 11x11 Gaussian window
  (`sigma=1.5`), computed via depthwise `conv2d` on (B,C,H,W) tensors.
- **`combined_loss`**: Combines them as `0.8 * L1 + 0.2 * (1 - SSIM)`.

The rendered output is clamped to [0, 1] before loss computation
(`renders[0].clamp(0, 1)`) to prevent spurious gradients from out-of-range values,
matching the reference implementation.

### Optimization (Paper Section 5)

The paper uses Adam with specific per-parameter learning rates and schedules.
Each parameter group has its own Adam optimizer (required by gsplat's
strategy API for densification to inject/remove parameters).

| Parameter | Learning Rate | Notes |
|-----------|--------------|-------|
| Position (means) | `0.00016 * spatial_lr_scale` | Exponential decay to `0.0000016 * spatial_lr_scale` |
| Scale | 0.005 | Constant |
| Rotation | 0.001 | Constant |
| Opacity | 0.025 | Constant |
| SH DC (sh0) | 0.0025 | Constant |
| SH rest | 0.000125 (0.0025/20) | Constant |

Key optimizer details matching the paper:

- **Adam epsilon = 1e-15** (not the PyTorch default of 1e-8). This ensures very
  small gradients still produce meaningful parameter updates, which is critical
  for the fine-grained splat adjustments that 3DGS relies on.
- **Position LR schedule** (`_get_expon_lr_func`): Log-linear interpolation from
  `lr_init` to `lr_final` over the full training run, with an optional cosine
  warmup ramp. Adopted from Plenoxels/JaxNeRF.
- **`spatial_lr_scale`**: The position LR is multiplied by the spatial extent of
  the scene (radius of the camera orbit). This normalizes the step size so that
  the optimizer moves Gaussians by a consistent fraction of scene extent
  regardless of absolute scale.
- **No gradient clipping**: The reference 3DGS never clips gradients. Clipping
  would suppress the viewspace position gradient signal that the densification
  strategy uses for clone/split decisions.

### Spherical Harmonics (Paper Section 4)

Color is represented using spherical harmonics up to degree 3 (16 coefficients
per color channel). The SH degree is increased progressively during training:

```python
active_sh_degree = min(max_sh_degree, step // 1000)
```

This starts with degree 0 (constant color) and unlocks one degree every 1000
steps (degree 1 at step 1000, degree 2 at 2000, degree 3 at 3000), allowing the
model to first learn coarse geometry before fitting view-dependent appearance.

SH coefficients are packed into a `(N, (deg+1)^2, 3)` tensor and passed to
gsplat's rasterizer which evaluates them per-Gaussian per-pixel.

### Adaptive Density Control (Paper Section 5.2)

The paper's densification strategy (clone small under-reconstructed Gaussians,
split large over-reconstructed ones, prune transparent/oversized ones) is
handled by gsplat's `DefaultStrategy`:

| Paper Concept | gsplat Parameter | Value |
|---------------|------------------|-------|
| Densify start iteration | `refine_start_iter` | 500 |
| Densify stop iteration | `refine_stop_iter` | 15000 |
| Densify every N steps | `refine_every` | 100 |
| Opacity reset interval | `reset_every` | 3000 |
| Min opacity threshold (prune) | `prune_opa` | 0.005 |
| Gradient threshold (clone/split) | `grow_grad2d` | 0.0002 |
| Scale threshold (split vs clone) | `grow_scale3d` | 0.01 |
| Max scale threshold (prune) | `prune_scale3d` | 0.1 |

The strategy requires a `step_pre_backward` call (after rasterization, before
`loss.backward()`) to capture 2D gradient statistics, and a `step_post_backward`
call to execute the actual densification operations.

### Camera Sampling (Paper Section 5)

The paper randomly samples training views at each step. This implementation uses
**epoch-based sampling**: all cameras are shuffled into a pool, drawn without
replacement until exhausted, then reshuffled. This ensures every training view is
seen exactly once per epoch, which provides more uniform coverage than pure
random sampling with replacement.

### Camera Model and Intrinsics

Camera data is loaded from COLMAP binary files (`cameras.bin`, `images.bin`,
`points3D.bin`) via `services/colmap_loader.py`. The loader handles a critical
subtlety: COLMAP stores intrinsics at the original capture resolution, but images
on disk may already be downscaled. The code computes a `disk_scale` ratio between
the actual image dimensions and COLMAP's recorded resolution, then multiplies by
the user's `image_scale` to get the correct total intrinsics scale factor:

```python
disk_scale = orig_width / colmap_width
intrinsics_scale = disk_scale * image_scale
fx, fy, cx, cy = get_intrinsics(camera, scale=intrinsics_scale)
```

This prevents focal-length mismatch artifacts (blurry renders, incorrect
perspective) when training images differ in resolution from the COLMAP
reconstruction.

NeRF-format datasets (`transforms.json`) are also supported, with FOV-based
intrinsics and camera-to-world matrix inversion.

### Output Formats

- **PLY**: Standard 3DGS point cloud format with all Gaussian attributes
  (position, scale, rotation, opacity, SH coefficients). Compatible with most
  3DGS viewers. SH rest coefficients are stored in channel-major order
  `[R_0..R_n, G_0..G_n, B_0..B_n]`.
- **GLB**: glTF binary with the `KHR_gaussian_splatting` extension
  (`services/gltf_export.py`). Handles quaternion WXYZ-to-XYZW conversion and
  opacity logit-to-sigmoid conversion for the glTF spec.

## Project Structure

```
splat-trainer/
  train.py              # CLI training driver
  main.py               # FastAPI server
  config.py             # Quality presets, checkpoint iterations
  render.py             # Render images from trained model (GT comparisons + orbit)
  viewer.py             # Interactive real-time viewer (pygame + gsplat)
  services/
    training.py         # Core training loop (GaussianSplatTrainer)
    camera_utils.py     # COLMAP / NeRF data loading, intrinsics scaling
    colmap_loader.py    # COLMAP binary file readers
    loss_utils.py       # L1 + SSIM loss, PSNR computation
    gltf_export.py      # GLB export with KHR_gaussian_splatting
    job_manager.py      # Async job queue for the API server
    storage.py          # File storage utilities
  scripts/
    reconstruct_mesh.py # Traditional mesh reconstruction (PatchMatch + Poisson)
```

## Usage

### Training (CLI)

```bash
python train.py --data /path/to/dataset --output /path/to/output
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Path to COLMAP or NeRF dataset |
| `--output` | (required) | Output directory for checkpoints and final model |
| `--iterations` | from preset | Number of training iterations |
| `--quality` | `balanced` | Preset: `fast` (10K iter), `balanced` (30K), `high` (50K) |
| `--image-scale` | `0.5` | Image resolution scale (0.5 = half res, 1.0 = full res) |
| `--format` | `both` | Output format: `ply`, `glb`, or `both` |

### Training (API Server)

```bash
python main.py
# POST /api/jobs with dataset upload
```

### Rendering

Render training views with ground-truth comparisons and orbit fly-around:

```bash
python render.py --ply output/point_cloud.ply --data /path/to/dataset --output renders/
```

### Interactive Viewer

Real-time 3D viewer with mouse orbit, pan, and zoom:

```bash
python viewer.py --ply output/point_cloud.ply --data /path/to/dataset
```

Controls: left-click drag = orbit, right-click drag = pan, scroll = zoom,
R = reset, Q/ESC = quit.

## Dependencies

- **PyTorch** (with CUDA)
- **gsplat** - differentiable Gaussian rasterization (requires ninja + MSVC on Windows for JIT CUDA compilation)
- **plyfile** - PLY file I/O
- **Pillow** - image loading
- **pygame** - interactive viewer
- **numpy**, **scipy**
