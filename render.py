"""
Render images from a trained 3D Gaussian Splatting model.

Loads a trained PLY file and renders:
  1) All training camera views (comparison with ground truth)
  2) An orbit video around the scene (novel views)

Usage:
    python render.py --ply I:/Deco/truck_output_fixed2/point_cloud.ply --data I:/Deco/truck --output I:/Deco/truck_renders
    python render.py --ply I:/Deco/truck_output_fixed2/point_cloud.ply --data I:/Deco/truck --output I:/Deco/truck_renders --orbit-only --orbit-frames 120
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def load_gaussians_from_ply(ply_path: Path, device: torch.device):
    """Load trained Gaussian parameters from a 3DGS-format PLY file."""
    from plyfile import PlyData

    logger.info(f"Loading Gaussians from {ply_path}")
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n = len(vertex)
    logger.info(f"Loaded {n:,} Gaussians")

    # Positions
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    # SH degree-0
    sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)

    # Count higher-order SH coefficients
    sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
    num_sh_rest = len(sh_rest_names)
    sh_rest = None
    if num_sh_rest > 0:
        # SH rest is stored channel-major: R_0..R_k, G_0..G_k, B_0..B_k
        num_coeffs_per_channel = num_sh_rest // 3
        sh_rest_flat = np.zeros((n, num_sh_rest), dtype=np.float32)
        for i, name in enumerate(sh_rest_names):
            sh_rest_flat[:, i] = np.array(vertex[name], dtype=np.float32)
        # Reshape from channel-major to (N, K, 3)
        sh_rest = np.zeros((n, num_coeffs_per_channel, 3), dtype=np.float32)
        for c in range(3):
            for k in range(num_coeffs_per_channel):
                sh_rest[:, k, c] = sh_rest_flat[:, c * num_coeffs_per_channel + k]

    # Opacity (logit-space)
    opacities = np.array(vertex["opacity"], dtype=np.float32)

    # Scales (log-space)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)

    # Rotations (quaternion wxyz)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)

    # Determine SH degree from coefficient count
    if sh_rest is not None:
        total_sh = 1 + sh_rest.shape[1]  # +1 for degree 0
        sh_degree = int(math.sqrt(total_sh)) - 1
    else:
        sh_degree = 0

    logger.info(f"SH degree: {sh_degree} ({1 + (num_sh_rest // 3) if num_sh_rest else 1} coefficients)")

    return {
        "means": torch.tensor(means, device=device),
        "sh0": torch.tensor(sh0, device=device),
        "sh_rest": torch.tensor(sh_rest, device=device) if sh_rest is not None else None,
        "opacities": torch.tensor(opacities, device=device),
        "scales": torch.tensor(scales, device=device),
        "quats": torch.tensor(quats, device=device),
        "sh_degree": sh_degree,
    }


def render_view(gaussians, viewmat, K, width, height):
    """Render a single view using gsplat rasterization."""
    from gsplat.rendering import rasterization

    quats = gaussians["quats"]
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Build SH coefficients
    if gaussians["sh_rest"] is not None:
        sh_coeffs = torch.cat([gaussians["sh0"].unsqueeze(1), gaussians["sh_rest"]], dim=1)
    else:
        sh_coeffs = gaussians["sh0"].unsqueeze(1)

    renders, alphas, info = rasterization(
        means=gaussians["means"],
        quats=quats,
        scales=torch.exp(gaussians["scales"]),
        opacities=torch.sigmoid(gaussians["opacities"]),
        colors=sh_coeffs,
        sh_degree=gaussians["sh_degree"],
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        packed=False,
        render_mode="RGB",
        absgrad=False,
    )

    # (1, H, W, 3) -> (H, W, 3), clamped to valid range
    image = renders[0].clamp(0, 1)
    return image


def generate_orbit_cameras(center, radius, elevation_deg, num_frames, up=None):
    """Generate camera-to-world matrices for an orbit around a center point."""
    if up is None:
        up = np.array([0, -1, 0], dtype=np.float32)  # Y-down (COLMAP convention)

    viewmats = []
    elevation = np.radians(elevation_deg)

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames

        # Camera position on orbit
        cam_x = center[0] + radius * np.cos(angle) * np.cos(elevation)
        cam_y = center[1] - radius * np.sin(elevation)
        cam_z = center[2] + radius * np.sin(angle) * np.cos(elevation)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Look-at matrix
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            up_alt = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(forward, up_alt)
            right_norm = np.linalg.norm(right)
        right = right / right_norm

        actual_up = np.cross(right, forward)

        # World-to-camera
        R = np.stack([right, -actual_up, forward], axis=0)  # (3, 3)
        t = -R @ cam_pos

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        viewmats.append(w2c)

    return viewmats


def save_image(tensor, path):
    """Save a (H, W, 3) float tensor as an image."""
    img_np = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(str(path))


def make_comparison(rendered, ground_truth):
    """Create a side-by-side comparison image."""
    return np.concatenate([rendered, ground_truth], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Render views from trained 3DGS model")
    parser.add_argument("--ply", type=Path, required=True, help="Path to trained PLY file")
    parser.add_argument("--data", type=Path, required=True, help="Path to training data (for cameras)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for renders")
    parser.add_argument("--image-scale", type=float, default=0.5, help="Image scale (default: 0.5)")
    parser.add_argument("--orbit-frames", type=int, default=60, help="Number of orbit frames (default: 60)")
    parser.add_argument("--orbit-elevation", type=float, default=15, help="Orbit elevation in degrees")
    parser.add_argument("--orbit-only", action="store_true", help="Skip training view renders")
    parser.add_argument("--no-orbit", action="store_true", help="Skip orbit video")
    parser.add_argument("--max-train-views", type=int, default=0, help="Max training views to render (0=all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Gaussians
    gaussians = load_gaussians_from_ply(args.ply, device)

    # Load camera data
    os.environ["SPLAT_IMAGE_SCALE"] = str(args.image_scale)
    from services.camera_utils import load_training_data
    data = load_training_data(args.data, device=device, image_scale=args.image_scale)

    viewmats = data["viewmats"]
    Ks = data["Ks"]
    images = data["images"]
    width = data["width"]
    height = data["height"]
    scene_center = data["scene_center"].cpu().numpy()
    scene_scale = data["scene_scale"]

    args.output.mkdir(parents=True, exist_ok=True)

    # --- Render training views ---
    if not args.orbit_only:
        train_dir = args.output / "training_views"
        train_dir.mkdir(exist_ok=True)
        compare_dir = args.output / "comparisons"
        compare_dir.mkdir(exist_ok=True)

        num_views = len(viewmats)
        if args.max_train_views > 0:
            num_views = min(num_views, args.max_train_views)

        logger.info(f"Rendering {num_views} training views...")

        psnrs = []
        for i in range(num_views):
            with torch.no_grad():
                rendered = render_view(gaussians, viewmats[i], Ks[i], width, height)

            # Save rendered image
            save_image(rendered, train_dir / f"render_{i:04d}.png")

            # Save comparison (rendered | ground truth)
            gt = images[i]
            rendered_np = (rendered.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gt.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            comparison = make_comparison(rendered_np, gt_np)
            Image.fromarray(comparison).save(str(compare_dir / f"compare_{i:04d}.png"))

            # PSNR
            mse = ((rendered - gt) ** 2).mean().item()
            psnr = -10 * math.log10(mse) if mse > 0 else float("inf")
            psnrs.append(psnr)

            if (i + 1) % 25 == 0 or i == 0:
                logger.info(f"  View {i+1}/{num_views}: PSNR={psnr:.2f} dB")

        avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0
        logger.info(f"Average PSNR across {len(psnrs)} views: {avg_psnr:.2f} dB")

    # --- Render orbit video ---
    if not args.no_orbit:
        orbit_dir = args.output / "orbit"
        orbit_dir.mkdir(exist_ok=True)

        # Estimate orbit radius from camera positions
        cam_centers = []
        for i in range(len(viewmats)):
            w2c = viewmats[i].cpu().numpy()
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            cam_centers.append(-(R.T @ t))
        cam_centers = np.array(cam_centers)
        orbit_center = cam_centers.mean(axis=0)
        orbit_radius = np.linalg.norm(cam_centers - orbit_center, axis=1).mean()

        logger.info(f"Orbit center: {orbit_center}, radius: {orbit_radius:.2f}")
        logger.info(f"Rendering {args.orbit_frames} orbit frames...")

        # Use the first camera's intrinsics for orbit renders
        K_orbit = Ks[0]

        orbit_viewmats = generate_orbit_cameras(
            center=orbit_center,
            radius=orbit_radius,
            elevation_deg=args.orbit_elevation,
            num_frames=args.orbit_frames,
        )

        for i, w2c in enumerate(orbit_viewmats):
            w2c_tensor = torch.tensor(w2c, device=device, dtype=torch.float32)
            with torch.no_grad():
                rendered = render_view(gaussians, w2c_tensor, K_orbit, width, height)
            save_image(rendered, orbit_dir / f"orbit_{i:04d}.png")

            if (i + 1) % 20 == 0:
                logger.info(f"  Orbit frame {i+1}/{args.orbit_frames}")

        logger.info(f"Orbit frames saved to {orbit_dir}")

        # Try to create video with ffmpeg if available
        try:
            import subprocess
            video_path = args.output / "orbit_video.mp4"
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", str(orbit_dir / "orbit_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(video_path),
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Orbit video saved: {video_path}")
            else:
                logger.warning(f"ffmpeg failed (video not created, frames still available): {result.stderr[:200]}")
        except FileNotFoundError:
            logger.info("ffmpeg not found - orbit frames saved as PNGs (install ffmpeg to create video)")

    logger.info("Rendering complete!")
    logger.info(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
