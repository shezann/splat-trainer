"""
Camera and image loading utilities for 3DGS training.
Handles NeRF-style transforms.json format from iOS capture.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_training_data(
    data_dir: Path,
    device: torch.device = torch.device("cuda"),
    image_scale: float = 1.0,
) -> dict:
    """
    Load cameras and images from transforms.json (NeRF format).

    Args:
        data_dir: Directory containing transforms.json and images
        device: Torch device for tensors
        image_scale: Scale factor for images (0.5 = half resolution)

    Returns:
        dict with:
            - viewmats: (N, 4, 4) world-to-camera matrices
            - Ks: (N, 3, 3) camera intrinsic matrices
            - images: (N, H, W, 3) training images as float32 [0, 1]
            - initial_points: (M, 3) initial 3D points for Gaussians
            - width: image width
            - height: image height
    """
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found in {data_dir}")

    with open(transforms_path) as f:
        transforms = json.load(f)

    # Parse camera intrinsics
    # NeRF format uses camera_angle_x for horizontal FOV
    camera_angle_x = transforms.get("camera_angle_x", 0.8)
    camera_angle_y = transforms.get("camera_angle_y")

    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError("No frames found in transforms.json")

    # Load first image to get dimensions
    first_frame = frames[0]
    first_image_path = _resolve_image_path(data_dir, first_frame["file_path"])
    if first_image_path is None:
        raise FileNotFoundError(f"Could not find image: {first_frame['file_path']}")

    with Image.open(first_image_path) as img:
        orig_width, orig_height = img.size

    # Apply scale
    width = int(orig_width * image_scale)
    height = int(orig_height * image_scale)

    # Compute focal lengths from FOV
    fx = width / (2 * np.tan(camera_angle_x / 2))
    if camera_angle_y is not None:
        fy = height / (2 * np.tan(camera_angle_y / 2))
    else:
        fy = fx  # Assume square pixels

    # Check for per-frame intrinsics (some iOS exports include these)
    fl_x = transforms.get("fl_x")
    fl_y = transforms.get("fl_y")
    cx = transforms.get("cx", width / 2)
    cy = transforms.get("cy", height / 2)

    if fl_x is not None:
        fx = fl_x * image_scale
    if fl_y is not None:
        fy = fl_y * image_scale
    cx = cx * image_scale
    cy = cy * image_scale

    logger.info(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Load all frames
    viewmats = []
    Ks = []
    images = []
    camera_positions = []

    for frame in frames:
        image_path = _resolve_image_path(data_dir, frame["file_path"])
        if image_path is None:
            logger.warning(f"Skipping missing image: {frame['file_path']}")
            continue

        # Load and process image
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if image_scale != 1.0:
                img = img.resize((width, height), Image.LANCZOS)
            image_np = np.array(img, dtype=np.float32) / 255.0

        images.append(image_np)

        # Parse transform matrix (camera-to-world, c2w)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        # Convert to world-to-camera (w2c) for gsplat
        # gsplat expects w2c matrices
        w2c = np.linalg.inv(c2w)
        viewmats.append(w2c)

        # Store camera position for point cloud initialization
        camera_positions.append(c2w[:3, 3])

        # Per-frame intrinsics (if available)
        frame_fx = frame.get("fl_x", fx)
        frame_fy = frame.get("fl_y", fy)
        frame_cx = frame.get("cx", cx)
        frame_cy = frame.get("cy", cy)

        if frame.get("fl_x") is not None:
            frame_fx *= image_scale
            frame_fy = frame.get("fl_y", frame_fx) * image_scale
            frame_cx = frame.get("cx", width / 2) * image_scale
            frame_cy = frame.get("cy", height / 2) * image_scale

        K = np.array([
            [frame_fx, 0, frame_cx],
            [0, frame_fy, frame_cy],
            [0, 0, 1],
        ], dtype=np.float32)
        Ks.append(K)

    if len(images) == 0:
        raise ValueError("No valid images found")

    logger.info(f"Loaded {len(images)} images at {width}x{height}")

    # Convert to tensors
    viewmats = torch.tensor(np.stack(viewmats), device=device, dtype=torch.float32)
    Ks = torch.tensor(np.stack(Ks), device=device, dtype=torch.float32)
    images = torch.tensor(np.stack(images), device=device, dtype=torch.float32)

    # Initialize points from camera frustums or mesh
    initial_points = _initialize_points_from_cameras(
        camera_positions, data_dir, device
    )

    return {
        "viewmats": viewmats,
        "Ks": Ks,
        "images": images,
        "initial_points": initial_points,
        "width": width,
        "height": height,
    }


def _resolve_image_path(data_dir: Path, file_path: str) -> Optional[Path]:
    """Resolve image path, handling various formats."""
    # Try direct path
    path = data_dir / file_path
    if path.exists():
        return path

    # Try with common extensions
    base_path = data_dir / file_path
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        test_path = base_path.with_suffix(ext)
        if test_path.exists():
            return test_path

    # Try in images subdirectory
    images_dir = data_dir / "images"
    if images_dir.exists():
        name = Path(file_path).name
        for ext in ["", ".jpg", ".jpeg", ".png"]:
            test_path = images_dir / (name + ext)
            if test_path.exists():
                return test_path

    return None


def _initialize_points_from_cameras(
    camera_positions: list,
    data_dir: Path,
    device: torch.device,
    num_random_points: int = 10000,
) -> torch.Tensor:
    """
    Initialize 3D points for Gaussian splatting.

    Tries to load from:
    1. Existing point cloud (points3D.ply, sparse.ply)
    2. Mesh file (mesh.ply, mesh.glb)
    3. Falls back to random points in camera frustum

    Args:
        camera_positions: List of camera positions
        data_dir: Data directory to search for existing geometry
        device: Torch device
        num_random_points: Number of random points if no geometry found

    Returns:
        (N, 3) tensor of initial 3D points
    """
    # Try to load existing point cloud
    point_cloud_files = [
        "points3D.ply",
        "sparse.ply",
        "point_cloud.ply",
        "sfm_points.ply",
    ]

    for filename in point_cloud_files:
        ply_path = data_dir / filename
        if ply_path.exists():
            points = _load_ply_points(ply_path)
            if points is not None and len(points) > 100:
                logger.info(f"Loaded {len(points)} points from {filename}")
                return torch.tensor(points, device=device, dtype=torch.float32)

    # Try to sample points from mesh
    mesh_files = ["mesh.ply", "mesh.glb", "mesh.obj"]
    for filename in mesh_files:
        mesh_path = data_dir / filename
        if mesh_path.exists():
            points = _sample_mesh_points(mesh_path, num_random_points)
            if points is not None:
                logger.info(f"Sampled {len(points)} points from {filename}")
                return torch.tensor(points, device=device, dtype=torch.float32)

    # Fall back to random initialization based on camera positions
    logger.info("No geometry found, initializing random points from camera positions")
    camera_positions = np.array(camera_positions)
    center = camera_positions.mean(axis=0)
    extent = np.abs(camera_positions - center).max() * 1.5

    # Random points in scene bounds
    points = np.random.uniform(
        center - extent,
        center + extent,
        (num_random_points, 3)
    ).astype(np.float32)

    return torch.tensor(points, device=device, dtype=torch.float32)


def _load_ply_points(ply_path: Path) -> Optional[np.ndarray]:
    """Load points from a PLY file."""
    try:
        from plyfile import PlyData

        plydata = PlyData.read(str(ply_path))
        vertex = plydata["vertex"]

        x = np.array(vertex["x"])
        y = np.array(vertex["y"])
        z = np.array(vertex["z"])

        return np.stack([x, y, z], axis=1).astype(np.float32)

    except Exception as e:
        logger.warning(f"Failed to load PLY {ply_path}: {e}")
        return None


def _sample_mesh_points(mesh_path: Path, num_points: int) -> Optional[np.ndarray]:
    """Sample points from a mesh surface."""
    try:
        import trimesh

        logger.info(f"Loading mesh from {mesh_path}")
        mesh = trimesh.load(str(mesh_path), force='mesh')

        # Handle Scene objects (GLB files often load as Scene)
        if isinstance(mesh, trimesh.Scene):
            logger.info(f"GLB loaded as Scene with {len(mesh.geometry)} geometries")
            if len(mesh.geometry) == 0:
                logger.warning("Scene has no geometry")
                return None
            # Combine all meshes in the scene
            meshes = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(meshes)
            logger.info(f"Combined into mesh with {len(mesh.vertices)} vertices")

        if hasattr(mesh, "sample") and hasattr(mesh, "area") and mesh.area > 0:
            points = mesh.sample(num_points)
            logger.info(f"Sampled {len(points)} points from mesh surface")
            return np.array(points, dtype=np.float32)
        elif hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            # Just use vertices if can't sample
            vertices = np.array(mesh.vertices, dtype=np.float32)
            logger.info(f"Using {len(vertices)} mesh vertices directly")
            if len(vertices) > num_points:
                indices = np.random.choice(len(vertices), num_points, replace=False)
                return vertices[indices]
            return vertices
        else:
            logger.warning(f"Mesh has no usable geometry: {type(mesh)}")

    except Exception as e:
        logger.exception(f"Failed to load mesh {mesh_path}: {e}")

    return None
