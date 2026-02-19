"""
Camera and image loading utilities for 3DGS training.
Handles NeRF-style transforms.json format and COLMAP sparse reconstructions.
"""

import json
import logging
import os
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
    Load cameras and images, auto-detecting format (COLMAP or NeRF).

    Args:
        data_dir: Directory containing training data
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
    # Check for COLMAP format first
    if (data_dir / "sparse" / "0" / "cameras.bin").exists():
        logger.info("Detected COLMAP format")
        return load_colmap_data(data_dir, device, image_scale)

    # Fall back to NeRF transforms.json
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"No training data found in {data_dir}. "
            f"Expected COLMAP (sparse/0/cameras.bin) or NeRF (transforms.json)."
        )

    logger.info("Detected NeRF format")
    return _load_nerf_data(data_dir, device, image_scale)


def load_colmap_data(
    data_dir: Path,
    device: torch.device = torch.device("cuda"),
    image_scale: float = 1.0,
) -> dict:
    """
    Load cameras and images from COLMAP sparse reconstruction.

    Expects:
        data_dir/sparse/0/cameras.bin
        data_dir/sparse/0/images.bin
        data_dir/sparse/0/points3D.bin
        data_dir/images/  (training images)
    """
    from services.colmap_loader import (
        read_cameras_binary,
        read_images_binary,
        read_points3D_binary,
        get_intrinsics,
        qvec2rotmat,
    )

    sparse_dir = data_dir / "sparse" / "0"

    # Read COLMAP binary files
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    colmap_images = read_images_binary(sparse_dir / "images.bin")
    xyzs, rgbs = read_points3D_binary(sparse_dir / "points3D.bin")

    # Find images directory
    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Determine image dimensions from the first image
    first_colmap_img = next(iter(colmap_images.values()))
    first_image_path = images_dir / first_colmap_img.name
    if not first_image_path.exists():
        # Try common subdirectories
        for subdir in ["", "images_2", "images_4", "images_8"]:
            test_dir = data_dir / subdir if subdir else images_dir
            test_path = test_dir / first_colmap_img.name
            if test_path.exists():
                images_dir = test_dir
                first_image_path = test_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find image: {first_colmap_img.name} in {images_dir}"
            )

    with Image.open(first_image_path) as img:
        orig_width, orig_height = img.size

    width = int(orig_width * image_scale)
    height = int(orig_height * image_scale)

    # COLMAP cameras.bin stores intrinsics at the original capture resolution,
    # but images on disk may already be downscaled (e.g. half-res).  Compute
    # the total scale from COLMAP camera resolution to our final render size.
    first_camera = cameras[next(iter(colmap_images.values())).camera_id]
    colmap_width = first_camera.width
    disk_scale = orig_width / colmap_width  # ratio of image-on-disk to COLMAP res
    intrinsics_scale = disk_scale * image_scale  # total scale for intrinsics
    if abs(disk_scale - 1.0) > 0.01:
        logger.info(
            "Images on disk (%dx%d) differ from COLMAP camera (%dx%d), "
            "disk_scale=%.4f, total intrinsics_scale=%.4f",
            orig_width, orig_height, colmap_width, first_camera.height,
            disk_scale, intrinsics_scale,
        )

    # Build camera matrices and load images
    viewmats = []
    Ks = []
    loaded_images = []
    camera_centers = []

    # Sort images by ID for deterministic ordering
    for image_id in sorted(colmap_images.keys()):
        colmap_img = colmap_images[image_id]
        camera = cameras[colmap_img.camera_id]

        # Find and load the image file
        image_path = images_dir / colmap_img.name
        if not image_path.exists():
            logger.warning(f"Skipping missing image: {colmap_img.name}")
            continue

        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if image_scale != 1.0:
                img = img.resize((width, height), Image.LANCZOS)
            image_np = np.array(img, dtype=np.float32) / 255.0

        loaded_images.append(image_np)

        # Build world-to-camera matrix directly from COLMAP qvec/tvec
        # COLMAP stores world-to-camera rotation and translation directly
        R = qvec2rotmat(colmap_img.qvec)
        t = colmap_img.tvec

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        viewmats.append(w2c)
        camera_centers.append(-(R.T @ t))

        # Build intrinsic matrix scaled from COLMAP resolution to our render size
        fx, fy, cx, cy = get_intrinsics(camera, scale=intrinsics_scale)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        Ks.append(K)

    if len(loaded_images) == 0:
        raise ValueError("No valid images found")

    logger.info(f"Loaded {len(loaded_images)} images at {width}x{height}")
    logger.info(f"Initial SfM points: {len(xyzs)}")

    scene_center, scene_scale = _estimate_scene_from_camera_positions(
        np.asarray(camera_centers, dtype=np.float32)
    )

    # COLMAP sparse points often include very far outliers. Keep a robust subset
    # near the camera orbit to stabilize early training.
    init_points_np = xyzs.astype(np.float32)
    init_colors_np = np.clip(rgbs.astype(np.float32) / 255.0, 0.0, 1.0)
    radius_mult = float(os.environ.get("SPLAT_INIT_POINT_RADIUS_MULT", "4.0"))
    if len(init_points_np) > 0 and scene_scale > 0:
        dists = np.linalg.norm(init_points_np - scene_center[None, :], axis=1)
        keep = dists <= (scene_scale * radius_mult)
        min_keep = max(1000, int(len(init_points_np) * 0.1))
        if int(keep.sum()) >= min_keep:
            dropped = len(init_points_np) - int(keep.sum())
            if dropped > 0:
                logger.info(
                    "Filtered %d sparse outlier points (radius_mult=%.2f, scene_scale=%.3f)",
                    dropped,
                    radius_mult,
                    scene_scale,
                )
            init_points_np = init_points_np[keep]
            init_colors_np = init_colors_np[keep]

    # Convert to tensors
    viewmats = torch.tensor(np.stack(viewmats), device=device, dtype=torch.float32)
    Ks = torch.tensor(np.stack(Ks), device=device, dtype=torch.float32)
    images_tensor = torch.tensor(
        np.stack(loaded_images), device=device, dtype=torch.float32
    )

    # Use real SfM points/colors from COLMAP for initialization.
    initial_points = torch.tensor(init_points_np, device=device, dtype=torch.float32)
    initial_colors = torch.tensor(init_colors_np, device=device, dtype=torch.float32)

    return {
        "viewmats": viewmats,
        "Ks": Ks,
        "images": images_tensor,
        "initial_points": initial_points,
        "initial_colors": initial_colors,
        "scene_center": torch.tensor(scene_center, device=device, dtype=torch.float32),
        "scene_scale": float(scene_scale),
        "width": width,
        "height": height,
    }


def _load_nerf_data(
    data_dir: Path,
    device: torch.device = torch.device("cuda"),
    image_scale: float = 1.0,
) -> dict:
    """Load cameras and images from transforms.json (NeRF format)."""
    transforms_path = data_dir / "transforms.json"
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

    # Intrinsics in transforms.json may reference a different resolution than
    # the actual images on disk (e.g. iOS exports intrinsics for 2160x3840
    # but saves images at 1080x1920). Compute the ratio to correct for this.
    json_w = transforms.get("w", orig_width)
    json_h = transforms.get("h", orig_height)
    res_scale_x = orig_width / json_w
    res_scale_y = orig_height / json_h
    if res_scale_x != 1.0 or res_scale_y != 1.0:
        logger.info(f"Resolution mismatch: images {orig_width}x{orig_height}, "
                     f"transforms.json {json_w}x{json_h}, "
                     f"scale=({res_scale_x:.3f}, {res_scale_y:.3f})")

    # Total scale: resolution mismatch * user-requested downscale
    total_scale_x = res_scale_x * image_scale
    total_scale_y = res_scale_y * image_scale

    # Compute focal lengths from FOV
    fx = width / (2 * np.tan(camera_angle_x / 2))
    if camera_angle_y is not None:
        fy = height / (2 * np.tan(camera_angle_y / 2))
    else:
        fy = fx  # Assume square pixels

    # Check for explicit intrinsics (overrides FOV-based calculation)
    fl_x = transforms.get("fl_x")
    fl_y = transforms.get("fl_y")
    cx_val = transforms.get("cx")
    cy_val = transforms.get("cy")

    if fl_x is not None:
        fx = fl_x * total_scale_x
    if fl_y is not None:
        fy = fl_y * total_scale_y
    cx = cx_val * total_scale_x if cx_val is not None else width / 2
    cy = cy_val * total_scale_y if cy_val is not None else height / 2

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

        # Parse transform matrix (camera-to-world, c2w).
        # iOS/ARKit exports use OpenCV convention (camera looks along +Z),
        # which matches gsplat's rasterizer.  No axis flip is needed.
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        # Convert to world-to-camera (w2c) for gsplat
        w2c = np.linalg.inv(c2w)
        viewmats.append(w2c)

        # Store camera position (column 3 of the c2w matrix)
        camera_positions.append(c2w[:3, 3])

        # Per-frame intrinsics (if available), scaled to actual render resolution
        pf_fx = frame.get("fl_x")
        pf_fy = frame.get("fl_y")
        pf_cx = frame.get("cx")
        pf_cy = frame.get("cy")

        frame_fx = pf_fx * total_scale_x if pf_fx is not None else fx
        frame_fy = pf_fy * total_scale_y if pf_fy is not None else fy
        frame_cx = pf_cx * total_scale_x if pf_cx is not None else cx
        frame_cy = pf_cy * total_scale_y if pf_cy is not None else cy

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

    scene_center, scene_scale = _estimate_scene_from_camera_positions(
        np.asarray(camera_positions, dtype=np.float32)
    )

    # Initialize points from camera frustums or mesh
    initial_points = _initialize_points_from_cameras(
        camera_positions, data_dir, device
    )

    return {
        "viewmats": viewmats,
        "Ks": Ks,
        "images": images,
        "initial_points": initial_points,
        "initial_colors": None,
        "scene_center": torch.tensor(scene_center, device=device, dtype=torch.float32),
        "scene_scale": float(scene_scale),
        "width": width,
        "height": height,
    }


def _estimate_scene_from_camera_positions(
    camera_positions: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Estimate scene center and scale from camera centers.

    Matches the spirit of reference 3DGS normalization (radius around cameras).
    """
    if camera_positions.size == 0:
        return np.zeros(3, dtype=np.float32), 1.0

    center = camera_positions.mean(axis=0).astype(np.float32)
    dists = np.linalg.norm(camera_positions - center[None, :], axis=1)
    radius = float(dists.max()) if len(dists) > 0 else 1.0
    scene_scale = max(radius * 1.1, 1e-3)
    return center, scene_scale


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
    mesh_files = ["mesh.ply", "mesh.glb", "mesh.obj", "mesh_hint.ply"]
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
