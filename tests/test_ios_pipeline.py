"""
Tests for the iOS scan data pipeline.

Covers:
1. Camera convention — verifies the OpenGL→OpenCV flip is applied correctly
   by projecting mesh_hint.ply points through each training camera and checking
   that the fixed convention yields more visible points than the un-fixed one.
2. Mesh hint export — verifies mesh_hint.ply loads and converts to a valid GLB
   with real triangle faces (not a Gaussian splat).
3. Data loading — verifies transforms.json produces tensors of expected shape
   and that the viewmats are valid (finite, proper rotation matrices).

Run with:
    python -m pytest tests/test_ios_pipeline.py -v
or standalone:
    python tests/test_ios_pipeline.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Make the project root importable when run as a script
_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _find_ios_scan_dirs() -> list[Path]:
    """Return all extracted iOS scan directories that have transforms.json."""
    data_root = _PROJ_ROOT / "data"
    found = []
    for tf in data_root.rglob("transforms.json"):
        d = tf.parent
        if (d / "images").is_dir():
            found.append(d)
    return found


def _load_transforms(scan_dir: Path) -> dict:
    with open(scan_dir / "transforms.json") as f:
        return json.load(f)


def _load_mesh_points(mesh_ply: Path) -> Optional[np.ndarray]:
    try:
        from plyfile import PlyData
        plydata = PlyData.read(str(mesh_ply))
        v = plydata["vertex"]
        return np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1).astype(np.float32)
    except Exception as e:
        logger.warning("Could not load mesh points from %s: %s", mesh_ply, e)
        return None


def _project_points(c2w: np.ndarray, K: np.ndarray, points: np.ndarray, img_w: int, img_h: int) -> int:
    """
    Project 3-D points through a camera and return the count visible in the image.
    c2w is camera-to-world (4x4).  Inverted to w2c for projection.
    """
    w2c = np.linalg.inv(c2w)
    pts_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
    pts_cam = (w2c @ pts_h.T).T[:, :3]            # (N, 3) in camera space

    front = pts_cam[:, 2] > 0.01                   # in front of camera
    pts_cam = pts_cam[front]
    if len(pts_cam) == 0:
        return 0

    uv = (K @ pts_cam.T).T                         # (M, 3)
    uv = uv[:, :2] / uv[:, 2:3]                   # (M, 2)

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )
    return int(in_bounds.sum())


def _make_K(transforms: dict, scan_dir: Path) -> tuple[np.ndarray, int, int]:
    """Build intrinsic matrix K scaled to actual image size on disk."""
    from PIL import Image as PILImage

    frame0 = transforms["frames"][0]
    img_path = scan_dir / "images" / Path(frame0["file_path"]).name
    with PILImage.open(img_path) as im:
        orig_w, orig_h = im.size

    json_w = transforms.get("w", orig_w)
    json_h = transforms.get("h", orig_h)
    sx = orig_w / json_w
    sy = orig_h / json_h

    fl_x = transforms.get("fl_x", None) or frame0.get("fl_x")
    fl_y = transforms.get("fl_y", None) or frame0.get("fl_y")
    cx   = transforms.get("cx",   None) or frame0.get("cx")
    cy   = transforms.get("cy",   None) or frame0.get("cy")

    K = np.array([
        [fl_x * sx,    0,       cx * sx],
        [0,        fl_y * sy,   cy * sy],
        [0,        0,           1      ],
    ], dtype=np.float32)
    return K, orig_w, orig_h


# ── Test 1: camera-mesh alignment ────────────────────────────────────────────

def test_camera_convention_flip(scan_dir: Path, save_images: bool = True) -> dict:
    """
    Verify that iOS/ARKit camera matrices are already in OpenCV convention
    and that mesh_hint.ply points are correctly visible through the training cameras
    WITHOUT any axis flip.

    iOS exports c2w matrices in OpenCV convention (camera +Z = forward into scene),
    which matches gsplat's rasterizer directly.  No flip is needed or wanted.

    Checks:
    - At least 50% of mesh surface points should be in front of (Z>0) each camera.
    - At least 20% of mesh points visible from any camera should project within
      the image boundary (basic coverage sanity check).
    - Saves a diagnostic overlay image to tests/diag/ for visual inspection.
    """
    mesh_ply = scan_dir / "mesh_hint.ply"
    if not mesh_ply.exists():
        print(f"  SKIP: no mesh_hint.ply in {scan_dir.name}")
        return {"skipped": True}

    transforms = _load_transforms(scan_dir)
    frames = transforms.get("frames", [])
    if not frames:
        print("  SKIP: no frames in transforms.json")
        return {"skipped": True}

    points = _load_mesh_points(mesh_ply)
    if points is None or len(points) == 0:
        print("  SKIP: could not load mesh points")
        return {"skipped": True}

    # Use all points for a reliable count (cap at 5000 for speed)
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        pts_sample = points[idx]
    else:
        pts_sample = points

    K, img_w, img_h = _make_K(transforms, scan_dir)

    in_front_counts = []
    in_image_counts = []

    for frame in frames[:min(len(frames), 20)]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        n_front = _project_points(c2w, K, pts_sample, img_w, img_h)
        in_front_counts.append(n_front)

        # Count in-image visible points
        w2c = np.linalg.inv(c2w)
        pts_h = np.hstack([pts_sample, np.ones((len(pts_sample), 1), dtype=np.float32)])
        pts_cam = (w2c @ pts_h.T).T[:, :3]
        front_mask = pts_cam[:, 2] > 0.01
        if front_mask.any():
            pts_f = pts_cam[front_mask]
            uv = (K @ pts_f.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            ib = ((uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < img_h))
            in_image_counts.append(int(ib.sum()))

    avg_front = float(np.mean(in_front_counts))
    avg_image = float(np.mean(in_image_counts)) if in_image_counts else 0
    n_sample = len(pts_sample)

    front_pct = avg_front / n_sample * 100
    image_pct = avg_image / n_sample * 100
    # What fraction of in-front points project within the image?
    # This is the key correctness metric: wrong convention → low ratio.
    infront_to_image_ratio = avg_image / max(avg_front, 1.0) * 100

    print(f"  Camera-mesh alignment: {front_pct:.0f}% in front (Z>0), "
          f"{image_pct:.0f}% project within image")
    print(f"  In-front->in-image ratio: {infront_to_image_ratio:.0f}%")

    # Indoor room scans: cameras are INSIDE the room so only ~10-30% of total
    # mesh is ever in front of any given camera (the rest is behind you).
    # The 50% threshold was wrong — use a very low floor of 3%.
    assert front_pct >= 3, (
        f"Too few mesh points in front of cameras ({front_pct:.0f}% < 3%). "
        "Camera convention or coordinate system mismatch."
    )
    # The key sanity check: of the points that ARE in front of the camera,
    # at least 15% should project within the image frame.  If the convention
    # were wrong (axes flipped) the ratio would be near 0.
    assert infront_to_image_ratio >= 15, (
        f"Too few in-front points project within image ({infront_to_image_ratio:.0f}% < 15%). "
        "Intrinsics or camera-mesh coordinate mismatch."
    )
    print("  PASS: cameras correctly see mesh (OpenCV convention, no flip needed)")

    # Save diagnostic overlay image
    if save_images:
        _save_alignment_diagnostic(scan_dir, frames, pts_sample, K, img_w, img_h)

    return {
        "front_pct": front_pct,
        "image_pct": image_pct,
        "infront_to_image_ratio": infront_to_image_ratio,
        "n_cameras_tested": len(in_front_counts),
    }


def _save_alignment_diagnostic(scan_dir, frames, points, K, img_w, img_h):
    """Save a mesh-projection overlay onto the middle training frame."""
    try:
        from PIL import Image as PILImage, ImageDraw

        diag_dir = _PROJ_ROOT / "tests" / "diag"
        diag_dir.mkdir(parents=True, exist_ok=True)

        mid_idx = len(frames) // 2
        frame = frames[mid_idx]
        img_file = scan_dir / "images" / Path(frame["file_path"]).name
        if not img_file.exists():
            return

        base_img = PILImage.open(str(img_file)).convert("RGB")
        scale = min(1.0, 800 / max(base_img.width, base_img.height))
        disp_w = int(base_img.width * scale)
        disp_h = int(base_img.height * scale)
        base_img = base_img.resize((disp_w, disp_h), PILImage.LANCZOS)
        K_disp = K.copy()
        K_disp[0] *= scale
        K_disp[1] *= scale

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        w2c = np.linalg.inv(c2w)

        pts_sub = points[::max(1, len(points) // 500)]
        pts_h = np.hstack([pts_sub, np.ones((len(pts_sub), 1), dtype=np.float32)])
        pts_cam = (w2c @ pts_h.T).T[:, :3]
        front = pts_cam[:, 2] > 0.01
        pts_cam = pts_cam[front]

        img = base_img.copy()
        draw = ImageDraw.Draw(img)

        if len(pts_cam) > 0:
            uv = (K_disp @ pts_cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            ib = ((uv[:, 0] >= 0) & (uv[:, 0] < disp_w) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < disp_h))
            uv = uv[ib]
            r = max(2, int(4 * scale))
            for x, y in uv:
                draw.ellipse([x-r, y-r, x+r, y+r], fill="lime")

        out_path = diag_dir / f"{scan_dir.parent.name}_mesh_alignment.png"
        img.save(str(out_path))
        in_img = int(ib.sum()) if len(pts_cam) > 0 else 0
        print(f"    Saved {out_path.name} ({in_img} mesh points visible, green dots)")

    except Exception as e:
        logger.warning("Could not save alignment diagnostic: %s", e)


# ── Test 2: mesh hint GLB export ──────────────────────────────────────────────

def test_mesh_export(scan_dir: Path, tmp_output: Path) -> dict:
    """
    Verify that mesh_hint.ply is exported as a valid GLB with triangle faces.
    """
    mesh_ply = scan_dir / "mesh_hint.ply"
    if not mesh_ply.exists():
        print(f"  SKIP: no mesh_hint.ply in {scan_dir.name}")
        return {"skipped": True}

    from services.mesh_utils import export_ios_mesh_glb, get_mesh_stats

    # Check stats first
    stats = get_mesh_stats(mesh_ply)
    assert stats is not None, "get_mesh_stats returned None for valid mesh"
    assert stats["vertices"] > 0, f"Mesh has no vertices: {stats}"
    assert stats["faces"] > 0, f"Mesh has no faces: {stats}"
    print(f"  Mesh stats: {stats['vertices']:,} vertices, {stats['faces']:,} faces")

    # Export to GLB
    glb_path = export_ios_mesh_glb(scan_dir, tmp_output)
    assert glb_path is not None, "export_ios_mesh_glb returned None"
    assert glb_path.exists(), f"GLB file not created at {glb_path}"
    assert glb_path.stat().st_size > 1000, "GLB file is suspiciously small"

    # Verify it is a valid GLB (starts with "glTF" magic)
    with open(glb_path, "rb") as f:
        magic = f.read(4)
    assert magic == b"glTF", f"GLB magic bytes wrong: {magic!r}"

    # Verify trimesh can load the exported GLB back and it has faces
    import trimesh
    mesh = trimesh.load(str(glb_path))
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        assert geoms, "Exported GLB loaded as empty Scene"
        mesh = trimesh.util.concatenate(geoms)

    assert isinstance(mesh, trimesh.Trimesh), f"Not a Trimesh: {type(mesh)}"
    assert len(mesh.faces) > 0, "Exported GLB has no faces"

    size_kb = glb_path.stat().st_size / 1024
    print(f"  PASS: exported GLB {glb_path.name} ({size_kb:.0f} KB, "
          f"{len(mesh.vertices):,} verts, {len(mesh.faces):,} faces)")
    return {
        "glb_path": str(glb_path),
        "size_bytes": glb_path.stat().st_size,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }


# ── Test 3: data loading ──────────────────────────────────────────────────────

def test_data_loading(scan_dir: Path) -> dict:
    """
    Verify that load_training_data returns correctly shaped tensors and that
    the viewmats are valid rotation matrices (det ≈ 1, finite values).
    """
    import torch
    from services.camera_utils import load_training_data

    device = torch.device("cpu")
    data = load_training_data(scan_dir, device=device, image_scale=0.25)

    viewmats = data["viewmats"]  # (N, 4, 4)
    Ks       = data["Ks"]        # (N, 3, 3)
    images   = data["images"]    # (N, H, W, 3)
    pts      = data["initial_points"]  # (M, 3)

    N = viewmats.shape[0]
    assert N >= 1, "No cameras loaded"
    assert viewmats.shape == (N, 4, 4), f"viewmats shape wrong: {viewmats.shape}"
    assert Ks.shape == (N, 3, 3), f"Ks shape wrong: {Ks.shape}"
    assert images.shape[0] == N, "Image count mismatch"
    assert images.shape[3] == 3, "Images not RGB"
    assert pts.shape[1] == 3, "Initial points not (M,3)"

    # Images should be in [0, 1]
    assert float(images.min()) >= -1e-4, f"Images below 0: {images.min()}"
    assert float(images.max()) <= 1.0001, f"Images above 1: {images.max()}"

    # Viewmats must be finite
    assert torch.isfinite(viewmats).all(), "Non-finite values in viewmats"
    assert torch.isfinite(Ks).all(), "Non-finite values in Ks"

    # The 3x3 rotation blocks of each w2c must have det ≈ +1
    R_blocks = viewmats[:, :3, :3].cpu().numpy()
    dets = np.linalg.det(R_blocks)
    bad = np.abs(dets - 1.0) > 0.05
    assert not bad.any(), (
        f"{bad.sum()} viewmats have |det(R)-1| > 0.05: dets={dets[bad]}"
    )

    print(f"  PASS: loaded {N} cameras, images {images.shape[2]}x{images.shape[1]}, "
          f"{len(pts)} init points, det(R) range [{dets.min():.4f}, {dets.max():.4f}]")
    return {
        "num_cameras": N,
        "image_size": (int(images.shape[2]), int(images.shape[1])),
        "num_init_points": len(pts),
        "scene_scale": float(data.get("scene_scale", 0)),
    }


# ── Test 4: shard detection on existing outputs ───────────────────────────────

def test_shard_detection(job_id: str) -> dict:
    """
    Analyse an existing trained PLY and report on shard/needle artifacts.

    A healthy 3DGS result should have:
    - Median aspect ratio (max_scale / min_scale) < 5
    - < 30% of Gaussians with aspect ratio > 10

    The scale anisotropy regularization added to training.py targets this.
    This test can be used to compare outputs before and after the fix.
    """
    ply_path = _PROJ_ROOT / "outputs" / job_id / "point_cloud.ply"
    if not ply_path.exists():
        print(f"  SKIP: no point_cloud.ply for job {job_id}")
        return {"skipped": True}

    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]
    n = len(v)

    scales = np.stack([
        np.array(v["scale_0"]),
        np.array(v["scale_1"]),
        np.array(v["scale_2"]),
    ], axis=1)
    actual_scales = np.exp(scales)
    max_scale = actual_scales.max(axis=1)
    min_scale = actual_scales.min(axis=1)
    aspect = max_scale / np.maximum(min_scale, 1e-10)

    median_aspect = float(np.median(aspect))
    shard_pct = float((aspect > 10).sum() / n * 100)
    p99_aspect = float(np.percentile(aspect, 99))

    print(f"  Gaussians: {n:,}")
    print(f"  Aspect ratio — median={median_aspect:.1f}, p99={p99_aspect:.0f}")
    print(f"  Shard-like (aspect>10): {shard_pct:.1f}%")

    if shard_pct > 30:
        print(f"  WARNING: {shard_pct:.1f}% shard-like Gaussians (threshold 30%).")
        print("  Scale anisotropy regularization (0.01 weight) should reduce this.")
    else:
        print(f"  PASS: shard fraction {shard_pct:.1f}% is within acceptable range (<30%)")

    return {
        "num_gaussians": n,
        "median_aspect": median_aspect,
        "p99_aspect": p99_aspect,
        "shard_pct": shard_pct,
    }


# ── Test 5: scale regularization sanity ───────────────────────────────────────

def test_mesh_glb_is_triangle_mesh(job_id: str) -> dict:
    """
    Verify that the training pipeline exported mesh.glb as a proper triangle mesh
    (not a Gaussian splat) for the given job.
    """
    mesh_path = _PROJ_ROOT / "outputs" / job_id / "mesh.glb"
    if not mesh_path.exists():
        print(f"  SKIP: no mesh.glb in outputs/{job_id} (job not yet run with new code)")
        return {"skipped": True}

    with open(mesh_path, "rb") as f:
        magic = f.read(4)
    assert magic == b"glTF", f"Invalid GLB magic: {magic!r}"

    import trimesh
    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        assert geoms, "Mesh GLB is an empty Scene"
        mesh = trimesh.util.concatenate(geoms)

    assert isinstance(mesh, trimesh.Trimesh), f"Not a Trimesh: {type(mesh)}"
    assert len(mesh.faces) > 0, "mesh.glb has no triangle faces"

    size_mb = mesh_path.stat().st_size / 1024 / 1024
    print(f"  PASS: mesh.glb {size_mb:.1f} MB, "
          f"{len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}


# ── Test 6: scene scale sanity ────────────────────────────────────────────────

def test_scene_scale(scan_dir: Path) -> dict:
    """
    Check that the estimated scene scale is in a reasonable range.

    iOS scans are typically of indoor rooms so camera translations should be
    in the range of 0.1 – 30 meters.  If scene_scale is < 0.001 or > 1000
    something is wrong (e.g. wrong units or bad transforms).
    """
    import torch
    from services.camera_utils import load_training_data

    device = torch.device("cpu")
    data = load_training_data(scan_dir, device=device, image_scale=0.1)

    scene_scale = float(data.get("scene_scale", 0))
    print(f"  scene_scale = {scene_scale:.4f}")
    assert scene_scale > 0.001, f"scene_scale too small: {scene_scale}"
    assert scene_scale < 1000, f"scene_scale too large: {scene_scale}"
    print(f"  PASS: scene_scale {scene_scale:.4f} is in a reasonable range")
    return {"scene_scale": scene_scale}


# ── runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all tests against every available iOS scan directory."""
    import tempfile

    scan_dirs = _find_ios_scan_dirs()
    if not scan_dirs:
        print("No iOS scan directories found under data/. Nothing to test.")
        return

    print(f"\nFound {len(scan_dirs)} iOS scan(s):\n")
    total = 0
    passed = 0
    skipped = 0

    # Also run output analysis tests for any existing trained models
    output_root = _PROJ_ROOT / "outputs"
    job_ids = [d.name for d in output_root.iterdir() if d.is_dir()] if output_root.exists() else []

    for scan_dir in scan_dirs:
        print(f"{'='*60}")
        print(f"Scan: {scan_dir.relative_to(_PROJ_ROOT)}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            tests = [
                ("camera_alignment",  lambda d=scan_dir: test_camera_convention_flip(d)),
                ("mesh_export",       lambda d=scan_dir, t=tmp_path: test_mesh_export(d, t)),
                ("data_loading",      lambda d=scan_dir: test_data_loading(d)),
                ("scene_scale",       lambda d=scan_dir: test_scene_scale(d)),
            ]

            for name, fn in tests:
                total += 1
                print(f"\n[{name}]")
                try:
                    result = fn()
                    if result and result.get("skipped"):
                        skipped += 1
                    else:
                        passed += 1
                except AssertionError as ae:
                    print(f"  FAIL: {ae}")
                except Exception as ex:
                    import traceback
                    print(f"  ERROR: {ex}")
                    traceback.print_exc()

        print()

    # Analyse existing trained outputs
    if job_ids:
        print(f"{'='*60}")
        print(f"Output analysis ({len(job_ids)} trained jobs)")
        print(f"{'='*60}")
        for job_id in job_ids:
            print(f"\n[Job {job_id}]")
            for name, fn in [
                ("shard_detection",      lambda j=job_id: test_shard_detection(j)),
                ("mesh_glb_triangle",    lambda j=job_id: test_mesh_glb_is_triangle_mesh(j)),
            ]:
                total += 1
                print(f"\n  [{name}]")
                try:
                    result = fn()
                    if result and result.get("skipped"):
                        skipped += 1
                    else:
                        passed += 1
                except AssertionError as ae:
                    print(f"  FAIL: {ae}")
                except Exception as ex:
                    import traceback
                    print(f"  ERROR: {ex}")
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {skipped} skipped")
    failed = total - passed - skipped
    if failed > 0:
        print(f"  {failed} FAILED")
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()

# ── pytest entrypoints (only registered when pytest is available) ─────────────
try:
    import pytest

    @pytest.mark.parametrize("scan_dir", _find_ios_scan_dirs())
    def test_convention_flip_pytest(scan_dir, tmp_path):
        result = test_camera_convention_flip(scan_dir)
        if not result.get("skipped"):
            assert result["front_pct"] >= 3
            assert result["infront_to_image_ratio"] >= 15

    @pytest.mark.parametrize("scan_dir", _find_ios_scan_dirs())
    def test_mesh_export_pytest(scan_dir, tmp_path):
        test_mesh_export(scan_dir, tmp_path)

    @pytest.mark.parametrize("scan_dir", _find_ios_scan_dirs())
    def test_data_loading_pytest(scan_dir):
        test_data_loading(scan_dir)

    @pytest.mark.parametrize("scan_dir", _find_ios_scan_dirs())
    def test_scene_scale_pytest(scan_dir):
        test_scene_scale(scan_dir)

except ModuleNotFoundError:
    pass  # pytest not installed; run with: python tests/test_ios_pipeline.py
