"""
Image-to-mesh reconstruction pipeline using pycolmap.

Pipeline:
1) Sparse reconstruction (or reuse existing sparse model)
2) Image undistortion for MVS
3) PatchMatch stereo
4) Stereo fusion -> fused point cloud
5) Poisson meshing -> mesh.ply
6) Convert mesh.ply -> mesh.glb

Example:
    python scripts/reconstruct_mesh.py ^
      --dataset I:/Deco/truck ^
      --output I:/Deco/truck_mesh_output
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pycolmap
import trimesh


logger = logging.getLogger("reconstruct_mesh")


def _find_colmap_exe(explicit: Optional[Path]) -> Optional[Path]:
    """
    Find a COLMAP executable suitable for dense MVS.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)

    env_exe = Path(os.environ.get("COLMAP_EXE", "")) if os.environ.get("COLMAP_EXE") else None
    if env_exe:
        candidates.append(env_exe)

    # Common local portable install path in this workspace.
    workspace_root = Path(__file__).resolve().parents[2]
    candidates.append(workspace_root / "tools" / "colmap-cuda" / "COLMAP.bat")
    candidates.append(workspace_root / "tools" / "colmap-cuda" / "bin" / "colmap.exe")

    which_colmap = shutil.which("colmap")
    if which_colmap:
        candidates.append(Path(which_colmap))
    which_colmap_bat = shutil.which("COLMAP.bat")
    if which_colmap_bat:
        candidates.append(Path(which_colmap_bat))

    seen = set()
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p
    return None


def _run_command(cmd: list[str], cwd: Optional[Path] = None) -> None:
    logger.info("Running command: %s", " ".join(str(x) for x in cmd))
    subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd) if cwd else None,
    )


def _colmap_cmd(colmap_exe: Path, subcommand: str, args: list[str]) -> list[str]:
    exe = str(colmap_exe)
    if colmap_exe.suffix.lower() in (".bat", ".cmd"):
        return ["cmd", "/c", exe, subcommand, *args]
    return [exe, subcommand, *args]


def _is_colmap_model_dir(path: Path) -> bool:
    return any((path / f).exists() for f in ("cameras.bin", "cameras.txt")) and any(
        (path / f).exists() for f in ("images.bin", "images.txt")
    )


def _pick_sparse_model(dataset_dir: Path, explicit_sparse: Optional[Path]) -> Optional[Path]:
    if explicit_sparse:
        return explicit_sparse if _is_colmap_model_dir(explicit_sparse) else None

    common = [
        dataset_dir / "sparse" / "0",
        dataset_dir / "sparse",
    ]
    for p in common:
        if _is_colmap_model_dir(p):
            return p
    return None


def _build_sparse_reconstruction(dataset_dir: Path, sparse_out_root: Path) -> Path:
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    db_path = sparse_out_root / "database.db"
    sparse_out_root.mkdir(parents=True, exist_ok=True)
    map_dir = sparse_out_root / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting features...")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
    )

    logger.info("Matching features (exhaustive)...")
    pycolmap.match_exhaustive(
        database_path=str(db_path),
    )

    logger.info("Running incremental mapping...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(map_dir),
    )
    if not reconstructions:
        raise RuntimeError("incremental_mapping produced no reconstruction")

    first_id = sorted(reconstructions.keys())[0]
    sparse_model = map_dir / str(first_id)
    if not _is_colmap_model_dir(sparse_model):
        raise RuntimeError(f"Sparse model missing from expected path: {sparse_model}")
    return sparse_model


def _run_dense_and_mesh(
    dataset_dir: Path,
    sparse_model_dir: Path,
    output_dir: Path,
    max_image_size: int,
    poisson_depth: int,
    colmap_exe: Optional[Path],
) -> tuple[Path, Path, Path]:
    images_dir = dataset_dir / "images"
    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    if colmap_exe is not None:
        logger.info("Using external COLMAP CLI for dense reconstruction: %s", colmap_exe)

        undistort_args = [
            "--image_path", str(images_dir),
            "--input_path", str(sparse_model_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ]
        _run_command(_colmap_cmd(colmap_exe, "image_undistorter", undistort_args))

        patch_args = [
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
        ]
        if max_image_size > 0:
            patch_args.extend(["--PatchMatchStereo.max_image_size", str(max_image_size)])
        _run_command(_colmap_cmd(colmap_exe, "patch_match_stereo", patch_args))
    else:
        logger.info("Undistorting images for MVS (pycolmap)...")
        pycolmap.undistort_images(
            output_path=str(dense_dir),
            input_path=str(sparse_model_dir),
            image_path=str(images_dir),
            output_type="COLMAP",
        )

        logger.info("Running PatchMatch stereo (pycolmap)...")
        patch_opts = pycolmap.PatchMatchOptions()
        if max_image_size > 0:
            patch_opts.max_image_size = max_image_size
        pycolmap.patch_match_stereo(
            workspace_path=str(dense_dir),
            workspace_format="COLMAP",
            options=patch_opts,
        )

    fused_ply = output_dir / "fused.ply"
    if colmap_exe is not None:
        fusion_args = [
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused_ply),
        ]
        if max_image_size > 0:
            fusion_args.extend(["--StereoFusion.max_image_size", str(max_image_size)])
        _run_command(_colmap_cmd(colmap_exe, "stereo_fusion", fusion_args))
    else:
        logger.info("Running stereo fusion (pycolmap)...")
        fusion_opts = pycolmap.StereoFusionOptions()
        if max_image_size > 0:
            fusion_opts.max_image_size = max_image_size
        pycolmap.stereo_fusion(
            output_path=str(fused_ply),
            workspace_path=str(dense_dir),
            workspace_format="COLMAP",
            input_type="geometric",
            options=fusion_opts,
        )
    if not fused_ply.exists():
        raise RuntimeError("stereo_fusion did not create fused.ply")

    mesh_ply = output_dir / "mesh_poisson.ply"
    if colmap_exe is not None:
        poisson_args = [
            "--input_path", str(fused_ply),
            "--output_path", str(mesh_ply),
            "--PoissonMeshing.depth", str(poisson_depth),
        ]
        _run_command(_colmap_cmd(colmap_exe, "poisson_mesher", poisson_args))
    else:
        logger.info("Running Poisson meshing (pycolmap)...")
        poisson_opts = pycolmap.PoissonMeshingOptions()
        poisson_opts.depth = poisson_depth
        pycolmap.poisson_meshing(
            input_path=str(fused_ply),
            output_path=str(mesh_ply),
            options=poisson_opts,
        )
    if not mesh_ply.exists():
        raise RuntimeError("poisson_meshing did not create mesh_poisson.ply")

    mesh_glb = output_dir / "mesh.glb"
    _mesh_ply_to_glb(mesh_ply, mesh_glb)

    return fused_ply, mesh_ply, mesh_glb


def _mesh_ply_to_glb(mesh_ply: Path, mesh_glb: Path) -> None:
    logger.info(f"Converting mesh PLY to GLB: {mesh_ply} -> {mesh_glb}")
    mesh = trimesh.load(str(mesh_ply), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unexpected mesh type from PLY: {type(mesh)}")
    mesh.export(str(mesh_glb))


def _camera_orbit_center_and_scale(
    reconstruction: pycolmap.Reconstruction,
) -> Tuple[np.ndarray, float]:
    """
    Compute robust scene center/scale from camera centers.
    """
    centers = []
    for img in reconstruction.images.values():
        T = img.cam_from_world()
        R = T.rotation.matrix()
        t = np.asarray(T.translation, dtype=np.float32)
        centers.append(-(R.T @ t))

    if not centers:
        return np.zeros(3, dtype=np.float32), 1.0

    centers_np = np.asarray(centers, dtype=np.float32)
    center = centers_np.mean(axis=0)
    radius = float(np.linalg.norm(centers_np - center[None, :], axis=1).max())
    return center, max(radius * 1.1, 1e-3)


def _write_points_ply(points_xyz: np.ndarray, points_rgb: np.ndarray, out_ply: Path) -> None:
    from plyfile import PlyData, PlyElement

    normals = np.zeros_like(points_xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    arr = np.empty(len(points_xyz), dtype=dtype)
    arr["x"] = points_xyz[:, 0]
    arr["y"] = points_xyz[:, 1]
    arr["z"] = points_xyz[:, 2]
    arr["nx"] = normals[:, 0]
    arr["ny"] = normals[:, 1]
    arr["nz"] = normals[:, 2]
    arr["red"] = points_rgb[:, 0]
    arr["green"] = points_rgb[:, 1]
    arr["blue"] = points_rgb[:, 2]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(out_ply))


def _export_filtered_sparse_points(
    reconstruction: pycolmap.Reconstruction,
    out_ply: Path,
    min_track_len: int,
    max_reproj_error: float,
    crop_radius_mult: float,
) -> dict:
    """
    Export filtered sparse points to PLY.

    Filters weak/outlier points that commonly cause blob-like fallback meshes.
    """
    center, scene_scale = _camera_orbit_center_and_scale(reconstruction)
    crop_radius = crop_radius_mult * scene_scale if crop_radius_mult > 0 else np.inf

    xyz = []
    rgb = []
    total = 0
    reject_track = 0
    reject_error = 0
    reject_radius = 0

    for p in reconstruction.points3D.values():
        total += 1
        tr_len = p.track.length()
        if tr_len < min_track_len:
            reject_track += 1
            continue

        err = float(p.error)
        if err > max_reproj_error:
            reject_error += 1
            continue

        pos = np.asarray(p.xyz, dtype=np.float32)
        if np.isfinite(crop_radius):
            if float(np.linalg.norm(pos - center)) > crop_radius:
                reject_radius += 1
                continue

        xyz.append(pos)
        rgb.append(np.asarray(p.color, dtype=np.uint8))

    xyz_np = np.asarray(xyz, dtype=np.float32)
    rgb_np = np.asarray(rgb, dtype=np.uint8)

    if len(xyz_np) < 5000:
        raise RuntimeError(
            "Too few sparse points after filtering; "
            f"kept={len(xyz_np)} from total={total}"
        )

    _write_points_ply(xyz_np, rgb_np, out_ply)

    return {
        "total_points": total,
        "kept_points": int(len(xyz_np)),
        "rejected_track": reject_track,
        "rejected_error": reject_error,
        "rejected_radius": reject_radius,
        "scene_scale": float(scene_scale),
        "crop_radius": float(crop_radius),
        "bbox_min": xyz_np.min(axis=0).tolist(),
        "bbox_max": xyz_np.max(axis=0).tolist(),
    }


def _run_sparse_fallback_mesh(
    sparse_model_dir: Path,
    output_dir: Path,
    sparse_poisson_depth: int,
    min_component_faces: int,
    min_track_len: int,
    max_reproj_error: float,
    crop_radius_mult: float,
    outlier_prop_threshold: float,
    target_faces: int,
) -> tuple[Path, Path]:
    """
    Fallback mesh generation from sparse COLMAP points when dense MVS is unavailable.
    """
    import pymeshlab as ml

    sparse_points_ply = output_dir / "sparse_points.ply"
    logger.info("Exporting filtered sparse COLMAP points...")
    reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
    stats = _export_filtered_sparse_points(
        reconstruction=reconstruction,
        out_ply=sparse_points_ply,
        min_track_len=min_track_len,
        max_reproj_error=max_reproj_error,
        crop_radius_mult=crop_radius_mult,
    )
    logger.info(
        "Sparse filters: kept=%d/%d (track reject=%d, error reject=%d, radius reject=%d), "
        "scene_scale=%.3f, crop_radius=%.3f",
        stats["kept_points"],
        stats["total_points"],
        stats["rejected_track"],
        stats["rejected_error"],
        stats["rejected_radius"],
        stats["scene_scale"],
        stats["crop_radius"],
    )
    logger.info(
        "Filtered sparse bbox: %s .. %s",
        np.array2string(np.asarray(stats["bbox_min"]), precision=3),
        np.array2string(np.asarray(stats["bbox_max"]), precision=3),
    )

    ms = ml.MeshSet()
    ms.load_new_mesh(str(sparse_points_ply))
    logger.info("Removing point cloud outliers (fallback mesh path)...")
    ms.compute_selection_point_cloud_outliers(
        knearest=32,
        propthreshold=outlier_prop_threshold,
    )
    ms.meshing_remove_selected_vertices()

    logger.info("Estimating/smoothing point normals (fallback mesh path)...")
    ms.compute_normal_for_point_clouds(k=24, smoothiter=2)
    ms.apply_normal_point_cloud_smoothing(k=24, usedist=True)

    logger.info("Running screened Poisson on sparse points (fallback mesh path)...")
    ms.generate_surface_reconstruction_screened_poisson(
        depth=sparse_poisson_depth,
        fulldepth=max(5, min(sparse_poisson_depth - 1, 8)),
    )
    if min_component_faces > 0:
        logger.info(
            f"Removing tiny disconnected mesh components (< {min_component_faces} faces)..."
        )
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=min_component_faces,
            removeunref=True,
        )

    if target_faces > 0 and ms.current_mesh().face_number() > target_faces:
        logger.info(f"Decimating mesh to ~{target_faces} faces...")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            qualitythr=0.4,
            autoclean=True,
        )

    mesh_ply = output_dir / "mesh_sparse_poisson.ply"
    ms.save_current_mesh(str(mesh_ply), binary=True)
    if not mesh_ply.exists():
        raise RuntimeError("Sparse fallback meshing did not create mesh_sparse_poisson.ply")

    mesh_glb = output_dir / "mesh.glb"
    _mesh_ply_to_glb(mesh_ply, mesh_glb)
    return mesh_ply, mesh_glb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from images with pycolmap.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory containing images/")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for reconstruction artifacts")
    parser.add_argument(
        "--colmap-exe",
        type=Path,
        default=None,
        help="Path to external COLMAP executable (.exe/.bat). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--sparse-model",
        type=Path,
        default=None,
        help="Existing COLMAP sparse model directory (e.g. dataset/sparse/0). If omitted, auto-detect or rebuild.",
    )
    parser.add_argument(
        "--rebuild-sparse",
        action="store_true",
        help="Ignore existing sparse model and rebuild from images.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Max image size used by dense steps. -1 keeps original resolution.",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=11,
        help="Poisson depth (higher = more detail and memory use).",
    )
    parser.add_argument(
        "--sparse-poisson-depth",
        type=int,
        default=9,
        help="Poisson depth used by sparse fallback meshing.",
    )
    parser.add_argument(
        "--min-component-faces",
        type=int,
        default=5000,
        help="Remove disconnected components smaller than this face count in sparse fallback mode.",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=6,
        help="Minimum COLMAP track length for sparse fallback points.",
    )
    parser.add_argument(
        "--max-reproj-error",
        type=float,
        default=2.0,
        help="Maximum COLMAP reprojection error for sparse fallback points.",
    )
    parser.add_argument(
        "--sparse-crop-radius-mult",
        type=float,
        default=1.8,
        help="Keep sparse points within this multiple of camera-orbit scene scale.",
    )
    parser.add_argument(
        "--sparse-outlier-prop-threshold",
        type=float,
        default=0.9,
        help="Pymeshlab point-cloud outlier threshold (higher removes more points).",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=250000,
        help="Optional target face count for fallback mesh decimation (0 disables).",
    )
    parser.add_argument(
        "--no-sparse-fallback",
        action="store_true",
        help="Fail if dense PatchMatch is unavailable instead of using sparse fallback.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove and recreate output directory before processing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_dir = args.dataset.resolve()
    output_dir = args.output.resolve()

    if not dataset_dir.exists():
        logger.error(f"Dataset does not exist: {dataset_dir}")
        return 1
    if not (dataset_dir / "images").exists():
        logger.error(f"Dataset is missing images directory: {dataset_dir / 'images'}")
        return 1

    if args.clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        sparse_model_dir: Optional[Path] = None
        if not args.rebuild_sparse:
            sparse_model_dir = _pick_sparse_model(dataset_dir, args.sparse_model)

        if sparse_model_dir is None:
            logger.info("No usable sparse model found, building sparse reconstruction...")
            sparse_root = output_dir / "sparse_build"
            sparse_model_dir = _build_sparse_reconstruction(dataset_dir, sparse_root)
        else:
            logger.info(f"Using existing sparse model: {sparse_model_dir}")

        colmap_exe = _find_colmap_exe(args.colmap_exe)
        if colmap_exe:
            logger.info(f"Detected COLMAP executable: {colmap_exe}")
        else:
            logger.info("No external COLMAP executable detected; using pycolmap dense path.")

        fused_ply = None
        mesh_ply = None
        mesh_glb = None
        try:
            fused_ply, mesh_ply, mesh_glb = _run_dense_and_mesh(
                dataset_dir=dataset_dir,
                sparse_model_dir=sparse_model_dir,
                output_dir=output_dir,
                max_image_size=args.max_image_size,
                poisson_depth=args.poisson_depth,
                colmap_exe=colmap_exe,
            )
        except Exception as dense_exc:
            if args.no_sparse_fallback:
                raise
            logger.warning(f"Dense path failed, switching to sparse fallback: {dense_exc}")
            mesh_ply, mesh_glb = _run_sparse_fallback_mesh(
                sparse_model_dir=sparse_model_dir,
                output_dir=output_dir,
                sparse_poisson_depth=args.sparse_poisson_depth,
                min_component_faces=args.min_component_faces,
                min_track_len=args.min_track_len,
                max_reproj_error=args.max_reproj_error,
                crop_radius_mult=args.sparse_crop_radius_mult,
                outlier_prop_threshold=args.sparse_outlier_prop_threshold,
                target_faces=args.target_faces,
            )

        elapsed = time.time() - start
        logger.info("Reconstruction completed.")
        if fused_ply:
            logger.info(f"Fused cloud: {fused_ply}")
        else:
            logger.info("Fused cloud: (not available, sparse fallback used)")
        logger.info(f"Mesh PLY:    {mesh_ply}")
        logger.info(f"Mesh GLB:    {mesh_glb}")
        logger.info(f"Elapsed:     {elapsed:.1f}s")
        return 0
    except Exception as exc:
        logger.exception(f"Reconstruction failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
