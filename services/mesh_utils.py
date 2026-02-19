"""
Mesh processing utilities for iOS LiDAR/ARKit scan data.

iOS scans include a mesh_hint.ply — a triangle mesh produced by ARKit's
scene reconstruction. This module converts it to a standard GLB file
(real triangle faces, not a Gaussian splat) so any 3D viewer can open it.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def export_ios_mesh_glb(data_dir: Path, output_dir: Path) -> Optional[Path]:
    """
    Find and export the iOS LiDAR mesh hint as a proper triangle-mesh GLB.

    Looks for mesh_hint.ply in data_dir, loads it as a Trimesh triangle mesh,
    and writes a standard glTF binary (GLB) with real faces to output_dir.

    Args:
        data_dir:   Directory containing the iOS scan export (has mesh_hint.ply).
        output_dir: Directory to write the output mesh.glb.

    Returns:
        Path to mesh.glb on success, None on failure.
    """
    mesh_hint_path = data_dir / "mesh_hint.ply"
    if not mesh_hint_path.exists():
        logger.info("No mesh_hint.ply found in %s", data_dir)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mesh.glb"

    try:
        import trimesh

        logger.info("Loading iOS LiDAR mesh from %s", mesh_hint_path)
        mesh = trimesh.load(str(mesh_hint_path), force="mesh")

        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if not geoms:
                logger.warning("mesh_hint.ply loaded as empty Scene")
                return None
            mesh = trimesh.util.concatenate(geoms)

        if not isinstance(mesh, trimesh.Trimesh):
            logger.warning("mesh_hint.ply is not a triangle mesh: %s", type(mesh))
            return None

        num_verts = len(mesh.vertices)
        num_faces = len(mesh.faces)
        logger.info("iOS mesh: %d vertices, %d faces", num_verts, num_faces)

        if num_verts == 0 or num_faces == 0:
            logger.warning("mesh_hint.ply has no geometry (empty mesh)")
            return None

        mesh.export(str(output_path))
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info("Saved room mesh GLB: %s (%.1f MB)", output_path, size_mb)
        return output_path

    except ImportError:
        logger.error("trimesh is required for mesh export. Install with: pip install trimesh")
        return None
    except Exception as e:
        logger.exception("Failed to export iOS mesh hint GLB: %s", e)
        return None


def get_mesh_stats(mesh_hint_path: Path) -> Optional[dict]:
    """
    Return basic statistics about an iOS mesh_hint.ply without exporting.

    Useful for validation and logging.

    Returns:
        Dict with 'vertices', 'faces', 'bounds_min', 'bounds_max', or None on error.
    """
    try:
        import trimesh
        import numpy as np

        mesh = trimesh.load(str(mesh_hint_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if not geoms:
                return None
            mesh = trimesh.util.concatenate(geoms)

        if not isinstance(mesh, trimesh.Trimesh):
            return None

        verts = mesh.vertices
        return {
            "vertices": len(verts),
            "faces": len(mesh.faces),
            "bounds_min": verts.min(axis=0).tolist(),
            "bounds_max": verts.max(axis=0).tolist(),
        }
    except Exception as e:
        logger.warning("Could not read mesh stats from %s: %s", mesh_hint_path, e)
        return None
