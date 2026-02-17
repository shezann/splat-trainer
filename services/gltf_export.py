"""
glTF/GLB exporter with KHR_gaussian_splatting extension.

Constructs GLB files manually because pygltflib cannot handle colon-namespaced
attribute names like "KHR_gaussian_splatting:ROTATION".

GLB format: 12-byte header + JSON chunk + BIN chunk.

Data transformations during export:
    Quaternion: WXYZ (gsplat) -> XYZW (glTF spec)
    Opacity:    logit space   -> [0,1] via sigmoid
    Scale:      log-space     -> log-space (KHR spec)
    SH DC:      raw coeffs    -> raw SH coefficients
    SH rest:    raw coeffs    -> raw coeffs (no conversion)
"""

import json
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# glTF constants
GLTF_FLOAT = 5126        # GL_FLOAT
GLTF_POINTS = 0          # Primitive mode
GLTF_ARRAY_BUFFER = 34962
GLB_MAGIC = 0x46546C67   # "glTF" in little-endian
GLB_VERSION = 2
GLB_JSON_CHUNK = 0x4E4F534A   # "JSON"
GLB_BIN_CHUNK = 0x004E4942    # "BIN\0"


def _iter_sh_degree_ranges(num_coeffs: int):
    """
    Yield (start_idx, count, degree) ranges for SH rest coefficients.

    Coefficient counts by degree are: 3, 5, 7, 9, ...
    """
    start_idx = 0
    degree = 1
    while start_idx < num_coeffs:
        count = 2 * degree + 1
        end_idx = start_idx + count
        if end_idx > num_coeffs:
            break
        yield start_idx, count, degree
        start_idx = end_idx
        degree += 1


def save_gltf(path: Path, means, scales, quats, opacities, sh0, sh_rest=None):
    """
    Export Gaussians as GLB with KHR_gaussian_splatting extension.

    Args:
        path: Output .glb file path
        means: (N, 3) positions (numpy or torch tensor)
        scales: (N, 3) log-scales
        quats: (N, 4) quaternions in WXYZ order
        opacities: (N,) logit-space opacities
        sh0: (N, 3) SH degree-0 coefficients
        sh_rest: (N, K, 3) higher-order SH coefficients, or None
    """
    # Convert tensors to numpy if needed
    means = _to_numpy(means)
    scales = _to_numpy(scales)
    quats = _to_numpy(quats)
    opacities = _to_numpy(opacities)
    sh0 = _to_numpy(sh0)
    if sh_rest is not None:
        sh_rest = _to_numpy(sh_rest)

    num_gaussians = len(means)
    if num_gaussians == 0:
        raise ValueError("Cannot export empty Gaussian set")

    logger.info(f"Exporting {num_gaussians} Gaussians to GLB: {path}")

    # --- Data transformations ---

    # Quaternion: WXYZ -> XYZW
    quats_xyzw = quats[:, [1, 2, 3, 0]].astype(np.float32)
    # Normalize safely
    norms = np.linalg.norm(quats_xyzw, axis=1, keepdims=True)
    quats_xyzw = quats_xyzw / np.maximum(norms, 1e-8)

    # Opacity: logit -> sigmoid
    opacities_sigmoid = (
        1.0 / (1.0 + np.exp(-np.clip(opacities, -80.0, 80.0)))
    ).astype(np.float32)

    # Scale: KHR_gaussian_splatting stores scale values in log-space.
    scales_log = scales.astype(np.float32)

    # SH degree 0 coefficients are already in raw coefficient space.
    C0 = 0.28209479177387814
    sh_degree0 = sh0.astype(np.float32)
    color_rgb = np.clip(sh_degree0 * C0 + 0.5, 0.0, 1.0).astype(np.float32)
    color0 = np.concatenate(
        [color_rgb, opacities_sigmoid.reshape(-1, 1)],
        axis=1,
    )

    # Positions
    positions = means.astype(np.float32)

    # --- Build binary buffer ---
    # Each attribute gets its own bufferView
    buffer_data = bytearray()
    buffer_views = []
    accessors = []
    attributes = {}

    def add_attribute(name, data, accessor_type, component_type=GLTF_FLOAT):
        """Add a data array as a buffer view + accessor + attribute."""
        data_bytes = data.tobytes()

        # Pad to 4-byte alignment
        offset = len(buffer_data)
        buffer_data.extend(data_bytes)
        padding = (4 - len(buffer_data) % 4) % 4
        buffer_data.extend(b"\x00" * padding)

        bv_index = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data_bytes),
            "target": GLTF_ARRAY_BUFFER,
        })

        acc_index = len(accessors)
        accessor = {
            "bufferView": bv_index,
            "componentType": component_type,
            "count": num_gaussians,
            "type": accessor_type,
        }

        # Add min/max for POSITION (required by glTF spec)
        if name == "POSITION":
            accessor["min"] = data.min(axis=0).tolist()
            accessor["max"] = data.max(axis=0).tolist()

        accessors.append(accessor)
        attributes[name] = acc_index

    # Core attributes
    add_attribute("POSITION", positions, "VEC3")
    add_attribute("COLOR_0", color0, "VEC4")
    add_attribute("KHR_gaussian_splatting:ROTATION", quats_xyzw, "VEC4")
    add_attribute("KHR_gaussian_splatting:SCALE", scales_log, "VEC3")
    add_attribute(
        "KHR_gaussian_splatting:OPACITY",
        opacities_sigmoid.reshape(-1, 1),
        "SCALAR",
    )
    add_attribute("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", sh_degree0, "VEC3")

    # Higher-order SH coefficients
    if sh_rest is not None and sh_rest.shape[1] > 0:
        sh_rest_np = sh_rest.astype(np.float32)
        for start_idx, count, degree in _iter_sh_degree_ranges(sh_rest_np.shape[1]):
            for coef_idx in range(count):
                sh_idx = start_idx + coef_idx
                attr_name = (
                    f"KHR_gaussian_splatting:SH_DEGREE_{degree}_COEF_{coef_idx}"
                )
                add_attribute(attr_name, sh_rest_np[:, sh_idx, :], "VEC3")

    # --- Build glTF JSON ---
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "splat-trainer",
        },
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "extensionsRequired": ["KHR_gaussian_splatting"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "mode": GLTF_POINTS,
                        "attributes": attributes,
                        "extensions": {
                            "KHR_gaussian_splatting": {
                                "kernel": "ellipse",
                                "colorSpace": "srgb_rec709_display",
                                "sortingMethod": "cameraDistance",
                                "projection": "perspective",
                            }
                        },
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    # --- Assemble GLB ---
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")

    # Pad JSON to 4-byte alignment with spaces
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_padding

    # Pad binary to 4-byte alignment with zeros
    bin_padding = (4 - len(buffer_data) % 4) % 4
    buffer_data.extend(b"\x00" * bin_padding)

    # GLB header: magic + version + total length
    json_chunk_length = len(json_bytes)
    bin_chunk_length = len(buffer_data)
    total_length = (
        12  # GLB header
        + 8 + json_chunk_length  # JSON chunk header + data
        + 8 + bin_chunk_length  # BIN chunk header + data
    )

    with open(path, "wb") as f:
        # GLB header
        f.write(struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_length))

        # JSON chunk
        f.write(struct.pack("<II", json_chunk_length, GLB_JSON_CHUNK))
        f.write(json_bytes)

        # BIN chunk
        f.write(struct.pack("<II", bin_chunk_length, GLB_BIN_CHUNK))
        f.write(buffer_data)

    file_size = path.stat().st_size
    logger.info(
        f"Saved GLB: {path} ({num_gaussians} Gaussians, "
        f"{file_size / 1024 / 1024:.1f} MB)"
    )


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor or numpy array to numpy."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
