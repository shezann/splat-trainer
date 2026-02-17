"""
COLMAP binary file readers for loading sparse reconstruction data.
Reads cameras.bin, images.bin, and points3D.bin files.

Adapted from the reference COLMAP loader (GRAPHDECO/Inria).
"""

import collections
import logging
import struct
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Named tuples for COLMAP data structures
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {cm.model_id: cm for cm in CAMERA_MODELS}


class ColmapImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (WXYZ) to 3x3 rotation matrix."""
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ])


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    """Parse cameras.bin → {camera_id: Camera}."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = _read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    logger.info(f"Read {len(cameras)} cameras from {path}")
    return cameras


def read_images_binary(path: Path) -> Dict[int, ColmapImage]:
    """Parse images.bin → {image_id: Image} with qvec, tvec, name."""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = _read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]

            # Read null-terminated image name
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]

            num_points2D = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = _read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            ) if num_points2D > 0 else np.zeros((0, 2))
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    logger.info(f"Read {len(images)} images from {path}")
    return images


def read_points3D_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse points3D.bin → (xyz, rgb) numpy arrays."""
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))

        for p_id in range(num_points):
            binary_point_line_properties = _read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            xyzs[p_id] = binary_point_line_properties[1:4]
            rgbs[p_id] = binary_point_line_properties[4:7]
            # Skip track data
            track_length = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            _read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )

    logger.info(f"Read {num_points} 3D points from {path}")
    return xyzs.astype(np.float32), rgbs.astype(np.float32)


def get_intrinsics(
    camera: Camera, scale: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Extract fx, fy, cx, cy from a COLMAP camera, applying scale.

    Handles:
    - SIMPLE_PINHOLE: params = [f, cx, cy] → fx=fy=f
    - PINHOLE: params = [fx, fy, cx, cy]
    - SIMPLE_RADIAL: params = [f, cx, cy, k] → fx=fy=f, ignore k
    - RADIAL: params = [f, cx, cy, k1, k2] → fx=fy=f, ignore distortion
    - OPENCV: params = [fx, fy, cx, cy, k1, k2, p1, p2] → ignore distortion
    """
    params = camera.params
    model = camera.model

    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model == "PINHOLE":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        # Ignore distortion param k
    elif model in ("RADIAL", "RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        # Ignore distortion params k1, k2
    elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        # Ignore distortion params
    elif model == "FOV":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        # Ignore FOV param
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return fx * scale, fy * scale, cx * scale, cy * scale
