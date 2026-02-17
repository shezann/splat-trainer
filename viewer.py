"""
Interactive 3D Gaussian Splatting Viewer.

Loads a trained PLY and lets you orbit around the scene with mouse controls.

Controls:
    Left-click + drag:  Orbit (rotate around scene)
    Right-click + drag: Pan
    Scroll wheel:       Zoom in/out
    R:                  Reset camera
    Q / ESC:            Quit

Usage:
    python viewer.py --ply I:/Deco/truck_output_fixed3/point_cloud_5000.ply --data I:/Deco/truck
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def load_gaussians_from_ply(ply_path, device):
    """Load Gaussian parameters from PLY."""
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n = len(vertex)
    print(f"Loaded {n:,} Gaussians from {ply_path}")

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    sh0 = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)

    sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
    num_sh_rest = len(sh_rest_names)
    sh_rest = None
    if num_sh_rest > 0:
        num_coeffs_per_channel = num_sh_rest // 3
        sh_rest_flat = np.zeros((n, num_sh_rest), dtype=np.float32)
        for i, name in enumerate(sh_rest_names):
            sh_rest_flat[:, i] = np.array(vertex[name], dtype=np.float32)
        sh_rest = np.zeros((n, num_coeffs_per_channel, 3), dtype=np.float32)
        for c in range(3):
            for k in range(num_coeffs_per_channel):
                sh_rest[:, k, c] = sh_rest_flat[:, c * num_coeffs_per_channel + k]

    opacities = np.array(vertex["opacity"], dtype=np.float32)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(np.float32)

    if sh_rest is not None:
        total_sh = 1 + sh_rest.shape[1]
        sh_degree = int(math.sqrt(total_sh)) - 1
    else:
        sh_degree = 0

    return {
        "means": torch.tensor(means, device=device),
        "sh0": torch.tensor(sh0, device=device),
        "sh_rest": torch.tensor(sh_rest, device=device) if sh_rest is not None else None,
        "opacities": torch.tensor(opacities, device=device),
        "scales": torch.tensor(scales, device=device),
        "quats": torch.tensor(quats, device=device),
        "sh_degree": sh_degree,
    }


def render_frame(gaussians, viewmat, K, width, height):
    """Render one frame."""
    from gsplat.rendering import rasterization

    quats = gaussians["quats"]
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

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

    return renders[0].clamp(0, 1)


class OrbitCamera:
    """Simple orbit camera controller."""

    def __init__(self, center, radius, width, height, fx, fy):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.azimuth = 0.0      # horizontal angle in radians
        self.elevation = 0.2    # vertical angle in radians
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.pan_offset = np.zeros(3, dtype=np.float64)

    def get_view_matrix(self):
        """Compute world-to-camera matrix from orbit parameters."""
        # Camera position on sphere
        cos_el = math.cos(self.elevation)
        sin_el = math.sin(self.elevation)
        cos_az = math.cos(self.azimuth)
        sin_az = math.sin(self.azimuth)

        cam_pos = self.center + self.pan_offset + self.radius * np.array([
            cos_el * sin_az,
            -sin_el,
            cos_el * cos_az,
        ])

        target = self.center + self.pan_offset

        # Look-at
        forward = target - cam_pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, -1, 0], dtype=np.float64)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            up = np.array([0, 0, 1], dtype=np.float64)
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
        right = right / right_norm
        actual_up = np.cross(right, forward)

        R = np.stack([right, -actual_up, forward], axis=0)
        t = -R @ cam_pos

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3, 3] = t.astype(np.float32)
        return w2c

    def get_K(self):
        return np.array([
            [self.fx, 0, self.width / 2],
            [0, self.fy, self.height / 2],
            [0, 0, 1],
        ], dtype=np.float32)

    def orbit(self, dx, dy):
        self.azimuth += dx * 0.01
        self.elevation = np.clip(self.elevation + dy * 0.01, -math.pi / 2 + 0.1, math.pi / 2 - 0.1)

    def pan(self, dx, dy):
        # Pan in camera-local right/up directions
        w2c = self.get_view_matrix()
        right = w2c[0, :3]
        up = -w2c[1, :3]
        scale = self.radius * 0.002
        self.pan_offset += (right * dx + up * dy) * scale

    def zoom(self, delta):
        self.radius *= (1.0 - delta * 0.1)
        self.radius = max(0.1, self.radius)

    def reset(self, center, radius):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.azimuth = 0.0
        self.elevation = 0.2
        self.pan_offset = np.zeros(3, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Interactive 3DGS Viewer")
    parser.add_argument("--ply", type=Path, required=True, help="Trained PLY file")
    parser.add_argument("--data", type=Path, required=True, help="Training data dir (for camera info)")
    parser.add_argument("--width", type=int, default=960, help="Window width")
    parser.add_argument("--height", type=int, default=540, help="Window height")
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("pygame is required for the viewer. Install with: pip install pygame")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Gaussians
    gaussians = load_gaussians_from_ply(args.ply, device)

    # Get scene info from training data
    from services.camera_utils import load_training_data
    os.environ["SPLAT_IMAGE_SCALE"] = "1.0"
    data = load_training_data(args.data, device=device, image_scale=1.0)

    # Compute scene center and radius from camera positions
    viewmats = data["viewmats"]
    cam_centers = []
    for i in range(len(viewmats)):
        w2c = viewmats[i].cpu().numpy()
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cam_centers.append(-(R.T @ t))
    cam_centers = np.array(cam_centers)
    scene_center = cam_centers.mean(axis=0)
    scene_radius = np.linalg.norm(cam_centers - scene_center, axis=1).mean()

    # Use the intrinsics from the first camera
    K_data = data["Ks"][0].cpu().numpy()
    data_width = data["width"]
    data_height = data["height"]

    # Scale intrinsics to viewer window size
    sx = args.width / data_width
    sy = args.height / data_height
    fx = K_data[0, 0] * sx
    fy = K_data[1, 1] * sy

    # Free training images from GPU memory
    del data

    # Init camera
    camera = OrbitCamera(
        center=scene_center,
        radius=scene_radius,
        width=args.width,
        height=args.height,
        fx=fx,
        fy=fy,
    )

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("3D Gaussian Splatting Viewer - Drag to orbit, Scroll to zoom, R to reset, Q to quit")
    clock = pygame.time.Clock()

    print("\n=== 3DGS Interactive Viewer ===")
    print("  Left-click + drag:  Orbit")
    print("  Right-click + drag: Pan")
    print("  Scroll wheel:       Zoom")
    print("  R:                  Reset camera")
    print("  Q / ESC:            Quit")
    print("================================\n")

    running = True
    dragging = False
    drag_button = 0
    last_mouse = (0, 0)
    need_render = True
    last_frame = None
    frame_count = 0
    fps_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    camera.reset(scene_center, scene_radius)
                    need_render = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):  # Left or right click
                    dragging = True
                    drag_button = event.button
                    last_mouse = event.pos
                elif event.button == 4:  # Scroll up
                    camera.zoom(1)
                    need_render = True
                elif event.button == 5:  # Scroll down
                    camera.zoom(-1)
                    need_render = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                if drag_button == 1:  # Left click = orbit
                    camera.orbit(dx, dy)
                elif drag_button == 3:  # Right click = pan
                    camera.pan(-dx, dy)
                last_mouse = event.pos
                need_render = True

        if need_render:
            w2c = camera.get_view_matrix()
            K = camera.get_K()

            w2c_t = torch.tensor(w2c, device=device, dtype=torch.float32)
            K_t = torch.tensor(K, device=device, dtype=torch.float32)

            with torch.no_grad():
                image = render_frame(gaussians, w2c_t, K_t, args.width, args.height)

            # Convert to pygame surface
            img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            # pygame expects (width, height, 3) but we have (height, width, 3)
            surface = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            need_render = False
            frame_count += 1

            # FPS counter
            now = time.time()
            if now - fps_time >= 2.0:
                fps = frame_count / (now - fps_time)
                pygame.display.set_caption(f"3DGS Viewer | {fps:.1f} FPS | {gaussians['means'].shape[0]:,} Gaussians")
                frame_count = 0
                fps_time = now

        clock.tick(60)

    pygame.quit()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
