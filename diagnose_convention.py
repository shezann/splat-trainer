"""
Diagnostic: project mesh_hint points onto a training image
to verify camera convention (OpenGL vs OpenCV).

Saves two images:
  - diag_with_flip.png: OpenGL→OpenCV flip applied (current fix)
  - diag_no_flip.png: no flip (original code)

If the projected points (green dots) align with the scene geometry
in the image, that convention is correct.
"""
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from plyfile import PlyData

data_dir = Path("data/5ff1f00b/extracted/training_export_77232A63-8C38-427A-9DD1-B0AD689AF84C")

# Load transforms
with open(data_dir / "transforms.json") as f:
    transforms = json.load(f)

# Load mesh hint points
ply = PlyData.read(str(data_dir / "mesh_hint.ply"))
verts = ply["vertex"]
points = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
print(f"Loaded {len(points)} mesh points")
print(f"Points range: x=[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
      f"y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
      f"z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

# Load a training image (use frame 25 for a middle frame)
frame_idx = 25
frame = transforms["frames"][frame_idx]
img_path = data_dir / "images" / Path(frame["file_path"]).name
img = Image.open(str(img_path)).convert("RGB")
orig_w, orig_h = img.size
print(f"Image: {img_path.name}, size={orig_w}x{orig_h}")

# Intrinsics from transforms.json, scaled to actual image size
json_w = transforms.get("w", orig_w)
json_h = transforms.get("h", orig_h)
res_scale_x = orig_w / json_w
res_scale_y = orig_h / json_h
print(f"Resolution scale: {res_scale_x:.3f} x {res_scale_y:.3f} (json {json_w}x{json_h} -> actual {orig_w}x{orig_h})")

fl_x = transforms["fl_x"] * res_scale_x
fl_y = transforms["fl_y"] * res_scale_y
cx = transforms["cx"] * res_scale_x
cy = transforms["cy"] * res_scale_y
print(f"Scaled intrinsics: fx={fl_x:.1f}, fy={fl_y:.1f}, cx={cx:.1f}, cy={cy:.1f}")

K = np.array([
    [fl_x, 0, cx],
    [0, fl_y, cy],
    [0, 0, 1],
], dtype=np.float32)

c2w_raw = np.array(frame["transform_matrix"], dtype=np.float32)

def project_points(c2w, K, points_3d):
    """Project 3D points to 2D using w2c and K."""
    w2c = np.linalg.inv(c2w)
    # Transform to camera space
    pts_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    pts_cam = (w2c @ pts_h.T).T  # (N, 4)

    # Only keep points in front of camera (z > 0.01)
    mask = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[mask]

    # Project
    pts_2d = (K @ pts_cam[:, :3].T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

    # Filter to image bounds
    in_bounds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < orig_w) & \
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < orig_h)

    return pts_2d[in_bounds], mask.sum(), in_bounds.sum()

def draw_points(img, pts_2d, color="lime"):
    """Draw projected points on image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x, y in pts_2d:
        r = 2
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    return out

# Test 1: WITH OpenGL→OpenCV flip
c2w_flipped = c2w_raw.copy()
c2w_flipped[:3, 1:3] *= -1
pts_flip, n_front_flip, n_vis_flip = project_points(c2w_flipped, K, points)
print(f"\nWITH flip: {n_front_flip} points in front of camera, {n_vis_flip} visible in image")
img_flip = draw_points(img, pts_flip, "lime")
img_flip.save("diag_with_flip.png")

# Test 2: WITHOUT flip (original)
pts_noflip, n_front_noflip, n_vis_noflip = project_points(c2w_raw, K, points)
print(f"WITHOUT flip: {n_front_noflip} points in front of camera, {n_vis_noflip} visible in image")
img_noflip = draw_points(img, pts_noflip, "red")
img_noflip.save("diag_no_flip.png")

# Test 3: flip rows instead of columns (alternative convention)
c2w_rowflip = c2w_raw.copy()
c2w_rowflip[1:3, :] *= -1
pts_rowflip, n_front_rowflip, n_vis_rowflip = project_points(c2w_rowflip, K, points)
print(f"ROW flip: {n_front_rowflip} points in front of camera, {n_vis_rowflip} visible in image")
img_rowflip = draw_points(img, pts_rowflip, "cyan")
img_rowflip.save("diag_row_flip.png")

print("\nSaved: diag_with_flip.png, diag_no_flip.png, diag_row_flip.png")
print("Check which one has green/red/cyan dots aligned with scene geometry.")
