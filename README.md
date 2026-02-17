## Mesh Reconstruction (Images -> 3D Model)

This project now includes a mesh reconstruction script:

- `scripts/reconstruct_mesh.py`

### Quick run (truck test data)

From `I:\Deco`:

```bat
run_mesh.bat
```

Outputs:

- `I:\Deco\truck_mesh_output\mesh_sparse_poisson.ply`
- `I:\Deco\truck_mesh_output\mesh.glb`

### Direct CLI

```bat
python I:\Deco\splat-trainer\scripts\reconstruct_mesh.py ^
  --dataset I:\Deco\truck ^
  --output I:\Deco\truck_mesh_output ^
  --max-image-size 1600 ^
  --poisson-depth 11 ^
  --clean
```

### Notes on quality

- If dense MVS is available, the script uses:
  1) undistort images
  2) patch match stereo
  3) stereo fusion
  4) poisson meshing
- If dense MVS is unavailable (e.g. non-CUDA COLMAP build), the script falls back to:
  1) export sparse COLMAP points
  2) estimate normals
  3) screened Poisson reconstruction
- The sparse fallback produces a real mesh GLB, but quality is lower than full dense MVS.
