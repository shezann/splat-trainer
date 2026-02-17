"""
Standalone CLI training script for 3D Gaussian Splatting.

Bypasses FastAPI server and drives GaussianSplatTrainer directly.

Usage:
    python train.py --data I:/Deco/truck --output I:/Deco/truck_output
    python train.py --data I:/Deco/truck --output I:/Deco/truck_output --quality fast --format glb
    python train.py --data I:/Deco/truck --output I:/Deco/truck_output --iterations 1000 --image-scale 0.5
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.training import GaussianSplatTrainer, TrainingProgress


def main():
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splatting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (COLMAP or NeRF format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output files",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (overrides quality preset)",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="balanced",
        help="Quality preset (default: balanced)",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=0.5,
        help="Image scale factor (default: 0.5 = half resolution)",
    )
    parser.add_argument(
        "--format",
        choices=["ply", "glb", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    # Determine iterations from quality preset if not specified
    from config import QUALITY_PRESETS
    preset = QUALITY_PRESETS.get(args.quality, QUALITY_PRESETS["balanced"])
    iterations = args.iterations if args.iterations is not None else preset["iterations"]

    print(f"Training Configuration:")
    print(f"  Data:       {args.data}")
    print(f"  Output:     {args.output}")
    print(f"  Quality:    {args.quality}")
    print(f"  Iterations: {iterations}")
    print(f"  Image scale: {args.image_scale}")
    print(f"  Format:     {args.format}")
    print()

    # Progress callback
    start_time = time.time()

    def on_progress(progress: TrainingProgress):
        pct = progress.iteration / progress.total_iterations * 100
        elapsed = time.time() - start_time
        rate = progress.iteration / elapsed if elapsed > 0 else 0
        eta = (progress.total_iterations - progress.iteration) / rate if rate > 0 else 0

        print(
            f"  [{pct:5.1f}%] Step {progress.iteration}/{progress.total_iterations} | "
            f"Loss: {progress.loss:.4f} | PSNR: {progress.psnr:.2f}dB | "
            f"Gaussians: {progress.num_gaussians:,} | "
            f"ETA: {eta:.0f}s"
        )

    # Patch image_scale into camera_utils by monkeypatching the training call
    # The trainer uses a fixed 0.5 scale internally - we override via environment
    import os
    os.environ["SPLAT_IMAGE_SCALE"] = str(args.image_scale)

    # Run training
    trainer = GaussianSplatTrainer()

    async def run():
        return await trainer.train(
            data_dir=args.data,
            output_dir=args.output,
            iterations=iterations,
            quality_preset=args.quality,
            progress_callback=on_progress,
            output_format=args.format,
        )

    result = asyncio.run(run())

    # Print results
    print()
    if result["success"]:
        print("Training completed successfully!")
        print(f"  Iterations: {result['iterations_completed']}")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Final PSNR: {result['final_psnr']:.2f} dB")
        print(f"  Gaussians:  {result['num_gaussians']:,}")
        if result.get("output_path"):
            print(f"  Output:     {result['output_path']}")
        if result.get("gltf_output_path"):
            print(f"  glTF:       {result['gltf_output_path']}")
    else:
        print(f"Training failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
