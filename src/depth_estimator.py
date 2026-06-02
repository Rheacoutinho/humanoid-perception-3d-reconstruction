"""
depth_estimator.py
------------------
Wraps Depth-Anything V2 Small for per-frame depth estimation.

Design decisions:
- Uses the Small variant (80MB) so it runs on CPU at ~15 FPS
- Input size fixed at 518px (DA2's native resolution)
- Outputs are RELATIVE depth maps (0-1 normalised range)
- Scale calibration happens later in cloud_builder.py
- Single model instance loaded once, reused across all frames
"""

import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import pipeline as hf_pipeline
from pathlib import Path
import sys
import os

# Add project root to path so config.py is always findable
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEPTH_MODEL, DEPTH_INPUT_SIZE


class DepthEstimator:
    """
    Wraps Depth-Anything V2 Small.
    Load once, call estimate() on each frame.
    """

    def __init__(self, device: str = None):
        """
        Load the model. Automatically picks CUDA if available, else CPU.

        Parameters
        ----------
        device : "cuda", "cpu", or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device    = device
        self.device_id = 0 if device == "cuda" else -1

        print(f"Loading Depth-Anything V2 Small on {device}...")
        print(f"  Model : {DEPTH_MODEL}")
        print(f"  (~80MB download on first run, then cached)")

        self.pipe = hf_pipeline(
            task    = "depth-estimation",
            model   = DEPTH_MODEL,
            device  = self.device_id,
        )

        # Warm up — run one dummy inference so the first real frame is fast
        dummy = PILImage.fromarray(
            np.zeros((DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, 3), dtype=np.uint8)
        )
        self.pipe(dummy)

        print(f"✓ Depth estimator ready")

    def estimate(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on a single BGR image (OpenCV format).

        Parameters
        ----------
        image_bgr : (H, W, 3) uint8 numpy array in BGR format

        Returns
        -------
        depth : (H, W) float32 numpy array
                Values are RELATIVE depth in range [0, 1] approx.
                Larger value = further from camera.
                Call calibrate_scale() to convert to metric metres.
        """
        h, w = image_bgr.shape[:2]

        # Convert BGR (OpenCV) → RGB (PIL)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Run inference
        result = self.pipe(pil_image)

        # Extract depth array
        depth = np.array(result["depth"], dtype=np.float32)

        # Resize back to original frame size if the model changed it
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

	# ── Normalise to 0–1 range 
        # HuggingFace DA2 pipeline returns 0–255 float32
        # Normalise here so all downstream code always gets 0–1
        d_max = depth.max()
        if d_max > 1.0:
            depth = depth / d_max
        return depth

    def estimate_batch(
        self,
        frame_paths: list,
        output_dir: str,
        show_progress: bool = True,
    ) -> dict:
        """
        Run depth estimation on a list of frame file paths.
        Saves each depth map as a .npy file.
        Skips frames that already have a saved depth map.

        Parameters
        ----------
        frame_paths  : list of paths to PNG frame files
        output_dir   : directory to save .npy depth maps
        show_progress: print progress every 10 frames

        Returns
        -------
        stats : dict with per-frame depth statistics
                {
                  "frame_000.png": {"min": 0.1, "max": 0.9, "mean": 0.45},
                  ...
                }
        """
        os.makedirs(output_dir, exist_ok=True)
        stats = {}

        for i, frame_path in enumerate(frame_paths):
            fname      = os.path.basename(frame_path)
            depth_path = os.path.join(output_dir, f"depth_{i:04d}.npy")

            # Skip if already computed — important for resuming after crash
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                stats[fname] = {
                    "min"  : float(depth.min()),
                    "max"  : float(depth.max()),
                    "mean" : float(depth.mean()),
                    "path" : depth_path,
                }
                continue

            # Load frame and estimate
            img_bgr = cv2.imread(frame_path)
            if img_bgr is None:
                print(f"  WARNING: could not read {frame_path} — skipping")
                continue

            depth = self.estimate(img_bgr)
            np.save(depth_path, depth)

            stats[fname] = {
                "min"  : float(depth.min()),
                "max"  : float(depth.max()),
                "mean" : float(depth.mean()),
                "path" : depth_path,
            }

            if show_progress and ((i + 1) % 10 == 0 or i == 0):
                print(f"  [{i+1:3d}/{len(frame_paths)}] {fname} "
                      f"— depth range: {depth.min():.3f}–{depth.max():.3f}")

        print(f"\n✓ Depth estimation complete: {len(stats)} frames")
        return stats


def compute_global_scale(depth_stats: dict, near_anchor_m: float = 0.5) -> float:
    """
    Compute a single global scale factor converting relative depth to metres.

    IMPORTANT: Depth-Anything V2 via HuggingFace pipeline returns values
    in 0–255 range (uint8 mapped to float32). We normalise to 0–1 first
    before computing the scale factor.

    Strategy:
    - Normalise each depth map to 0–1
    - Take the 5th percentile (closest reliable surface)
    - Assume that corresponds to near_anchor_m in the real world
    - scale = near_anchor_m / median(5th_percentile_values)
    - Final usage: depth_metres = (raw_depth / 255.0) * scale
    """
    near_values = []

    for fname, s in depth_stats.items():
        depth_path = s["path"]
        depth      = np.load(depth_path).astype(np.float32)

        # Normalise to 0–1 regardless of original range
        d_max = depth.max()
        if d_max > 1.0:
            depth = depth / d_max   # normalise by actual max, not fixed 255
                                     # handles both 0-255 and 0-1 outputs

        valid = depth[depth > 0.01]
        if len(valid) > 100:
            near_values.append(float(np.percentile(valid, 5)))

    if not near_values:
        print("WARNING: could not compute scale — using default 1.0")
        return 1.0

    median_near = float(np.median(near_values))

    if median_near < 1e-6:
        print("WARNING: near depth near zero — using default scale 1.0")
        return 1.0

    scale = near_anchor_m / median_near

    print(f"Global depth scale:")
    print(f"  Median near depth (normalised, 5th pct) : {median_near:.4f}")
    print(f"  Assumed near distance                   : {near_anchor_m}m")
    print(f"  Scale factor                            : {scale:.4f}")
    print(f"  Expected max depth                      : "
          f"~{1.0 * scale:.2f}m")

    return scale


	
def renormalise_existing_depths(depths_dir: str) -> None:
    """
    One-time fix: renormalise existing depth .npy files from 0–255 to 0–1.
    Safe to run multiple times — skips files already in 0–1 range.
    """
    import glob
    files = sorted(glob.glob(os.path.join(depths_dir, "depth_*.npy")))
    fixed = 0
    for fpath in files:
        d = np.load(fpath).astype(np.float32)
        if d.max() > 1.0:
            d = d / d.max()
            np.save(fpath, d)
            fixed += 1
    print(f"✓ Renormalised {fixed}/{len(files)} depth files in {depths_dir}")