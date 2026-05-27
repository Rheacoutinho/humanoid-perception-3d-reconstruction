"""
segmentor.py
------------
Wraps FastSAM for CPU-friendly instance segmentation.

Why FastSAM instead of SAM2:
  - FastSAM-s is 23MB vs SAM2's 900MB
  - Runs at ~25ms per frame on CPU vs SAM2's GPU requirement
  - Accuracy is slightly lower but more than sufficient for
    generating regions to embed with CLIP
  - Fully open source, Apache 2.0 license

What this module does:
  - Takes a BGR frame
  - Returns a list of binary masks, one per detected instance
  - Each mask is a (H, W) boolean numpy array
  - Masks are filtered by minimum area to remove noise
  - Masks are deduplicated using IoU to remove near-duplicates
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FASTSAM_CONF, FASTSAM_IOU


class Segmentor:
    """
    FastSAM-based instance segmentor.
    Loads the model once, call segment() on each frame.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Parameters
        ----------
        checkpoint_path : path to FastSAM-s.pt checkpoint file
        device          : "cuda", "cpu", or None (auto-detect)
        """
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"FastSAM checkpoint not found: {checkpoint_path}\n"
                f"Download from: "
                f"https://huggingface.co/spaces/An-619/FastSAM/"
                f"resolve/main/weights/FastSAM-s.pt"
            )

        print(f"Loading FastSAM from: {checkpoint_path}")
        print(f"Device: {device}")

        # FastSAM uses the ultralytics YOLO interface
        from ultralytics import YOLO
        self.model = YOLO(checkpoint_path)

        print(f"✓ Segmentor ready")

    def segment(
        self,
        image_bgr: np.ndarray,
        min_area_fraction: float = 0.002,
        max_area_fraction: float = 0.95,
        iou_dedup_threshold: float = 0.85,
    ) -> list:
        """
        Segment an image into instance masks.

        Parameters
        ----------
        image_bgr           : (H, W, 3) uint8 BGR image
        min_area_fraction   : discard masks smaller than this fraction
                              of total image area (removes tiny noise)
                              0.002 = 0.2% of image = ~220px on 512×218
        max_area_fraction   : discard masks larger than this fraction
                              (removes full-image background masks)
                              0.95 = 95% of image
        iou_dedup_threshold : discard duplicate masks with IoU > this
                              0.85 = masks >85% overlapping are duplicates

        Returns
        -------
        masks : list of (H, W) bool numpy arrays
                One mask per detected instance, filtered and deduplicated
                Empty list if no valid masks found
        """
        H, W    = image_bgr.shape[:2]
        min_px  = int(H * W * min_area_fraction)
        max_px  = int(H * W * max_area_fraction)

        # Run FastSAM
        # verbose=False suppresses YOLO's per-frame console output
        results = self.model(
            image_bgr,
            device  = self.device,
            conf    = FASTSAM_CONF,
            iou     = FASTSAM_IOU,
            verbose = False,
            retina_masks = True,  # higher-resolution masks
        )

        if not results or results[0].masks is None:
            return []

        # Extract masks from results
        raw_masks = results[0].masks.data  # tensor (N, H, W)

        masks_np = []
        for i in range(raw_masks.shape[0]):
            mask = raw_masks[i].cpu().numpy().astype(bool)

            # Resize to original image size if needed
            if mask.shape != (H, W):
                mask_uint8 = mask.astype(np.uint8) * 255
                mask_uint8 = cv2.resize(
                    mask_uint8, (W, H),
                    interpolation=cv2.INTER_NEAREST
                )
                mask = mask_uint8.astype(bool)

            masks_np.append(mask)

        # Filter by area
        filtered = []
        for mask in masks_np:
            area = mask.sum()
            if min_px <= area <= max_px:
                filtered.append(mask)

        # Deduplicate by IoU
        # Remove masks that overlap heavily with a larger mask
        deduplicated = self._dedup_masks(filtered, iou_dedup_threshold)

        return deduplicated

    def _dedup_masks(
        self,
        masks: list,
        iou_threshold: float,
    ) -> list:
        """
        Remove near-duplicate masks using IoU.

        Sort by area descending (keep larger masks preferentially).
        For each mask, check IoU against all already-kept masks.
        Discard if IoU > threshold with any kept mask.
        """
        if not masks:
            return []

        # Sort by area descending
        masks_sorted = sorted(masks, key=lambda m: m.sum(), reverse=True)

        kept = [masks_sorted[0]]

        for candidate in masks_sorted[1:]:
            is_duplicate = False
            for kept_mask in kept:
                intersection = (candidate & kept_mask).sum()
                union        = (candidate | kept_mask).sum()
                iou          = intersection / (union + 1e-8)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

        return kept

    def segment_keyframes(
        self,
        frame_paths: list,
        keyframe_step: int = 5,
        save_dir: str = None,
    ) -> dict:
        """
        Run segmentation on keyframes and optionally save visualisations.

        Parameters
        ----------
        frame_paths   : list of all frame paths
        keyframe_step : process every Nth frame
        save_dir      : if provided, save mask visualisation images here

        Returns
        -------
        results : dict mapping frame_index → list of masks
                  { 0: [mask1, mask2, ...], 5: [...], ... }
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        keyframe_indices = list(range(0, len(frame_paths), keyframe_step))
        results          = {}

        print(f"Segmenting {len(keyframe_indices)} keyframes "
              f"(every {keyframe_step} frames)...")

        for ki, frame_idx in enumerate(keyframe_indices):
            img_bgr = cv2.imread(frame_paths[frame_idx])
            if img_bgr is None:
                continue

            masks = self.segment(img_bgr)
            results[frame_idx] = masks

            if save_dir and ki < 5:
                # Save a visualisation for the first 5 keyframes
                vis = self._visualise_masks(img_bgr, masks)
                vis_path = os.path.join(
                    save_dir, f"masks_frame{frame_idx:04d}.png"
                )
                cv2.imwrite(vis_path, vis)

            if (ki + 1) % 5 == 0 or ki == 0:
                print(f"  Keyframe {ki+1}/{len(keyframe_indices)} "
                      f"(frame {frame_idx}): {len(masks)} masks")

        total_masks = sum(len(v) for v in results.values())
        print(f"\n✓ Segmentation complete")
        print(f"  Keyframes processed : {len(results)}")
        print(f"  Total masks         : {total_masks}")
        print(f"  Avg masks/frame     : "
              f"{total_masks/max(len(results),1):.1f}")

        return results

    def _visualise_masks(
        self,
        image_bgr: np.ndarray,
        masks: list,
    ) -> np.ndarray:
        """
        Draw coloured masks overlaid on the original image.
        Used for debugging and README screenshots.
        """
        vis    = image_bgr.copy().astype(np.float32)
        np.random.seed(42)

        for mask in masks:
            colour = np.random.randint(0, 255, 3).astype(np.float32)
            for c in range(3):
                vis[:, :, c] = np.where(
                    mask,
                    vis[:, :, c] * 0.5 + colour[c] * 0.5,
                    vis[:, :, c]
                )

        return vis.astype(np.uint8)