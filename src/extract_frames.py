"""
extract_frames.py
-----------------
Takes a video file and extracts a clean set of frames for 3D reconstruction.

Steps:
  1. Open the video and read basic metadata (fps, total frames, duration)
  2. Sample N candidate frames evenly across the video
  3. Score each candidate frame for sharpness (blur detection)
  4. Discard frames below the sharpness threshold
  5. Resize kept frames to the target resolution
  6. Save to output folder as PNG files
  7. Save a metadata JSON so later stages know the frame list
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path


def compute_sharpness(frame_bgr: np.ndarray) -> float:
    """
    Compute a sharpness score for a frame using the Laplacian variance method.
    
    How it works:
    - Convert frame to greyscale
    - Apply the Laplacian operator (a second-order edge detector)
    - Compute the variance of the result
    - Sharp images have lots of edges → high variance → high score
    - Blurry images have few edges → low variance → low score
    
    Returns a float. Typical values:
        < 50  → very blurry
        50–150 → acceptable
        > 150 → sharp
    """
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    score = laplacian.var()
    return float(score)


def resize_frame(frame_bgr: np.ndarray, max_side: int = 512) -> np.ndarray:
    """
    Resize a frame so its longest side equals max_side pixels.
    Preserves the aspect ratio exactly.
    
    MASt3R expects images no larger than 512px on the long side.
    Larger images slow it down with no quality benefit.
    """
    h, w = frame_bgr.shape[:2]
    
    if max(h, w) <= max_side:
        # Already small enough, return as-is
        return frame_bgr
    
    if w >= h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 80,
    max_side: int = 512,
    sharpness_threshold: float = 50.0,
    min_frames: int = 30,
) -> dict:
    """
    Main function. Extracts clean frames from a video file.

    Parameters
    ----------
    video_path        : path to the input .mp4 (or any video file)
    output_dir        : folder where frames will be saved as PNG
    num_frames        : how many candidate frames to sample from the video
    max_side          : resize so longest side = this many pixels
    sharpness_threshold : frames below this score are discarded as blurry
    min_frames        : if too many frames are discarded, lower threshold
                        automatically until we have at least this many

    Returns
    -------
    A dict with metadata about the extraction:
        {
          "video_path": ...,
          "total_video_frames": ...,
          "fps": ...,
          "duration_seconds": ...,
          "frames_sampled": ...,
          "frames_kept": ...,
          "sharpness_threshold_used": ...,
          "output_dir": ...,
          "frame_files": [list of saved filenames],
          "frame_sharpness": {filename: score}
        }
    """

    # ------------------------------------------------------------------ #
    # 1. Open video and read metadata
    # ------------------------------------------------------------------ #
    video_path = str(video_path)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video loaded: {os.path.basename(video_path)}")
    print(f"  Total frames : {total_frames}")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Duration     : {duration:.1f} seconds")

    # ------------------------------------------------------------------ #
    # 2. Decide which frame indices to sample
    # ------------------------------------------------------------------ #
    # We spread num_frames evenly across the full video.
    # Example: 1000 total frames, num_frames=80
    #   → sample at indices [0, 12, 25, 37, 50, ...]
    
    actual_sample = min(num_frames, total_frames)
    sample_indices = np.linspace(0, total_frames - 1, actual_sample, dtype=int)
    sample_indices = list(dict.fromkeys(sample_indices.tolist()))  # deduplicate
    
    print(f"\nSampling {len(sample_indices)} candidate frames...")

    # ------------------------------------------------------------------ #
    # 3. Read and score each candidate frame
    # ------------------------------------------------------------------ #
    candidates = []  # list of (frame_index, sharpness_score, frame_bgr)
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            # Couldn't read this frame — skip
            continue
        
        score = compute_sharpness(frame)
        candidates.append((idx, score, frame))
    
    cap.release()
    print(f"  Successfully read {len(candidates)} frames")

    # ------------------------------------------------------------------ #
    # 4. Filter blurry frames, with automatic threshold lowering
    # ------------------------------------------------------------------ #
    # Start with the requested threshold.
    # If too many frames get discarded, lower the threshold and retry.
    # This handles videos that are overall soft/slightly blurry.
    
    threshold = sharpness_threshold
    
    while True:
        kept = [(idx, score, frame) for (idx, score, frame) in candidates
                if score >= threshold]
        
        if len(kept) >= min_frames:
            break
        
        if threshold <= 5.0:
            # We've lowered the bar as far as we'll go — keep everything
            kept = candidates
            break
        
        old_threshold = threshold
        threshold = threshold * 0.7  # lower by 30% and retry
        print(f"  Only {len(kept)} frames above threshold {old_threshold:.1f} "
              f"— retrying with threshold {threshold:.1f}")
    
    print(f"  Kept {len(kept)} frames after sharpness filter "
          f"(threshold used: {threshold:.1f})")

    # ------------------------------------------------------------------ #
    # 5 & 6. Resize and save kept frames
    # ------------------------------------------------------------------ #
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = []
    frame_sharpness = {}
    
    for i, (frame_idx, score, frame_bgr) in enumerate(kept):
        resized = resize_frame(frame_bgr, max_side=max_side)
        
        # Name files with zero-padded index so they sort correctly
        filename = f"frame_{i:04d}.png"
        save_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(save_path, resized)
        frame_files.append(filename)
        frame_sharpness[filename] = round(score, 2)
    
    print(f"\nSaved {len(frame_files)} frames to: {output_dir}")

    # ------------------------------------------------------------------ #
    # 7. Save metadata JSON
    # ------------------------------------------------------------------ #
    metadata = {
        "video_path": video_path,
        "total_video_frames": total_frames,
        "fps": round(fps, 3),
        "duration_seconds": round(duration, 2),
        "frames_sampled": len(sample_indices),
        "frames_kept": len(frame_files),
        "sharpness_threshold_used": round(threshold, 2),
        "max_side_pixels": max_side,
        "output_dir": output_dir,
        "frame_files": frame_files,
        "frame_sharpness": frame_sharpness,
    }
    
    metadata_path = os.path.join(output_dir, "frames_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Print a small sharpness summary
    scores = list(frame_sharpness.values())
    print(f"\nSharpness summary:")
    print(f"  Min  : {min(scores):.1f}")
    print(f"  Max  : {max(scores):.1f}")
    print(f"  Mean : {np.mean(scores):.1f}")

    return metadata


# ------------------------------------------------------------------ #
# Command-line interface
# Lets you run this directly: python src/extract_frames.py myvideo.mp4
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract clean frames from a video for 3D reconstruction"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file (e.g. my_room.mp4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="frames",
        help="Directory to save extracted frames (default: frames/)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=80,
        help="Number of candidate frames to sample from the video (default: 80)"
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=512,
        help="Resize so longest side = this many pixels (default: 512)"
    )
    parser.add_argument(
        "--sharpness_threshold",
        type=float,
        default=50.0,
        help="Minimum sharpness score to keep a frame (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    result = extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        max_side=args.max_side,
        sharpness_threshold=args.sharpness_threshold,
    )
    
    print(f"\nDone. {result['frames_kept']} frames ready for reconstruction.")