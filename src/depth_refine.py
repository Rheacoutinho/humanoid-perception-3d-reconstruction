"""
depth_refine.py
---------------
Takes the MASt3R point cloud + camera poses and enriches it using
Depth-Anything V2 per-frame depth predictions.

Steps:
  1. Load camera poses and intrinsics from poses.json
  2. For each frame, run Depth-Anything V2 to get a dense depth map
  3. Scale-align the predicted depth to match MASt3R metric scale
     (least-squares fit between predicted depth and MASt3R depth)
  4. Back-project every pixel into 3D world space using the pose
  5. Colour each point from the original image
  6. Fuse all frames into one combined point cloud
  7. Merge with the original MASt3R cloud and remove outliers
  8. Save the fused point cloud
"""

import os
import sys
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path


def load_poses_and_intrinsics(poses_path: str):
    """
    Load camera poses and intrinsics saved by reconstruct.py.

    Returns
    -------
    poses      : list of (4,4) numpy arrays — cam-to-world transforms
    intrinsics : list of (3,3) numpy arrays — camera K matrices
    frame_files: list of frame filenames in order
    depth_maps : list of paths to MASt3R depth .npy files
    """
    with open(poses_path, "r") as f:
        data = json.load(f)

    poses = []
    intrinsics = []
    frame_files = []
    depth_maps = []

    for i in range(data["num_frames"]):
        pose = np.array(data["poses"][i]["cam_to_world"])   # (4,4)
        K    = np.array(data["intrinsics"][i]["K"])         # (3,3)
        poses.append(pose)
        intrinsics.append(K)
        frame_files.append(data["poses"][i]["frame_file"])
        depth_maps.append(data["depth_maps"][i])

    print(f"Loaded {len(poses)} poses and intrinsics")
    return poses, intrinsics, frame_files, depth_maps


def run_depth_anything(image_bgr: np.ndarray, depth_pipe) -> np.ndarray:
    """
    Run Depth-Anything V2 on a single BGR image.

    Parameters
    ----------
    image_bgr : (H, W, 3) uint8 numpy array in BGR format (OpenCV default)
    depth_pipe: HuggingFace pipeline object loaded once at startup

    Returns
    -------
    depth_pred : (H, W) float32 numpy array
                 Values are RELATIVE depth (larger = further away)
                 Must be scale-aligned before use in 3D
    """
    from PIL import Image as PILImage

    # Convert BGR → RGB for the pipeline
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(image_rgb)

    # Run inference
    result = depth_pipe(pil_image)

    # Extract depth array and resize to match input image size
    depth_pred = np.array(result["depth"], dtype=np.float32)

    # Resize to exactly match the input image dimensions
    h, w = image_bgr.shape[:2]
    if depth_pred.shape != (h, w):
        depth_pred = cv2.resize(depth_pred, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth_pred


def scale_align_depth(
    depth_pred: np.ndarray,
    depth_mast3r: np.ndarray,
) -> tuple:
    """
    Scale-align Depth-Anything's relative depth to MASt3R's metric depth.

    Why we need this:
        Depth-Anything outputs values in an arbitrary range (e.g. 0.1–0.9)
        MASt3R outputs real-world metric depth in metres (e.g. 0.5–4.0)
        We find the scale factor s and shift b such that:
            depth_aligned = s * depth_pred + b
        matches the MASt3R depth as closely as possible.

    We use least-squares regression: fits s and b to minimise
        sum((s * depth_pred + b - depth_mast3r)^2)
    over all valid pixels.

    Parameters
    ----------
    depth_pred    : (H, W) Depth-Anything output (relative, arbitrary scale)
    depth_mast3r  : (H, W) MASt3R depth (metric, in metres)

    Returns
    -------
    depth_aligned : (H, W) Depth-Anything depth rescaled to metric
    scale         : the fitted scale factor s
    shift         : the fitted shift b
    """
    # Only use pixels where MASt3R has valid (positive) depth
    valid = (depth_mast3r > 0.01) & np.isfinite(depth_mast3r) & np.isfinite(depth_pred)

    if valid.sum() < 100:
        # Not enough valid pixels to fit — return pred as-is
        return depth_pred, 1.0, 0.0

    y = depth_mast3r[valid].flatten()   # target: MASt3R metric depth
    x = depth_pred[valid].flatten()     # source: Depth-Anything relative depth

    # Least-squares: solve [x, 1] @ [s, b]^T = y
    A = np.stack([x, np.ones_like(x)], axis=1)  # (N, 2)
    result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    scale, shift = result[0], result[1]

    depth_aligned = scale * depth_pred + shift

    # Clip to positive values — negative depth is physically impossible
    depth_aligned = np.clip(depth_aligned, 0.01, None)

    return depth_aligned, float(scale), float(shift)


def backproject_depth_to_points(
    depth_map: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    image_rgb: np.ndarray,
    max_depth: float = 10.0,
    min_depth: float = 0.05,
) -> tuple:
    """
    Back-project a depth map into 3D world-space points.

    How back-projection works:
        For each pixel (u, v) with depth d:
        1. Convert pixel to normalised camera ray:
               x_cam = (u - cx) / fx
               y_cam = (v - cy) / fy
        2. Scale by depth to get 3D point in camera space:
               P_cam = [x_cam * d, y_cam * d, d]
        3. Apply cam-to-world transform to get world space:
               P_world = cam_to_world @ [P_cam, 1]

    Parameters
    ----------
    depth_map    : (H, W) metric depth in metres
    K            : (3, 3) camera intrinsic matrix
    cam_to_world : (4, 4) camera pose matrix
    image_rgb    : (H, W, 3) uint8 RGB image for colours
    max_depth    : discard points further than this (metres)
    min_depth    : discard points closer than this (metres)

    Returns
    -------
    points_world : (N, 3) float64 XYZ coordinates
    colours      : (N, 3) uint8 RGB colours
    """
    h, w = depth_map.shape

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Build pixel coordinate grids
    # u_grid[v, u] = u (column index)
    # v_grid[v, u] = v (row index)
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))  # both (H, W)

    # Flatten everything for vectorised computation
    u_flat = u_grid.flatten().astype(np.float64)   # (H*W,)
    v_flat = v_grid.flatten().astype(np.float64)   # (H*W,)
    d_flat = depth_map.flatten().astype(np.float64) # (H*W,)

    # Validity mask — only keep pixels with sensible depth
    valid = (d_flat >= min_depth) & (d_flat <= max_depth) & np.isfinite(d_flat)

    u_flat = u_flat[valid]
    v_flat = v_flat[valid]
    d_flat = d_flat[valid]

    # Back-project to camera space
    x_cam = (u_flat - cx) / fx * d_flat   # (N,)
    y_cam = (v_flat - cy) / fy * d_flat   # (N,)
    z_cam = d_flat                          # (N,)

    # Stack into homogeneous coordinates (4, N)
    ones       = np.ones_like(z_cam)
    points_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=0)  # (4, N)

    # Transform to world space
    points_world_h = cam_to_world @ points_cam    # (4, N)
    points_world   = points_world_h[:3, :].T      # (N, 3)

    # Get colours for valid pixels
    colours_flat = image_rgb.reshape(-1, 3)        # (H*W, 3)
    colours      = colours_flat[valid]             # (N, 3)

    return points_world, colours


def fuse_depth_maps(
    frames_dir: str,
    poses_path: str,
    output_dir: str,
    mast3r_ply_path: str,
    depth_pipe,
    max_depth: float = 8.0,
    min_depth: float = 0.05,
    voxel_size: float = 0.02,
) -> dict:
    """
    Main function — fuses Depth-Anything depth maps with MASt3R point cloud.

    Parameters
    ----------
    frames_dir      : folder containing frame PNG files
    poses_path      : path to poses.json from MASt3R
    output_dir      : where to save outputs
    mast3r_ply_path : path to the raw MASt3R point cloud
    depth_pipe      : loaded Depth-Anything pipeline
    max_depth       : maximum depth to keep (metres)
    min_depth       : minimum depth to keep (metres)
    voxel_size      : voxel grid size for downsampling (metres)
                      smaller = denser but slower. 0.02 = 2cm resolution.

    Returns
    -------
    dict with output paths and point counts
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load poses and intrinsics
    # ------------------------------------------------------------------ #
    poses, intrinsics, frame_files, mast3r_depth_paths = load_poses_and_intrinsics(
        poses_path
    )
    num_frames = len(poses)

    # ------------------------------------------------------------------ #
    # 2. Process each frame
    # ------------------------------------------------------------------ #
    all_points  = []
    all_colours = []
    scale_shifts = []

    print(f"\nProcessing {num_frames} frames with Depth-Anything V2...")
    print("(Each frame takes ~1–2 seconds on T4)\n")

    for i in range(num_frames):
        frame_path = os.path.join(frames_dir, frame_files[i])

        # Load the frame image
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            print(f"  Frame {i}: could not read {frame_path} — skipping")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_bgr.shape[:2]

        # Run Depth-Anything V2
        depth_pred = run_depth_anything(img_bgr, depth_pipe)  # relative depth

        # Load MASt3R depth for scale alignment
        mast3r_depth = np.load(mast3r_depth_paths[i])  # metric depth

        # Resize MASt3R depth to match frame size if needed
        if mast3r_depth.shape != (h, w):
            mast3r_depth = cv2.resize(
                mast3r_depth, (w, h), interpolation=cv2.INTER_LINEAR
            )

        # Scale-align Depth-Anything to metric scale
        depth_aligned, scale, shift = scale_align_depth(depth_pred, mast3r_depth)
        scale_shifts.append({"frame": i, "scale": scale, "shift": shift})

        # Back-project to 3D world points
        points_world, colours = backproject_depth_to_points(
            depth_map    = depth_aligned,
            K            = intrinsics[i],
            cam_to_world = poses[i],
            image_rgb    = img_rgb,
            max_depth    = max_depth,
            min_depth    = min_depth,
        )

        all_points.append(points_world)
        all_colours.append(colours)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Frame {i+1:2d}/{num_frames} — "
                  f"{len(points_world):,} pts  "
                  f"scale={scale:.3f}  shift={shift:.3f}")

    # ------------------------------------------------------------------ #
    # 3. Stack all depth-refined points
    # ------------------------------------------------------------------ #
    depth_points  = np.concatenate(all_points,  axis=0)  # (Total, 3)
    depth_colours = np.concatenate(all_colours, axis=0)  # (Total, 3)
    print(f"\nDepth-Anything points total: {len(depth_points):,}")

    # ------------------------------------------------------------------ #
    # 4. Load original MASt3R cloud and merge
    # ------------------------------------------------------------------ #
    print("Merging with MASt3R point cloud...")
    pcd_mast3r = o3d.io.read_point_cloud(mast3r_ply_path)
    mast3r_pts = np.asarray(pcd_mast3r.points)
    mast3r_col = (np.asarray(pcd_mast3r.colors) * 255).astype(np.uint8)

    # Combine: MASt3R points first (higher quality), depth points second
    combined_points  = np.concatenate([mast3r_pts, depth_points],  axis=0)
    combined_colours = np.concatenate([mast3r_col, depth_colours], axis=0)

    print(f"Combined total: {len(combined_points):,} points")

    # ------------------------------------------------------------------ #
    # 5. Build Open3D cloud, downsample, remove outliers
    # ------------------------------------------------------------------ #
    pcd_fused = o3d.geometry.PointCloud()
    pcd_fused.points = o3d.utility.Vector3dVector(combined_points.astype(np.float64))
    pcd_fused.colors = o3d.utility.Vector3dVector(
        combined_colours.astype(np.float64) / 255.0
    )

    # Voxel downsample — merges nearby duplicate points into one
    # This is important because back-projected depth maps overlap heavily
    print(f"Voxel downsampling (voxel size = {voxel_size}m)...")
    pcd_down = pcd_fused.voxel_down_sample(voxel_size=voxel_size)
    print(f"After downsampling: {len(pcd_down.points):,} points")

    # Remove statistical outliers — cleans noise at scene edges
    print("Removing outliers...")
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    print(f"After outlier removal: {len(pcd_clean.points):,} points")

    # ------------------------------------------------------------------ #
    # 6. Save outputs
    # ------------------------------------------------------------------ #
    fused_ply_path = os.path.join(output_dir, "pointcloud_fused.ply")
    o3d.io.write_point_cloud(fused_ply_path, pcd_clean)
    print(f"\n✓ Fused point cloud saved: {fused_ply_path}")
    print(f"  File size: {os.path.getsize(fused_ply_path)/1e6:.1f} MB")

    # Save scale/shift values for debugging
    scale_path = os.path.join(output_dir, "depth_scale_shifts.json")
    with open(scale_path, "w") as f:
        json.dump(scale_shifts, f, indent=2)

    result = {
        "fused_ply_path":    fused_ply_path,
        "num_points_fused":  len(pcd_clean.points),
        "num_points_mast3r": len(pcd_mast3r.points),
        "num_frames":        num_frames,
        "voxel_size":        voxel_size,
    }

    print("\n=== Depth Fusion Summary ===")
    for k, v in result.items():
        print(f"  {k:25s}: {v}")

    return result


# ------------------------------------------------------------------ #
# Command-line interface
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    from transformers import pipeline as hf_pipeline

    parser = argparse.ArgumentParser(
        description="Fuse Depth-Anything V2 depth maps with MASt3R point cloud"
    )
    parser.add_argument("frames_dir",   type=str)
    parser.add_argument("poses_path",   type=str)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--mast3r_ply", type=str,
                        default="output/pointcloud_raw.ply")
    parser.add_argument("--voxel_size", type=float, default=0.02)
    args = parser.parse_args()

    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0,
    )

    fuse_depth_maps(
        frames_dir      = args.frames_dir,
        poses_path      = args.poses_path,
        output_dir      = args.output_dir,
        mast3r_ply_path = args.mast3r_ply,
        depth_pipe      = depth_pipe,
        voxel_size      = args.voxel_size,
    )