"""
reconstruct.py
--------------
Takes a folder of frames (output of extract_frames.py) and runs MASt3R
to produce:
  - A dense coloured point cloud  → output/pointcloud_raw.ply
  - Camera poses for every frame  → output/poses.json
  - Per-frame depth maps          → output/depths/frame_XXXX_depth.npy

How MASt3R works (simplified):
  1. Load all images
  2. Build a graph of image pairs (every image paired with its neighbours)
  3. Run the MASt3R transformer on each pair → pairwise 3D pointmaps
  4. Run global point cloud optimisation → one unified 3D scene
  5. Extract camera intrinsics + extrinsics from the optimised result
"""

import os
import sys
import json
import numpy as np
from pathlib import Path


def load_mast3r_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load the MASt3R model from a checkpoint file.
    MASt3R is built on top of DUSt3R so we import from the dust3r submodule.
    """
    # MASt3R must be on the path before this import
    from mast3r.model import AsymmetricMASt3R

    print(f"Loading MASt3R from: {checkpoint_path}")
    print(f"Device: {device}")

    model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
    model.eval()

    print("MASt3R model loaded successfully")
    return model


def select_keyframes(frames_dir: str, metadata_path: str, max_frames: int = 60):
    """
    Select which frames to feed into MASt3R.

    MASt3R memory usage scales with the number of image pairs (N^2 / 2).
    On a Colab T4 with 15GB VRAM, ~60 frames is the safe upper limit.
    If we have more than max_frames we subsample evenly.

    Returns a list of full file paths.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    all_frames = metadata["frame_files"]
    total = len(all_frames)

    if total <= max_frames:
        selected = all_frames
    else:
        indices = np.linspace(0, total - 1, max_frames, dtype=int)
        selected = [all_frames[i] for i in indices]
        print(f"Subsampled from {total} → {len(selected)} frames for MASt3R")

    frame_paths = [os.path.join(frames_dir, f) for f in selected]

    print(f"Using {len(frame_paths)} frames for reconstruction")
    return frame_paths


def run_mast3r_reconstruction(
    frame_paths: list,
    checkpoint_path: str,
    output_dir: str,
    device: str = "cuda",
    min_conf_threshold: float = 3.0,
):
    """
    Core reconstruction function.

    Parameters
    ----------
    frame_paths       : list of paths to frame PNG files
    checkpoint_path   : path to the MASt3R .pth checkpoint
    output_dir        : where to save all outputs
    device            : 'cuda' or 'cpu'
    min_conf_threshold: confidence threshold for point filtering.
                        Points below this confidence are discarded.
                        Lower = denser but noisier. Higher = cleaner but sparser.
                        3.0 is a good default.

    Returns
    -------
    dict with keys:
        pointcloud_path  : path to saved .ply file
        poses_path       : path to saved poses.json
        depths_dir       : path to folder of depth .npy files
        num_points       : number of points in the cloud
        num_frames       : number of frames processed
    """

    os.makedirs(output_dir, exist_ok=True)
    depths_dir = os.path.join(output_dir, "depths")
    os.makedirs(depths_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Imports — these must come after MASt3R is added to sys.path
    # ------------------------------------------------------------------ #
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    model = load_mast3r_model(checkpoint_path, device)

    # ------------------------------------------------------------------ #
    # 2. Load images into MASt3R's expected format
    # ------------------------------------------------------------------ #
    print(f"\nLoading {len(frame_paths)} images...")
    images = load_images(frame_paths, size=512, verbose=True)
    print(f"Images loaded: {len(images)}")

    # ------------------------------------------------------------------ #
    # 3. Build image pairs
    #
    # We use a "window" pairing strategy: each frame is paired with the
    # next `win_size` frames. This is more efficient than all-pairs and
    # works well for video (frames are temporally ordered).
    # ------------------------------------------------------------------ #
    win_size = 5  # each frame paired with 5 neighbours
    print(f"\nBuilding image pairs (window size = {win_size})...")
    pairs = make_pairs(
        images,
        scene_graph=f"swin-{win_size}",
        prefilter=None,
        symmetrize=True
    )
    print(f"Total pairs: {len(pairs)}")

    # ------------------------------------------------------------------ #
    # 4. Run MASt3R inference on all pairs
    #
    # This is the heavy compute step. For 60 frames with win_size=5
    # that's ~300 pairs. Each pair runs through the transformer.
    # On a T4 GPU this takes roughly 3-6 minutes.
    # ------------------------------------------------------------------ #
    print(f"\nRunning MASt3R inference on {len(pairs)} pairs...")
    print("This will take 3–8 minutes on a T4 GPU. Please wait...\n")

    output = inference(pairs, model, device, batch_size=1, verbose=True)

    # ------------------------------------------------------------------ #
    # 5. Global alignment
    #
    # This solves for a globally consistent set of camera poses and
    # 3D point positions. It's an optimisation problem — the aligner
    # adjusts poses until all the pairwise predictions agree.
    # ------------------------------------------------------------------ #
    print("\nRunning global alignment (optimising camera poses)...")

    # PointCloudOptimizer mode: full optimisation with confidence weighting
    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer
    )

    # Run the optimisation — 300 iterations is enough for most scenes
    loss = scene.compute_global_alignment(
        init="mst",
        niter=300,
        schedule="cosine",
        lr=0.01
    )
    print(f"Global alignment complete. Final loss: {loss:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Extract results from the optimised scene
    # ------------------------------------------------------------------ #
    print("\nExtracting 3D points and camera poses...")

    # get_pts3d() returns per-frame pointmaps: list of (H, W, 3) arrays
    # Each pixel has an XYZ coordinate in world space
    pts3d = scene.get_pts3d()

    # get_masks() returns confidence masks: list of (H, W) boolean arrays
    # True = confident point, False = discard
    masks = scene.get_masks()

    # get_im_poses() returns camera-to-world 4x4 transform matrices
    # Shape: (N, 4, 4)
    cam_poses = scene.get_im_poses()

    # get_intrinsics() returns camera intrinsic matrices
    # Shape: (N, 3, 3)  — focal length, principal point
    intrinsics = scene.get_intrinsics()

    # ------------------------------------------------------------------ #
    # 7. Apply confidence threshold and build point cloud arrays
    # ------------------------------------------------------------------ #
    print(f"Filtering points with confidence threshold: {min_conf_threshold}")

    # get_conf() returns per-frame confidence maps, shape (H, W)
    confs = scene.get_conf()

    all_points = []   # XYZ
    all_colours = []  # RGB

    frame_depth_maps = []

    for i, (pts, mask, conf, img_dict) in enumerate(
        zip(pts3d, masks, confs, images)
    ):
        pts_np   = pts.detach().cpu().numpy()    # (H, W, 3)
        conf_np  = conf.detach().cpu().numpy()   # (H, W)
        mask_np  = mask.detach().cpu().numpy()   # (H, W) bool

        # Combine the MASt3R mask with our confidence threshold
        conf_mask = conf_np >= min_conf_threshold
        combined_mask = mask_np & conf_mask       # (H, W) bool

        # Extract valid points
        valid_pts = pts_np[combined_mask]         # (M, 3)
        all_points.append(valid_pts)

        # Get colours from the original image
        # img_dict['img'] is a tensor of shape (1, 3, H, W) in [-1, 1]
        import torch
        img_tensor = img_dict['img']              # (1, 3, H, W)
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = ((img_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
        # img_np is now (H, W, 3) RGB uint8

        valid_colours = img_np[combined_mask]     # (M, 3)
        all_colours.append(valid_colours)

        # Save depth map for Task 3
        # Depth = Z coordinate (distance along camera axis)
        depth_map = pts_np[:, :, 2]               # (H, W)
        depth_path = os.path.join(depths_dir, f"frame_{i:04d}_depth.npy")
        np.save(depth_path, depth_map)
        frame_depth_maps.append(depth_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed frame {i+1}/{len(pts3d)} "
                  f"— {valid_pts.shape[0]} points kept")

    # Stack everything into single arrays
    points_array  = np.concatenate(all_points,  axis=0)   # (Total_M, 3)
    colours_array = np.concatenate(all_colours, axis=0)   # (Total_M, 3)

    print(f"\nTotal points in cloud: {points_array.shape[0]:,}")

    # ------------------------------------------------------------------ #
    # 8. Save point cloud as .ply
    # ------------------------------------------------------------------ #
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points_array.astype(np.float64))
    pcd.colors  = o3d.utility.Vector3dVector(colours_array.astype(np.float64) / 255.0)

    # Remove statistical outliers — points that are far from their neighbours
    # This cleans up floating noise before saving
    print("Removing outlier points...")
    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
    print(f"Points after outlier removal: {len(pcd_clean.points):,}")

    ply_path = os.path.join(output_dir, "pointcloud_raw.ply")
    o3d.io.write_point_cloud(ply_path, pcd_clean)
    print(f"Point cloud saved: {ply_path}")

    # ------------------------------------------------------------------ #
    # 9. Save camera poses as JSON
    # ------------------------------------------------------------------ #
    poses_list = []
    intrinsics_list = []

    for i in range(len(frame_paths)):
        pose_matrix = cam_poses[i].detach().cpu().numpy()  # (4, 4)
        K_matrix    = intrinsics[i].detach().cpu().numpy() # (3, 3)

        poses_list.append({
            "frame_index": i,
            "frame_file": os.path.basename(frame_paths[i]),
            "cam_to_world": pose_matrix.tolist(),   # 4x4 list of lists
        })
        intrinsics_list.append({
            "frame_index": i,
            "K": K_matrix.tolist(),                 # 3x3 list of lists
        })

    poses_data = {
        "num_frames": len(frame_paths),
        "poses": poses_list,
        "intrinsics": intrinsics_list,
        "depth_maps": frame_depth_maps,
    }

    poses_path = os.path.join(output_dir, "poses.json")
    with open(poses_path, "w") as f:
        json.dump(poses_data, f, indent=2)

    print(f"Camera poses saved: {poses_path}")

    # ------------------------------------------------------------------ #
    # 10. Summary
    # ------------------------------------------------------------------ #
    result = {
        "pointcloud_path": ply_path,
        "poses_path": poses_path,
        "depths_dir": depths_dir,
        "num_points": len(pcd_clean.points),
        "num_frames": len(frame_paths),
        "final_loss": float(loss),
    }

    print("\n=== Reconstruction Summary ===")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    return result


# ------------------------------------------------------------------ #
# Command-line interface
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MASt3R reconstruction on extracted frames"
    )
    parser.add_argument("frames_dir",   type=str, help="Path to frames/ folder")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--max_frames", type=int, default=60)

    args = parser.parse_args()

    frames_meta = os.path.join(args.frames_dir, "frames_metadata.json")
    frame_paths = select_keyframes(args.frames_dir, frames_meta, args.max_frames)

    run_mast3r_reconstruction(
        frame_paths=frame_paths,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )