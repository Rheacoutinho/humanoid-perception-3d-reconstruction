"""
cloud_builder.py
----------------
Fuses per-frame depth maps + camera poses into a single 3D point cloud,
then embeds CLIP features into every point.

Two outputs:
  1. pointcloud_rgb.ply   — coloured point cloud (XYZ + RGB)
  2. pointcloud_clip.npz  — same points + 512-dim CLIP embedding per point
                            (too large for .ply, stored as numpy archive)

Why CLIP embeddings per point:
  - Each 3D point inherits the CLIP embedding of the 2D mask it belongs to
  - This means every point encodes semantic meaning, not just geometry
  - Query engine can then do cosine similarity between a text query
    and all point embeddings to find the matching 3D region
  - This is the core of language-queryable 3D reconstruction

Design decisions:
  - Points from multiple frames that land in the same voxel are merged
    (voxel downsampling) — keeps memory manageable
  - CLIP embeddings are averaged when multiple frames see the same voxel
    — improves robustness of the semantic embedding
  - We process frames in batches to avoid OOM on Colab
"""

import numpy as np
import cv2
import open3d as o3d
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VOXEL_SIZE,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    OUTLIER_NEIGHBORS,
    OUTLIER_STD_RATIO,
)


class CloudBuilder:
    """
    Builds a language-embedded point cloud from depth maps + poses + CLIP.
    """

    def __init__(self, poses_data: dict):
        """
        Parameters
        ----------
        poses_data : dict loaded from poses.json
                     Must contain K, image_w, image_h, scale, poses
        """
        self.K       = np.array(poses_data["K"],    dtype=np.float64)
        self.W       = poses_data["image_w"]
        self.H       = poses_data["image_h"]
        self.scale   = poses_data["scale"]
        self.poses   = poses_data["poses"]
        self.N       = poses_data["num_frames"]

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        # Pre-build pixel grids — computed once, reused every frame
        u_grid, v_grid = np.meshgrid(
            np.arange(self.W), np.arange(self.H)
        )
        self.u_flat = u_grid.flatten().astype(np.float64)
        self.v_flat = v_grid.flatten().astype(np.float64)

        print(f"CloudBuilder initialised")
        print(f"  Frames    : {self.N}")
        print(f"  Image     : {self.W} × {self.H}")
        print(f"  Scale     : {self.scale:.4f}")
        print(f"  Depth rng : {MIN_DEPTH_M}m – {MAX_DEPTH_M}m")

    def backproject_frame(
        self,
        frame_idx: int,
        frame_path: str,
        depths_dir: str,
    ):
        """
        Back-project one frame's depth map into 3D world points.

        Parameters
        ----------
        frame_idx  : index of this frame in poses list
        frame_path : path to the RGB frame PNG
        depths_dir : directory containing depth_XXXX.npy files

        Returns
        -------
        points  : (N, 3) float64 — world XYZ coordinates
        colours : (N, 3) uint8  — RGB colours from the frame
        mask_2d : (H, W) bool   — which pixels produced valid points
                                  (used later for CLIP mask projection)
        """
        pose     = np.array(
            self.poses[frame_idx]["cam_to_world"], dtype=np.float64
        )  # (4, 4)
        depth_np = np.load(
            os.path.join(depths_dir, f"depth_{frame_idx:04d}.npy")
        ).astype(np.float64)  # (H, W) — already normalised 0-1

        # Convert to metric depth
        depth_m = depth_np * self.scale  # (H, W)

        # Load RGB frame
        img_bgr = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize depth to match frame exactly if needed
        if depth_m.shape != (self.H, self.W):
            depth_m = cv2.resize(
                depth_m, (self.W, self.H), interpolation=cv2.INTER_LINEAR
            )

        # Flatten depth for vectorised ops
        d_flat = depth_m.flatten()  # (H*W,)

        # Build validity mask
        valid = (
            (d_flat >= MIN_DEPTH_M) &
            (d_flat <= MAX_DEPTH_M) &
            np.isfinite(d_flat)
        )

        d = d_flat[valid]
        u = self.u_flat[valid]
        v = self.v_flat[valid]

        # Back-project to camera space
        # x_cam = (u - cx) / fx * depth
        # y_cam = (v - cy) / fy * depth
        # z_cam = depth
        x_cam = (u - self.cx) / self.fx * d
        y_cam = (v - self.cy) / self.fy * d
        z_cam = d
        ones  = np.ones_like(z_cam)

        # Stack into (4, N) homogeneous points
        pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=0)

        # Transform to world space: (4, N) -> (N, 3)
        pts_world = (pose @ pts_cam)[:3, :].T

        # Get RGB colours for valid pixels
        colours = img_rgb.reshape(-1, 3)[valid]  # (N, 3)

        # Build 2D mask of valid pixels (for CLIP mask projection later)
        mask_2d = valid.reshape(self.H, self.W)

        return pts_world, colours, mask_2d

    def build_rgb_cloud(
        self,
        frame_paths: list,
        depths_dir: str,
        output_dir: str,
        batch_size: int = 10,
    ) -> o3d.geometry.PointCloud:
        """
        Fuse all frames into one RGB point cloud.

        Processes frames in batches to avoid OOM.
        Voxel-downsamples after each batch to keep memory bounded.

        Parameters
        ----------
        frame_paths : list of paths to frame PNG files
        depths_dir  : directory with depth_XXXX.npy files
        output_dir  : where to save pointcloud_rgb.ply
        batch_size  : frames per batch (10 is safe for Colab T4)

        Returns
        -------
        pcd : cleaned Open3D PointCloud
        """
        os.makedirs(output_dir, exist_ok=True)

        all_points  = []
        all_colours = []
        total_raw   = 0

        print(f"Back-projecting {len(frame_paths)} frames "
              f"(batch size={batch_size})...")

        for batch_start in range(0, len(frame_paths), batch_size):
            batch_end   = min(batch_start + batch_size, len(frame_paths))
            batch_paths = frame_paths[batch_start:batch_end]

            batch_pts  = []
            batch_cols = []

            for local_i, fpath in enumerate(batch_paths):
                global_i = batch_start + local_i
                pts, cols, _ = self.backproject_frame(
                    global_i, fpath, depths_dir
                )
                batch_pts.append(pts)
                batch_cols.append(cols)
                total_raw += len(pts)

            # Stack batch
            batch_pts_arr  = np.concatenate(batch_pts,  axis=0)
            batch_cols_arr = np.concatenate(batch_cols, axis=0)

            # Intermediate voxel downsample to keep memory bounded
            pcd_batch = o3d.geometry.PointCloud()
            pcd_batch.points = o3d.utility.Vector3dVector(
                batch_pts_arr.astype(np.float64)
            )
            pcd_batch.colors = o3d.utility.Vector3dVector(
                batch_cols_arr.astype(np.float64) / 255.0
            )
            pcd_batch = pcd_batch.voxel_down_sample(VOXEL_SIZE)

            all_points.append(np.asarray(pcd_batch.points))
            all_colours.append(np.asarray(pcd_batch.colors))

            print(f"  Batch {batch_start//batch_size + 1}/"
                  f"{(len(frame_paths)-1)//batch_size + 1} "
                  f"— frames {batch_start+1}–{batch_end} "
                  f"→ {len(pcd_batch.points):,} pts after downsample")

        # Final merge and clean
        print(f"\nMerging all batches...")
        merged_pts  = np.concatenate(all_points,  axis=0)
        merged_cols = np.concatenate(all_colours, axis=0)
        print(f"  Total raw points : {total_raw:,}")
        print(f"  After batched DS : {len(merged_pts):,}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pts)
        pcd.colors = o3d.utility.Vector3dVector(merged_cols)

        # Final voxel downsample
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        print(f"  After final DS   : {len(pcd.points):,}")

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors = OUTLIER_NEIGHBORS,
            std_ratio    = OUTLIER_STD_RATIO,
        )
        print(f"  After outlier rm : {len(pcd.points):,}")

        # Save
        ply_path = os.path.join(output_dir, "pointcloud_rgb.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        size_mb = os.path.getsize(ply_path) / 1e6
        print(f"\n✓ RGB cloud saved : {ply_path}  ({size_mb:.1f} MB)")

        return pcd

    def embed_clip_features(
        self,
        pcd: o3d.geometry.PointCloud,
        frame_paths: list,
        depths_dir: str,
        clip_embedder,      # CLIPEmbedder instance from clip_embedder.py
        segmentor,          # Segmentor instance from segmentor.py
        output_dir: str,
        keyframe_step: int = 5,
    ) -> dict:
        """
        Assign CLIP embeddings to every point in the cloud.

        Strategy:
          1. For every keyframe (every `keyframe_step` frames):
             a. Run FastSAM to get instance masks
             b. Run CLIP on each masked crop to get a 512-dim embedding
             c. Project each mask into 3D using the frame's depth + pose
             d. Assign the CLIP embedding to all projected points
          2. Points seen from multiple keyframes get their embeddings averaged

        Parameters
        ----------
        pcd           : RGB point cloud from build_rgb_cloud()
        frame_paths   : list of frame PNG paths
        depths_dir    : directory with depth maps
        clip_embedder : CLIPEmbedder instance (from clip_embedder.py)
        segmentor     : Segmentor instance (from segmentor.py)
        output_dir    : where to save pointcloud_clip.npz
        keyframe_step : process every Nth frame for CLIP embedding
                        5 = process 20 keyframes from 100 total

        Returns
        -------
        clip_data : dict with keys:
            "points"     : (M, 3) float32 XYZ
            "colours"    : (M, 3) float32 RGB in [0, 1]
            "embeddings" : (M, 512) float32 CLIP embeddings
            "embed_counts": (M,) int — how many frames contributed
        """
        os.makedirs(output_dir, exist_ok=True)

        pts_all  = np.asarray(pcd.points,  dtype=np.float32)   # (M, 3)
        cols_all = np.asarray(pcd.colors,  dtype=np.float32)   # (M, 3)
        M        = len(pts_all)

        # Initialise embedding accumulator
        embeddings    = np.zeros((M, 512), dtype=np.float32)
        embed_counts  = np.zeros(M, dtype=np.int32)

        # Build a KD-tree for fast nearest-point lookup
        # We use Open3D's KDTree which is much faster than scipy for 3D
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        # Select keyframes
        keyframe_indices = list(range(0, len(frame_paths), keyframe_step))
        print(f"Embedding CLIP features from {len(keyframe_indices)} "
              f"keyframes (every {keyframe_step} frames)...")

        for ki, frame_idx in enumerate(keyframe_indices):
            fpath    = frame_paths[frame_idx]
            img_bgr  = cv2.imread(fpath)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Get masks for this frame
            masks = segmentor.segment(img_bgr)  # list of (H, W) bool arrays

            if not masks:
                continue

            # Get depth and pose for this frame
            depth_np = np.load(
                os.path.join(depths_dir, f"depth_{frame_idx:04d}.npy")
            ).astype(np.float64)
            depth_m = depth_np * self.scale
            pose    = np.array(
                self.poses[frame_idx]["cam_to_world"], dtype=np.float64
            )

            frame_embedded = 0

            for mask_2d in masks:
                # Skip very small masks — likely noise
                if mask_2d.sum() < 100:
                    continue

                # Get CLIP embedding for this masked region
                embedding = clip_embedder.embed_masked_region(
                    img_rgb, mask_2d
                )  # (512,) float32

                if embedding is None:
                    continue

                # Project mask pixels into 3D world space
                # Find which pixels are in this mask
                mask_flat = mask_2d.flatten()
                u_mask    = self.u_flat[mask_flat]
                v_mask    = self.v_flat[mask_flat]
                d_flat    = depth_m.flatten()
                d_mask    = d_flat[mask_flat]

                # Filter valid depths
                valid = (
                    (d_mask >= MIN_DEPTH_M) &
                    (d_mask <= MAX_DEPTH_M) &
                    np.isfinite(d_mask)
                )
                if valid.sum() < 10:
                    continue

                u_v = u_mask[valid]
                v_v = v_mask[valid]
                d_v = d_mask[valid]

                # Back-project to world
                x_cam = (u_v - self.cx) / self.fx * d_v
                y_cam = (v_v - self.cy) / self.fy * d_v
                z_cam = d_v
                ones  = np.ones_like(z_cam)
                pts_cam   = np.stack([x_cam, y_cam, z_cam, ones])
                pts_world = (pose @ pts_cam)[:3].T  # (K, 3)

                # For each projected point, find nearest cloud point
                # and accumulate the CLIP embedding
                # We subsample to max 200 points per mask for speed
                step = max(1, len(pts_world) // 200)
                for pt3d in pts_world[::step]:
                    # Search for nearest neighbour in cloud
                    _, idx_nn, dist_sq = pcd_tree.search_knn_vector_3d(
                        pt3d.astype(np.float64), 1
                    )
                    nn_idx  = idx_nn[0]
                    nn_dist = dist_sq[0] ** 0.5

                    # Only assign if the nearest cloud point is within 5cm
                    if nn_dist < 0.05:
                        embeddings[nn_idx]   += embedding
                        embed_counts[nn_idx] += 1
                        frame_embedded       += 1

            if (ki + 1) % 5 == 0 or ki == 0:
                print(f"  Keyframe {ki+1}/{len(keyframe_indices)} "
                      f"(frame {frame_idx}) "
                      f"— {len(masks)} masks "
                      f"→ {frame_embedded} embeddings assigned")

        # Average embeddings where multiple frames contributed
        has_embed = embed_counts > 0
        embeddings[has_embed] = (
            embeddings[has_embed] /
            embed_counts[has_embed, np.newaxis]
        )

        # L2-normalise all embeddings — required for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        embeddings = embeddings / norms

        n_embedded = has_embed.sum()
        print(f"\n  Points with CLIP embeddings : {n_embedded:,} / {M:,} "
              f"({n_embedded/M*100:.1f}%)")

        # Save
        npz_path = os.path.join(output_dir, "pointcloud_clip.npz")
        np.savez_compressed(
            npz_path,
            points      = pts_all,
            colours     = cols_all,
            embeddings  = embeddings,
            embed_counts = embed_counts,
        )
        size_mb = os.path.getsize(npz_path) / 1e6
        print(f"✓ CLIP cloud saved : {npz_path}  ({size_mb:.1f} MB)")

        clip_data = {
            "points"      : pts_all,
            "colours"     : cols_all,
            "embeddings"  : embeddings,
            "embed_counts": embed_counts,
        }
        return clip_data