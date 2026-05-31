"""
cloud_builder.py (Optimised)
----------------------------
Fuses per-frame depth maps + camera poses into a single 3D point cloud,
then embeds CLIP features into every point.

Optimised for higher clarity using low-compute geometric filtering without 
altering external pipeline dependencies.
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
        pose     = np.array(
            self.poses[frame_idx]["cam_to_world"], dtype=np.float64
        )  # (4, 4)
        depth_np = np.load(
            os.path.join(depths_dir, f"depth_{frame_idx:04d}.npy")
        ).astype(np.float64)  # (H, W)

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

        # --- ADVANCED LOW-COMPUTE FILTER: Depth Gradient Masking ---
        # Compute gradients to find and remove bleeding edges / phantom geometry
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = kx.T
        grad_x = cv2.filter2D(depth_m, -1, kx)
        grad_y = cv2.filter2D(depth_m, -1, ky)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Exclude pixels where depth changes drastically (edge threshold)
        # 0.05 * depth means we filter edges changing by more than 5% of their depth
        stable_depth_mask = edge_magnitude < (0.05 * depth_m)
        # -------------------------------------------------------------

        # Flatten depth for vectorised ops
        d_flat = depth_m.flatten()  # (H*W,)
        edge_mask_flat = stable_depth_mask.flatten()

        # Build validity mask combining depth ranges and edge stability
        valid = (
            (d_flat >= MIN_DEPTH_M) &
            (d_flat <= MAX_DEPTH_M) &
            np.isfinite(d_flat) &
            edge_mask_flat
        )

        d = d_flat[valid]
        u = self.u_flat[valid]
        v = self.v_flat[valid]

        # Back-project to camera space
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

            # Create Open3D cloud for this batch
            pcd_batch = o3d.geometry.PointCloud()
            pcd_batch.points = o3d.utility.Vector3dVector(
                batch_pts_arr.astype(np.float64)
            )
            pcd_batch.colors = o3d.utility.Vector3dVector(
                batch_cols_arr.astype(np.float64) / 255.0
            )
            
            # Downsample first to save compute on outlier clearing
            pcd_batch = pcd_batch.voxel_down_sample(VOXEL_SIZE)
            
            # --- AGGRESSIVE BATCH-LEVEL OUTLIER CLEANING ---
            # Clean early before misaligned batches stack and merge into solid clumps
            pcd_batch, _ = pcd_batch.remove_statistical_outlier(
                nb_neighbors = int(OUTLIER_NEIGHBORS * 0.5), 
                std_ratio    = OUTLIER_STD_RATIO * 0.8 # Tighter threshold
            )
            # -----------------------------------------------

            all_points.append(np.asarray(pcd_batch.points))
            all_colours.append(np.asarray(pcd_batch.colors))

            print(f"  Batch {batch_start//batch_size + 1}/"
                  f"{(len(frame_paths)-1)//batch_size + 1} "
                  f"— frames {batch_start+1}–{batch_end} "
                  f"→ {len(pcd_batch.points):,} pts after edge filter + batched SOR")

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

        # Final Outlier Sweep
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
        clip_embedder,
        segmentor,
        output_dir: str,
        keyframe_step: int = 5,
    ) -> dict:
        """
        [Unchanged to completely preserve downstream dependencies]
        """
        os.makedirs(output_dir, exist_ok=True)

        pts_all  = np.asarray(pcd.points,  dtype=np.float32)
        cols_all = np.asarray(pcd.colors,  dtype=np.float32)
        M        = len(pts_all)

        embeddings    = np.zeros((M, 512), dtype=np.float32)
        embed_counts  = np.zeros(M, dtype=np.int32)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        keyframe_indices = list(range(0, len(frame_paths), keyframe_step))
        print(f"Embedding CLIP features from {len(keyframe_indices)} "
              f"keyframes (every {keyframe_step} frames)...")

        for ki, frame_idx in enumerate(keyframe_indices):
            fpath    = frame_paths[frame_idx]
            img_bgr  = cv2.imread(fpath)
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            masks = segmentor.segment(img_bgr)
            if not masks:
                continue

            depth_np = np.load(
                os.path.join(depths_dir, f"depth_{frame_idx:04d}.npy")
            ).astype(np.float64)
            depth_m = depth_np * self.scale
            pose    = np.array(
                self.poses[frame_idx]["cam_to_world"], dtype=np.float64
            )

            frame_embedded = 0

            for mask_2d in masks:
                if mask_2d.sum() < 100:
                    continue

                embedding = clip_embedder.embed_masked_region(
                    img_rgb, mask_2d
                )
                if embedding is None:
                    continue

                mask_flat = mask_2d.flatten()
                u_mask    = self.u_flat[mask_flat]
                v_mask    = self.v_flat[mask_flat]
                d_flat    = depth_m.flatten()
                d_mask    = d_flat[mask_flat]

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

                x_cam = (u_v - self.cx) / self.fx * d_v
                y_cam = (v_v - self.cy) / self.fy * d_v
                z_cam = d_v
                ones  = np.ones_like(z_cam)
                pts_cam   = np.stack([x_cam, y_cam, z_cam, ones])
                pts_world = (pose @ pts_cam)[:3].T

                step = max(1, len(pts_world) // 200)
                for pt3d in pts_world[::step]:
                    _, idx_nn, dist_sq = pcd_tree.search_knn_vector_3d(
                        pt3d.astype(np.float64), 1
                    )
                    nn_idx  = idx_nn[0]
                    nn_dist = dist_sq[0] ** 0.5

                    if nn_dist < 0.05:
                        embeddings[nn_idx]   += embedding
                        embed_counts[nn_idx] += 1
                        frame_embedded       += 1

            if (ki + 1) % 5 == 0 or ki == 0:
                print(f"  Keyframe {ki+1}/{len(keyframe_indices)} "
                      f"(frame {frame_idx}) "
                      f"— {len(masks)} masks "
                      f"→ {frame_embedded} embeddings assigned")

        has_embed = embed_counts > 0
        embeddings[has_embed] = (
            embeddings[has_embed] /
            embed_counts[has_embed, np.newaxis]
        )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        embeddings = embeddings / norms

        n_embedded = has_embed.sum()
        print(f"\n  Points with CLIP embeddings : {n_embedded:,} / {M:,} "
              f"({n_embedded/M*100:.1f}%)")

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