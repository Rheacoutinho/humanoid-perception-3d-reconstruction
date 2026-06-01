"""
cloud_builder.py (Fast Coarse-to-Fine Alignment)
------------------------------------------------
Fuses per-frame depth maps + camera poses into a single 3D point cloud,
then embeds CLIP features into every point.

Optimised for crisp, recognisable object surfaces and fast execution 
using decoupled coarse-to-fine keyframe ICP.
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

# Safe cross-version check for Open3D registration APIs
try:
    import open3d.pipelines.registration as o3d_reg
    _ICP_ALIGN = o3d_reg.registration_icp
    _ICP_ESTIMATE = o3d_reg.TransformationEstimationPointToPoint
    _ICP_CRITERIA = o3d_reg.ICPConvergenceCriteria
except (ImportError, AttributeError):
    _ICP_ALIGN = o3d.registration.registration_icp
    _ICP_ESTIMATE = o3d.registration.TransformationEstimationPointToPoint
    _ICP_CRITERIA = o3d.registration.ICPConvergenceCriteria


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
        )
        depth_np = np.load(
            os.path.join(depths_dir, f"depth_{frame_idx:04d}.npy")
        ).astype(np.float64)

        depth_m = depth_np * self.scale

        img_bgr = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if depth_m.shape != (self.H, self.W):
            depth_m = cv2.resize(
                depth_m, (self.W, self.H), interpolation=cv2.INTER_LINEAR
            )

        d_flat = depth_m.flatten()

        valid = (
            (d_flat >= MIN_DEPTH_M) &
            (d_flat <= MAX_DEPTH_M) &
            np.isfinite(d_flat)
        )

        d = d_flat[valid]
        u = self.u_flat[valid]
        v = self.v_flat[valid]

        x_cam = (u - self.cx) / self.fx * d
        y_cam = (v - self.cy) / self.fy * d
        z_cam = d
        ones  = np.ones_like(z_cam)

        pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=0)
        pts_world = (pose @ pts_cam)[:3, :].T
        colours = img_rgb.reshape(-1, 3)[valid]
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
        Assembles a recognizable, dense 3D scene using an accelerated coarse-to-fine 
        ICP loop across keyframes, mitigating tracking drift in seconds.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step = 3 samples ~26 frames evenly across your 78 total frames
        frame_step = 3 
        keyframe_indices = list(range(0, len(frame_paths), frame_step))
        
        print(f"Building crisp dense reconstruction using {len(keyframe_indices)} aligned keyframes...")

        # Initialize the global cloud container with the first frame
        pcd_global = o3d.geometry.PointCloud()
        pts_init, cols_init, _ = self.backproject_frame(0, frame_paths[0], depths_dir)
        pcd_global.points = o3d.utility.Vector3dVector(pts_init.astype(np.float64))
        pcd_global.colors = o3d.utility.Vector3dVector(cols_init.astype(np.float64) / 255.0)
        pcd_global = pcd_global.voxel_down_sample(VOXEL_SIZE)

        # Multi-stage thresholds based on structural dimensions
        coarse_threshold = VOXEL_SIZE * 6.0
        fine_threshold   = VOXEL_SIZE * 2.0

        for i, idx in enumerate(keyframe_indices[1:], start=1):
            pts, cols, _ = self.backproject_frame(idx, frame_paths[idx], depths_dir)
            
            pcd_source = o3d.geometry.PointCloud()
            pcd_source.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd_source.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
            
            # CRITICAL SPEED FIX: Downsample aggressively purely for tracking evaluation
            pcd_target_down = pcd_global.voxel_down_sample(VOXEL_SIZE * 4.0)
            pcd_source_down = pcd_source.voxel_down_sample(VOXEL_SIZE * 4.0)

            # Pass 1: Coarse Alignment (Pulls drifting objects together)
            reg_coarse = _ICP_ALIGN(
                pcd_source_down, pcd_target_down, coarse_threshold, np.identity(4),
                _ICP_ESTIMATE(), _ICP_CRITERIA(max_iteration=20)
            )
            
            # Pass 2: Fine Alignment (Locks object faces and boundaries)
            reg_fine = _ICP_ALIGN(
                pcd_source_down, pcd_target_down, fine_threshold, reg_coarse.transformation,
                _ICP_ESTIMATE(), _ICP_CRITERIA(max_iteration=25)
            )
            
            T_matrix = reg_fine.transformation
            
            # Apply verified adjustments across the system pipeline
            if reg_fine.fitness > 0.10:
                # Update trajectory tracking matrices in-place for downstream CLIP operations
                current_pose = np.array(self.poses[idx]["cam_to_world"], dtype=np.float64)
                self.poses[idx]["cam_to_world"] = T_matrix @ current_pose
                
                # Transform full-resolution source data
                pcd_source.transform(T_matrix)

            # Append aligned full-density points into global context
            pcd_global += pcd_source
            pcd_global = pcd_global.voxel_down_sample(VOXEL_SIZE)

            if i % 5 == 0 or idx == keyframe_indices[-1]:
                print(f"  Aligned Keyframe {i+1}/{len(keyframe_indices)} (Frame {idx}) "
                      f"→ Cloud Density: {len(pcd_global.points):,} pts")

        # Refine surface noise
        print("\nSharpening final structural features...")
        pcd_global, _ = pcd_global.remove_statistical_outlier(
            nb_neighbors=int(OUTLIER_NEIGHBORS * 1.2),
            std_ratio=OUTLIER_STD_RATIO * 0.8
        )
        print(f"✓ Crisp reconstruction complete: {len(pcd_global.points):,} total points.")

        ply_path = os.path.join(output_dir, "pointcloud_rgb.ply")
        o3d.io.write_point_cloud(ply_path, pcd_global)
        return pcd_global

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
        [Unchanged signature and logic to ensure perfect downstream safety]
        """
        os.makedirs(output_dir, exist_ok=True)

        pts_all  = np.asarray(pcd.points,  dtype=np.float32)
        cols_all = np.asarray(pcd.colors,  dtype=np.float32)
        M        = len(pts_all)

        embeddings    = np.zeros((M, 512), dtype=np.float32)
        embed_counts  = np.zeros(M, dtype=np.int32)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        keyframe_indices = list(range(0, len(frame_paths), keyframe_step))
        print(f"Embedding CLIP features from {len(keyframe_indices)} keyframes...")

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

                embedding = clip_embedder.embed_masked_region(img_rgb, mask_2d)
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

        has_embed = embed_counts > 0
        embeddings[has_embed] = embeddings[has_embed] / embed_counts[has_embed, np.newaxis]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        embeddings = embeddings / norms

        npz_path = os.path.join(output_dir, "pointcloud_clip.npz")
        np.savez_compressed(
            npz_path,
            points      = pts_all,
            colours     = cols_all,
            embeddings  = embeddings,
            embed_counts = embed_counts,
        )
        return {
            "points": pts_all,
            "colours": cols_all,
            "embeddings": embeddings,
            "embed_counts": embed_counts,
        }