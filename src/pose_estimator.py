"""
pose_estimator.py
-----------------
Estimates camera pose for every frame using ORB keypoints + depth-anchored PnP.

Why this approach instead of MASt3R or optical flow:
  - ORB runs on CPU in <5ms per frame — suitable for real-time
  - PnP with depth anchoring avoids the drift problem of integrating
    relative poses (which caused our 19m-wide room problem before)
  - Each pose is estimated relative to world frame 0, not the previous
    frame — so errors do not compound over time

How it works:
  1. Frame 0 is defined as the world origin (identity pose)
  2. For each subsequent frame i:
     a. Match ORB keypoints between frame i and recent reference frames
     b. For each match, look up the 3D world position of the reference
        keypoint using its depth map and the reference pose
     c. We now have 3D world points ↔ 2D image points in frame i
     d. Run PnP RANSAC to find the pose of frame i directly
  3. If PnP fails (too few matches), copy the previous pose + small step
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ORB_FEATURES,
    PNP_REPROJECTION_ERR,
    PNP_CONFIDENCE,
    MIN_PNP_INLIERS,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
)


class PoseEstimator:
    """
    Estimates camera-to-world poses for every frame in a video.
    """

    def __init__(self, image_w: int, image_h: int, focal_px: float = None):
        """
        Parameters
        ----------
        image_w   : frame width in pixels
        image_h   : frame height in pixels
        focal_px  : focal length in pixels
                    If None, estimated from image size using 70-degree FOV
                    (typical smartphone horizontal FOV)
        """
        self.W = image_w
        self.H = image_h

        # Estimate focal length if not provided
        if focal_px is None:
            # 70-degree FOV: focal = W / (2 * tan(35deg)) ≈ W / 1.4
            focal_px = image_w / 1.4

        self.focal = focal_px
        self.cx    = image_w  / 2.0
        self.cy    = image_h / 2.0

        # Camera intrinsic matrix K
        self.K = np.array([
            [focal_px,       0,  self.cx],
            [0,        focal_px,  self.cy],
            [0,               0,       1],
        ], dtype=np.float64)

        # ORB detector — fast, works on CPU, no GPU needed
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        print(f"PoseEstimator initialised")
        print(f"  Image size   : {image_w} × {image_h}")
        print(f"  Focal length : {focal_px:.1f} px")
        print(f"  cx, cy       : ({self.cx:.1f}, {self.cy:.1f})")

    def extract_keypoints(self, image_bgr: np.ndarray):
        """
        Detect ORB keypoints and descriptors in a BGR image.

        Returns
        -------
        keypoints   : list of cv2.KeyPoint
        descriptors : (N, 32) uint8 numpy array, or None if no points found
        """
        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(grey, None)
        return kps, des

    def backproject_point(
        self,
        u,
        v,
        depth_map,
        pose_cam_to_world,
        scale,
    ):
        """
        Back-project a 2D pixel (u, v) to a 3D world point
        using the depth map and camera pose.

        Parameters
        ----------
        u, v               : pixel coordinates (column, row)
        depth_map          : (H, W) relative depth map
        pose_cam_to_world  : (4, 4) camera-to-world transform
        scale              : multiply depth by this to get metres

        Returns
        -------
        point_world : (3,) float64 array, or None if depth invalid
        """
        u_i = int(round(u))
        v_i = int(round(v))

        # Clamp to valid pixel range
        u_i = np.clip(u_i, 0, self.W - 1)
        v_i = np.clip(v_i, 0, self.H - 1)

        depth_raw = float(depth_map[v_i, u_i])

        # Normalise from 0-255 range to 0-1 if needed
        if depth_raw > 1.0:
            depth_raw = depth_raw / 255.0

        depth_m = depth_raw * scale

        # Reject invalid depths
        if depth_m < MIN_DEPTH_M or depth_m > MAX_DEPTH_M:
            return None
        if not np.isfinite(depth_m):
            return None

        # Back-project to camera space
        x_cam = (u_i - self.cx) / self.focal * depth_m
        y_cam = (v_i - self.cy) / self.focal * depth_m
        z_cam = depth_m

        # Transform to world space
        pt_cam   = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
        pt_world = pose_cam_to_world @ pt_cam

        return pt_world[:3]

    def estimate_poses(
        self,
        frame_paths: list,
        depths_dir: str,
        global_scale: float,
    ) -> dict:
        """
        Estimate camera poses for all frames.

        Parameters
        ----------
        frame_paths  : list of paths to frame PNG files (in temporal order)
        depths_dir   : directory containing depth_XXXX.npy files
        global_scale : multiply depth values by this to get metres

        Returns
        -------
        poses_data : dict — see poses.json format below
        {
          "num_frames" : N,
          "K"          : [[fx,0,cx],[0,fy,cy],[0,0,1]],
          "image_w"    : W,
          "image_h"    : H,
          "focal_px"   : focal,
          "scale"      : global_scale,
          "poses"      : [
            {
              "frame_index"  : i,
              "frame_file"   : "frame_0000.png",
              "cam_to_world" : [[4x4 matrix as list of lists]],
              "pnp_inliers"  : int,
              "status"       : "origin" | "pnp" | "fallback"
            },
            ...
          ]
        }
        """
        N = len(frame_paths)
        print(f"Estimating poses for {N} frames...")

        # ── Step 1: Extract keypoints from all frames upfront ─────────────────
        print("  Extracting ORB keypoints...")
        all_kps  = []
        all_des  = []
        all_imgs = []

        for i, fpath in enumerate(frame_paths):
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read frame: {fpath}")
            kps, des = self.extract_keypoints(img_bgr)
            all_kps.append(kps)
            all_des.append(des)
            all_imgs.append(img_bgr)

            if (i + 1) % 20 == 0 or i == 0:
                n_kp = len(kps) if kps else 0
                print(f"    Frame {i+1:3d}/{N} — {n_kp} keypoints")

        # ── Step 2: Pose estimation loop ──────────────────────────────────────
        print("\n  Estimating poses via depth-anchored PnP...")

        # Frame 0 is the world origin
        global_poses = [np.eye(4, dtype=np.float64)]
        pose_log     = [{"status": "origin", "pnp_inliers": 0}]

        # Reference frames: always try matching against these
        # We maintain a small pool of recently-successful keyframes
        ref_pool = [0]  # start with just frame 0

        for i in range(1, N):
            depth_path = os.path.join(depths_dir, f"depth_{i:04d}.npy")
            depth_i    = np.load(depth_path)  # not used for PnP but loaded for later

            best_pose    = None
            best_inliers = 0
            best_status  = "fallback"

            # Try matching against each reference frame in the pool
            # Use the most recent first (usually most overlap)
            for ref_idx in sorted(ref_pool, reverse=True):

                if all_des[ref_idx] is None or all_des[i] is None:
                    continue

                # Match keypoints
                matches = self.bf.match(all_des[ref_idx], all_des[i])
                if len(matches) < 8:
                    continue

                # Keep best 70% of matches by descriptor distance
                matches = sorted(matches, key=lambda x: x.distance)
                matches = matches[:max(8, int(len(matches) * 0.7))]

                # Load reference depth map
                ref_depth_path = os.path.join(
                    depths_dir, f"depth_{ref_idx:04d}.npy"
                )
                ref_depth = np.load(ref_depth_path)
                ref_pose  = global_poses[ref_idx]

                # Build 3D world ↔ 2D current frame correspondences
                pts3d_world = []
                pts2d_curr  = []

                for m in matches:
                    # 2D position in reference frame
                    u_ref, v_ref = all_kps[ref_idx][m.queryIdx].pt

                    # Back-project to 3D world using reference depth + pose
                    pt_world = self.backproject_point(
                        u_ref, v_ref, ref_depth, ref_pose, global_scale
                    )
                    if pt_world is None:
                        continue

                    # 2D position in current frame
                    pts3d_world.append(pt_world)
                    pts2d_curr.append(all_kps[i][m.trainIdx].pt)

                if len(pts3d_world) < MIN_PNP_INLIERS:
                    continue

                pts3d_arr = np.array(pts3d_world, dtype=np.float32)
                pts2d_arr = np.array(pts2d_curr,  dtype=np.float32)

                # Solve PnP — find camera pose given 3D↔2D correspondences
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d_arr,
                    pts2d_arr,
                    self.K.astype(np.float32),
                    None,  # no distortion coefficients
                    reprojectionError = PNP_REPROJECTION_ERR,
                    confidence        = PNP_CONFIDENCE,
                    iterationsCount   = 1000,
                    flags             = cv2.SOLVEPNP_ITERATIVE,
                )

                if not success or inliers is None:
                    continue

                n_inliers = len(inliers)
                if n_inliers < MIN_PNP_INLIERS:
                    continue

                if n_inliers > best_inliers:
                    best_inliers = n_inliers

                    # Convert rvec/tvec → 4x4 world-to-camera matrix
                    R_mat, _ = cv2.Rodrigues(rvec)
                    T_wc     = np.eye(4, dtype=np.float64)
                    T_wc[:3, :3] = R_mat
                    T_wc[:3,  3] = tvec.flatten()

                    # Invert to get camera-to-world
                    best_pose   = np.linalg.inv(T_wc)
                    best_status = "pnp"

            # ── Accept or fall back ───────────────────────────────────────────
            if best_pose is not None and best_inliers >= MIN_PNP_INLIERS:
                global_poses.append(best_pose)
                pose_log.append({
                    "status"      : best_status,
                    "pnp_inliers" : best_inliers,
                })

                # Add to reference pool if this pose is high confidence
                # Keep pool size ≤ 8 to avoid slow matching
                if best_inliers >= 20:
                    ref_pool.append(i)
                    if len(ref_pool) > 8:
                        # Remove oldest non-origin reference
                        ref_pool = [0] + ref_pool[-7:]

            else:
                # Fallback: copy previous pose with tiny forward step
                # This keeps the trajectory moving rather than freezing
                prev_pose = global_poses[-1].copy()
                # Move 2cm forward along the camera's Z axis
                forward = prev_pose[:3, 2] * 0.02
                prev_pose[:3, 3] += forward
                global_poses.append(prev_pose)
                pose_log.append({
                    "status"      : "fallback",
                    "pnp_inliers" : 0,
                })

            if (i + 1) % 20 == 0 or i == 1:
                recent_log = pose_log[-min(20, len(pose_log)):]
                pnp_recent = sum(
                    1 for p in recent_log if p["status"] == "pnp"
                )
                print(f"    Frame {i+1:3d}/{N} — "
                      f"PnP success: {pnp_recent}/{len(recent_log)} recent  "
                      f"| best inliers: {best_inliers}")

        # ── Step 3: Summary statistics ────────────────────────────────────────
        n_pnp      = sum(1 for p in pose_log if p["status"] == "pnp")
        n_fallback = sum(1 for p in pose_log if p["status"] == "fallback")
        avg_inliers = np.mean([
            p["pnp_inliers"] for p in pose_log if p["pnp_inliers"] > 0
        ]) if n_pnp > 0 else 0

        print(f"\n  Pose estimation complete:")
        print(f"    PnP success  : {n_pnp}/{N}  "
              f"({n_pnp/N*100:.1f}%)")
        print(f"    Fallback     : {n_fallback}")
        print(f"    Avg inliers  : {avg_inliers:.1f}")
        print(f"    (Good = >60% PnP success, >15 avg inliers)")

        # ── Step 4: Build and return poses_data dict ──────────────────────────
        poses_list = []
        for i, (pose, log) in enumerate(zip(global_poses, pose_log)):
            poses_list.append({
                "frame_index"  : i,
                "frame_file"   : os.path.basename(frame_paths[i]),
                "cam_to_world" : pose.tolist(),
                "pnp_inliers"  : log["pnp_inliers"],
                "status"       : log["status"],
            })

        poses_data = {
            "num_frames" : N,
            "K"          : self.K.tolist(),
            "image_w"    : self.W,
            "image_h"    : self.H,
            "focal_px"   : float(self.focal),
            "scale"      : float(global_scale),
            "pnp_success_pct" : round(n_pnp / N * 100, 1),
            "avg_inliers"     : round(float(avg_inliers), 1),
            "poses"      : poses_list,
        }

        return poses_data

    def save_poses(self, poses_data: dict, output_path: str):
        """Save poses_data dict to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(poses_data, f, indent=2)
        print(f"✓ Poses saved: {output_path}")
        print(f"  ({poses_data['num_frames']} frames, "
              f"{poses_data['pnp_success_pct']}% PnP success)")

    def get_camera_positions(self, poses_data: dict) -> np.ndarray:
        """
        Extract camera XYZ positions from poses_data.
        Useful for visualising the camera trajectory.

        Returns
        -------
        positions : (N, 3) float64 array of camera world positions
        """
        positions = []
        for p in poses_data["poses"]:
            mat = np.array(p["cam_to_world"])
            positions.append(mat[:3, 3])
        return np.array(positions)