"""
accuracy.py
-----------
Measures the accuracy and quality of the 3D reconstruction pipeline.

Four metrics reported:

1. DEPTH CONSISTENCY
   How consistent are depth predictions across overlapping frames?
   Take pairs of frames that see the same 3D point from different angles.
   Reproject that point into both frames and compare depth values.
   Lower reprojection error = better depth accuracy.

2. POSE QUALITY
   How stable is the camera trajectory?
   Measures: smoothness of motion, absence of sudden jumps,
   angular velocity consistency.
   Reported as a 0-100 score.

3. QUERY PRECISION
   For each test query, how spatially compact are the results?
   Compact = the system found a specific region, not scattered noise.
   Reported per query as compactness (0-1) and precision@k (0-1).

4. EMBEDDING COVERAGE
   What fraction of the point cloud has semantic embeddings?
   Higher = more of the scene is queryable.
   Reported as a percentage.

All metrics saved to accuracy_report.json and visualised as a dashboard.
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def measure_depth_consistency(
    poses_data: dict,
    depths_dir: str,
    n_pairs: int = 20,
) -> dict:
    """
    Measure depth consistency across frame pairs.

    Strategy:
      For N random pairs of nearby frames (i, j):
        1. Take 100 random pixels from frame i with valid depth
        2. Back-project to 3D using frame i's pose
        3. Project those 3D points into frame j
        4. Compare projected depth with frame j's actual depth
        5. Record the absolute depth error in metres

    Returns
    -------
    dict with:
        mean_error_m    : mean absolute depth error in metres
        median_error_m  : median absolute depth error in metres
        pct_under_10cm  : % of points with error < 0.10m
        pct_under_20cm  : % of points with error < 0.20m
        score           : 0-100 quality score
    """
    import cv2

    poses   = poses_data["poses"]
    K       = np.array(poses_data["K"],  dtype=np.float64)
    W       = poses_data["image_w"]
    H       = poses_data["image_h"]
    scale   = poses_data["scale"]
    N       = poses_data["num_frames"]

    fx, fy  = K[0,0], K[1,1]
    cx, cy  = K[0,2], K[1,2]

    all_errors = []

    # Select random pairs of nearby frames
    rng        = np.random.default_rng(42)
    pair_steps = rng.integers(1, min(5, N//4), size=n_pairs)
    pair_starts = rng.integers(0, N - pair_steps - 1, size=n_pairs)

    for pair_idx in range(n_pairs):
        i = int(pair_starts[pair_idx])
        j = int(pair_starts[pair_idx] + pair_steps[pair_idx])

        depth_i_path = os.path.join(depths_dir, f"depth_{i:04d}.npy")
        depth_j_path = os.path.join(depths_dir, f"depth_{j:04d}.npy")

        if not (os.path.exists(depth_i_path) and
                os.path.exists(depth_j_path)):
            continue

        depth_i = np.load(depth_i_path).astype(np.float64) * scale
        depth_j = np.load(depth_j_path).astype(np.float64) * scale

        if depth_i.shape != (H, W):
            depth_i = cv2.resize(depth_i, (W, H))
        if depth_j.shape != (H, W):
            depth_j = cv2.resize(depth_j, (W, H))

        pose_i = np.array(poses[i]["cam_to_world"], dtype=np.float64)
        pose_j = np.array(poses[j]["cam_to_world"], dtype=np.float64)

        # World-to-camera for frame j
        T_j_inv = np.linalg.inv(pose_j)

        # Sample valid pixels from frame i
        valid_mask = (depth_i > 0.1) & (depth_i < 6.0)
        valid_px   = np.where(valid_mask)

        if len(valid_px[0]) < 50:
            continue

        n_sample = min(100, len(valid_px[0]))
        sample   = rng.choice(len(valid_px[0]), n_sample, replace=False)
        v_s      = valid_px[0][sample]
        u_s      = valid_px[1][sample]
        d_s      = depth_i[v_s, u_s]

        # Back-project to 3D world
        x_cam = (u_s - cx) / fx * d_s
        y_cam = (v_s - cy) / fy * d_s
        z_cam = d_s
        ones  = np.ones_like(z_cam)

        pts_cam_i   = np.stack([x_cam, y_cam, z_cam, ones])
        pts_world   = (pose_i @ pts_cam_i)[:3]  # (3, N)

        # Project into frame j
        pts_cam_j   = T_j_inv[:3, :3] @ pts_world + \
                      T_j_inv[:3, 3:4]
        z_proj      = pts_cam_j[2]
        u_proj      = fx * pts_cam_j[0] / (z_proj + 1e-8) + cx
        v_proj      = fy * pts_cam_j[1] / (z_proj + 1e-8) + cy

        # Check which projected points are within frame j bounds
        in_bounds = (
            (u_proj >= 0) & (u_proj < W) &
            (v_proj >= 0) & (v_proj < H) &
            (z_proj > 0.05)
        )

        if in_bounds.sum() < 10:
            continue

        u_valid = u_proj[in_bounds].astype(int)
        v_valid = v_proj[in_bounds].astype(int)
        z_valid = z_proj[in_bounds]

        # Get actual depth at projected locations in frame j
        depth_j_at_proj = depth_j[
            np.clip(v_valid, 0, H-1),
            np.clip(u_valid, 0, W-1)
        ]

        # Compute absolute depth error
        valid_depth = depth_j_at_proj > 0.05
        if valid_depth.sum() < 5:
            continue

        errors = np.abs(z_valid[valid_depth] -
                        depth_j_at_proj[valid_depth])
        all_errors.extend(errors.tolist())

    if not all_errors:
        return {
            "mean_error_m"   : 0.0,
            "median_error_m" : 0.0,
            "pct_under_10cm" : 0.0,
            "pct_under_20cm" : 0.0,
            "score"          : 0.0,
            "n_measurements" : 0,
        }

    errors_arr    = np.array(all_errors)
    mean_err      = float(errors_arr.mean())
    median_err    = float(np.median(errors_arr))
    pct_10        = float((errors_arr < 0.10).mean() * 100)
    pct_20        = float((errors_arr < 0.20).mean() * 100)

    # Score: 100 if median error < 5cm, 0 if > 50cm
    score = float(np.clip(
        100 * (1 - (median_err - 0.05) / (0.50 - 0.05)), 0, 100
    ))

    return {
        "mean_error_m"   : round(mean_err,    4),
        "median_error_m" : round(median_err,  4),
        "pct_under_10cm" : round(pct_10,      1),
        "pct_under_20cm" : round(pct_20,      1),
        "score"          : round(score,        1),
        "n_measurements" : len(all_errors),
    }


def measure_pose_quality(poses_data: dict) -> dict:
    """
    Measure camera trajectory quality.

    Metrics:
      - smoothness: are consecutive poses similar? (no sudden jumps)
      - pnp_success_rate: % of frames solved by PnP (vs fallback)
      - trajectory_length: total distance camera travelled

    Returns 0-100 score.
    """
    poses = poses_data["poses"]
    N     = len(poses)

    positions = np.array([
        np.array(p["cam_to_world"])[:3, 3]
        for p in poses
    ])

    # Step distances between consecutive frames
    step_dists = np.linalg.norm(
        np.diff(positions, axis=0), axis=1
    )

    traj_length  = float(step_dists.sum())
    mean_step    = float(step_dists.mean()) if len(step_dists) else 0
    max_step     = float(step_dists.max())  if len(step_dists) else 0

    # Smoothness: penalise large jumps
    # A jump > 10× the mean step is suspicious
    jump_threshold = mean_step * 10
    n_jumps        = int((step_dists > jump_threshold).sum())
    smoothness     = float(
        max(0, 1 - n_jumps / max(N - 1, 1)) * 100
    )

    pnp_pct = float(poses_data.get("pnp_success_pct", 0))

    # Combined score
    score = float(0.6 * pnp_pct + 0.4 * smoothness)

    return {
        "pnp_success_pct"  : pnp_pct,
        "smoothness_score" : round(smoothness,    1),
        "trajectory_length": round(traj_length,   3),
        "mean_step_m"      : round(mean_step,      4),
        "max_step_m"       : round(max_step,       4),
        "n_trajectory_jumps": n_jumps,
        "score"            : round(score,          1),
    }


def measure_query_quality(
    query_results: list,
) -> dict:
    """
    Measure quality of query engine results.

    Takes a list of query result dicts from QueryEngine.query()
    and computes aggregate accuracy metrics.
    """
    if not query_results:
        return {"mean_confidence": 0, "mean_compactness": 0,
                "mean_precision": 0, "score": 0, "per_query": []}

    per_query = []
    for r in query_results:
        per_query.append({
            "query"       : r["query"],
            "confidence"  : r["confidence"],
            "compactness" : r["compactness"],
            "precision_at_k": r["precision_at_k"],
            "top_k"       : r["top_k"],
        })

    confs      = [r["confidence"]    for r in query_results]
    compacts   = [r["compactness"]   for r in query_results]
    precisions = [r["precision_at_k"] for r in query_results]

    mean_conf    = float(np.mean(confs))
    mean_compact = float(np.mean(compacts))
    mean_prec    = float(np.mean(precisions))

    # Score: weighted average
    score = float(
        0.4 * mean_conf +
        0.3 * mean_compact * 100 +
        0.3 * mean_prec    * 100
    )

    return {
        "mean_confidence"   : round(mean_conf,    1),
        "mean_compactness"  : round(mean_compact, 3),
        "mean_precision_at_k": round(mean_prec,   3),
        "score"             : round(score,         1),
        "per_query"         : per_query,
    }


def measure_embedding_coverage(npz_path: str) -> dict:
    """
    Measure what fraction of the point cloud has CLIP embeddings.
    """
    data         = np.load(npz_path, allow_pickle=True)
    mask_ids     = data["mask_ids"]
    n_masks      = len(data["mask_embeddings"])
    M            = len(mask_ids)
    n_assigned   = int((mask_ids >= 0).sum())
    coverage_pct = float(n_assigned / M * 100)

    # Score: 100 if >80% covered, 0 if <5%
    score = float(np.clip(
        (coverage_pct - 5) / (80 - 5) * 100, 0, 100
    ))

    return {
        "total_points"   : M,
        "assigned_points": n_assigned,
        "coverage_pct"   : round(coverage_pct, 1),
        "n_mask_regions" : n_masks,
        "score"          : round(score, 1),
    }


def run_full_accuracy_report(
    poses_path      : str,
    depths_dir      : str,
    npz_path        : str,
    query_results   : list,
    output_path     : str,
) -> dict:
    """
    Run all four accuracy measurements and save a unified report.

    Parameters
    ----------
    poses_path    : path to poses.json
    depths_dir    : path to depths/ directory
    npz_path      : path to pointcloud_query.npz
    query_results : list of results from QueryEngine.multi_query()
    output_path   : where to save accuracy_report.json

    Returns
    -------
    report : dict with all metrics and an overall score
    """
    print("Running accuracy measurements...")

    with open(poses_path) as f:
        poses_data = json.load(f)

    # 1. Depth consistency
    print("  [1/4] Depth consistency...")
    depth_metrics = measure_depth_consistency(
        poses_data, depths_dir, n_pairs=20
    )
    print(f"        median error: {depth_metrics['median_error_m']*100:.1f}cm  "
          f"score: {depth_metrics['score']:.1f}/100")

    # 2. Pose quality
    print("  [2/4] Pose quality...")
    pose_metrics = measure_pose_quality(poses_data)
    print(f"        PnP: {pose_metrics['pnp_success_pct']}%  "
          f"score: {pose_metrics['score']:.1f}/100")

    # 3. Query quality
    print("  [3/4] Query quality...")
    query_metrics = measure_query_quality(query_results)
    print(f"        mean conf: {query_metrics['mean_confidence']:.1f}%  "
          f"score: {query_metrics['score']:.1f}/100")

    # 4. Embedding coverage
    print("  [4/4] Embedding coverage...")
    embed_metrics = measure_embedding_coverage(npz_path)
    print(f"        coverage: {embed_metrics['coverage_pct']:.1f}%  "
          f"score: {embed_metrics['score']:.1f}/100")

    # Overall score — weighted average
    overall = float(
        0.25 * depth_metrics["score"] +
        0.25 * pose_metrics["score"]  +
        0.35 * query_metrics["score"] +
        0.15 * embed_metrics["score"]
    )

    report = {
        "overall_score"     : round(overall, 1),
        "depth_consistency" : depth_metrics,
        "pose_quality"      : pose_metrics,
        "query_quality"     : query_metrics,
        "embedding_coverage": embed_metrics,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*45}")
    print(f"ACCURACY REPORT")
    print(f"{'='*45}")
    print(f"  Depth consistency : {depth_metrics['score']:5.1f}/100")
    print(f"  Pose quality      : {pose_metrics['score']:5.1f}/100")
    print(f"  Query quality     : {query_metrics['score']:5.1f}/100")
    print(f"  Embedding coverage: {embed_metrics['score']:5.1f}/100")
    print(f"{'='*45}")
    print(f"  OVERALL SCORE     : {overall:5.1f}/100")
    print(f"{'='*45}")
    print(f"\n✓ Report saved: {output_path}")

    return report