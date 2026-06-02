"""
pipeline.py
-----------
Single entry point that orchestrates the entire pipeline.

Call run_pipeline(video_path, output_dir) and get back
a fully queryable 3D scene.

This is what the Gradio app calls internally.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NUM_FRAMES, MAX_SIDE_PX, SHARPNESS_THRESHOLD, MIN_FRAMES,
    NEAR_ANCHOR_M, VOXEL_SIZE,
)


def extract_frames_inline(
    video_path: str,
    output_dir: str,
    num_frames: int = NUM_FRAMES,
    max_side: int   = MAX_SIDE_PX,
    sharpness: float = SHARPNESS_THRESHOLD,
    min_frames: int  = MIN_FRAMES,
) -> dict:
    """Extract frames from video — inline version for pipeline use."""
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "frames_metadata.json")

    # Return cached result if already done
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        if meta.get("frames_kept", 0) >= min_frames:
            return meta

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0

    indices = np.linspace(0, total-1, min(num_frames, total),
                          dtype=int).tolist()
    indices = list(dict.fromkeys(indices))

    candidates = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        candidates.append((idx, score, frame))
    cap.release()

    threshold = sharpness
    while True:
        kept = [(i,s,f) for i,s,f in candidates if s >= threshold]
        if len(kept) >= min_frames or threshold <= 5.0:
            break
        threshold *= 0.75
    if len(kept) < min_frames:
        kept = candidates

    frame_files = []
    sharpness_d = {}
    for i, (_, score, bgr) in enumerate(kept):
        h, w  = bgr.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)),
                             interpolation=cv2.INTER_AREA)
        fname = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), bgr)
        frame_files.append(fname)
        sharpness_d[fname] = round(score, 2)

    meta = {
        "video_path" : video_path,
        "fps"        : round(fps, 3),
        "duration_s" : round(dur, 2),
        "frames_kept": len(frame_files),
        "frame_files": frame_files,
        "sharpness"  : sharpness_d,
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


class Pipeline:
    """
    Orchestrates the full 3D scene understanding pipeline.

    Usage:
        pipeline = Pipeline(output_dir="output/my_scene")
        result   = pipeline.run(video_path="room.mp4")
        answer   = pipeline.query("where is the chair?")
    """

    def __init__(
        self,
        output_dir    : str,
        fastsam_ckpt  : str,
        groq_api_key  : str = "",
        hf_api_token  : str = "",
        device        : str = None,
    ):
        """
        Parameters
        ----------
        output_dir   : where to save all outputs
        fastsam_ckpt : path to FastSAM-s.pt checkpoint
        groq_api_key : optional Groq API key for VLM descriptions
        hf_api_token : optional HuggingFace token (fallback VLM)
        device       : "cuda", "cpu", or None (auto)
        """
        import torch

        self.output_dir   = output_dir
        self.fastsam_ckpt = fastsam_ckpt
        self.device       = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        os.makedirs(output_dir, exist_ok=True)

        # Set API keys in environment
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        if hf_api_token:
            os.environ["HF_API_TOKEN"] = hf_api_token

        # Output paths
        self.paths = {
            "frames_dir"   : os.path.join(output_dir, "frames"),
            "depths_dir"   : os.path.join(output_dir, "depths"),
            "poses_json"   : os.path.join(output_dir, "poses.json"),
            "rgb_ply"      : os.path.join(output_dir, "pointcloud_rgb.ply"),
            "query_npz"    : os.path.join(output_dir, "pointcloud_query.npz"),
            "scene_desc"   : os.path.join(output_dir, "scene_description.json"),
            "accuracy_json": os.path.join(output_dir, "accuracy_report.json"),
            "metadata"     : os.path.join(output_dir, "frames",
                                          "frames_metadata.json"),
        }

        # State — loaded lazily
        self._engine   = None
        self._embedder = None

        print(f"Pipeline initialised")
        print(f"  Output dir : {output_dir}")
        print(f"  Device     : {self.device}")

    def run(
        self,
        video_path    : str,
        progress_cb   = None,
        keyframe_step : int = 5,
    ) -> dict:
        """
        Run the full pipeline on a video file.

        Parameters
        ----------
        video_path    : path to input video
        progress_cb   : optional callback(stage, pct, message)
                        called at each stage for UI progress updates
        keyframe_step : process every Nth frame for CLIP embedding

        Returns
        -------
        result : dict with paths to all outputs and timing info
        """
        t_start = time.time()

        def progress(stage, pct, msg):
            elapsed = time.time() - t_start
            print(f"  [{pct:3.0f}%] {stage}: {msg} ({elapsed:.1f}s)")
            if progress_cb:
                progress_cb(stage, pct, msg)

        progress("Setup", 0, "Starting pipeline")

        # ── Stage 1: Frame extraction ─────────────────────────────────────────
        progress("Frames", 5, "Extracting frames from video")
        os.makedirs(self.paths["frames_dir"], exist_ok=True)

        meta = extract_frames_inline(
            video_path = video_path,
            output_dir = self.paths["frames_dir"],
        )
        frame_files = meta["frame_files"]
        frame_paths = [
            os.path.join(self.paths["frames_dir"], f)
            for f in frame_files
        ]
        progress("Frames", 10,
                 f"{meta['frames_kept']} frames extracted")

        # ── Stage 2: Depth estimation ─────────────────────────────────────────
        progress("Depth", 15, "Loading Depth-Anything V2")
        from depth_estimator import DepthEstimator, compute_global_scale
        depth_est = DepthEstimator(device=self.device)

        progress("Depth", 20, "Estimating depth for all frames")
        os.makedirs(self.paths["depths_dir"], exist_ok=True)
        depth_stats = depth_est.estimate_batch(
            frame_paths = frame_paths,
            output_dir  = self.paths["depths_dir"],
            show_progress = False,
        )
        global_scale = compute_global_scale(
            depth_stats, near_anchor_m=NEAR_ANCHOR_M
        )

        del depth_est
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        progress("Depth", 30,
                 f"Scale={global_scale:.4f}")

        # ── Stage 3: Pose estimation ──────────────────────────────────────────
        progress("Poses", 32, "Estimating camera poses")

        if os.path.exists(self.paths["poses_json"]):
            with open(self.paths["poses_json"]) as f:
                poses_data = json.load(f)
            progress("Poses", 40,
                     f"Loaded cached poses "
                     f"({poses_data['pnp_success_pct']}% PnP)")
        else:
            sample_img = cv2.imread(frame_paths[0])
            H, W = sample_img.shape[:2]

            from pose_estimator import PoseEstimator
            pose_est   = PoseEstimator(image_w=W, image_h=H)
            poses_data = pose_est.estimate_poses(
                frame_paths  = frame_paths,
                depths_dir   = self.paths["depths_dir"],
                global_scale = global_scale,
            )
            poses_data["scale"] = float(global_scale)
            pose_est.save_poses(poses_data, self.paths["poses_json"])
            del pose_est
            gc.collect()
            progress("Poses", 40,
                     f"{poses_data['pnp_success_pct']}% PnP success")

        # ── Stage 4: Point cloud fusion ───────────────────────────────────────
        progress("Cloud", 42, "Building RGB point cloud")
        import open3d as o3d
        from cloud_builder import CloudBuilder

        if os.path.exists(self.paths["rgb_ply"]):
            pcd = o3d.io.read_point_cloud(self.paths["rgb_ply"])
            progress("Cloud", 50,
                     f"Loaded cached cloud "
                     f"({len(pcd.points):,} pts)")
        else:
            builder = CloudBuilder(poses_data)
            pcd     = builder.build_rgb_cloud(
                frame_paths = frame_paths,
                depths_dir  = self.paths["depths_dir"],
                output_dir  = self.output_dir,
                batch_size  = 10,
            )
            del builder
            gc.collect()
            progress("Cloud", 50,
                     f"{len(pcd.points):,} points")

        # ── Stage 5: VLM scene description ───────────────────────────────────
        progress("VLM", 52, "Generating scene description")
        from vlm_describer import VLMDescriber

        if os.path.exists(self.paths["scene_desc"]):
            with open(self.paths["scene_desc"]) as f:
                scene_desc = json.load(f)
            progress("VLM", 58,
                     f"Loaded cached: "
                     f"{len(scene_desc.get('objects',[]))} objects")
        else:
            describer  = VLMDescriber()
            scene_desc = describer.describe_scene(
                frame_paths = frame_paths,
                n_keyframes = 5,
                output_path = self.paths["scene_desc"],
            )
            del describer
            gc.collect()
            progress("VLM", 58,
                     f"{len(scene_desc.get('objects',[]))} objects found")

        # ── Stage 6: CLIP embedding ───────────────────────────────────────────
        progress("CLIP", 60, "Embedding CLIP features")

        if os.path.exists(self.paths["query_npz"]):
            progress("CLIP", 80, "Loaded cached embeddings")
        else:
	    import open3d as o3d 
            from segmentor      import Segmentor
            from clip_embedder  import CLIPEmbedder
		
		# Reload pcd if not in scope (e.g. after cache hit in Stage 4)
            if not hasattr(locals(), 'pcd') or pcd is None:
                pcd = o3d.io.read_point_cloud(self.paths["rgb_ply"])

            pts_all  = np.asarray(pcd.points,  dtype=np.float32)
            cols_all = np.asarray(pcd.colors,  dtype=np.float32)
            pcd_ds   = pcd.voxel_down_sample(0.03)
            pts_ds   = np.asarray(pcd_ds.points,  dtype=np.float32)
            cols_ds  = np.asarray(pcd_ds.colors,  dtype=np.float32)
            M        = len(pts_ds)

            mask_ids       = np.full(M, -1, dtype=np.int32)
            all_mask_embs  = []
            all_mask_labs  = []
            all_mask_confs = []
            global_mid     = 0

            pcd_tree_o3d = o3d.geometry.PointCloud()
            pcd_tree_o3d.points = o3d.utility.Vector3dVector(
                pts_ds.astype(np.float64)
            )
            kd = o3d.geometry.KDTreeFlann(pcd_tree_o3d)

            fx = np.array(poses_data["K"])[0,0]
            fy = np.array(poses_data["K"])[1,1]
            cx = np.array(poses_data["K"])[0,2]
            cy = np.array(poses_data["K"])[1,2]

            kf_indices = list(range(0, len(frame_paths), keyframe_step))

            for ki, fidx in enumerate(kf_indices):
                progress("CLIP", 60 + int(20 * ki/len(kf_indices)),
                         f"Keyframe {ki+1}/{len(kf_indices)}")

                seg = Segmentor(checkpoint_path=self.fastsam_ckpt)
                img_bgr = cv2.imread(frame_paths[fidx])
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                H2, W2  = img_bgr.shape[:2]
                masks   = seg.segment(img_bgr)
                del seg; gc.collect()

                if not masks:
                    continue

                clip_emb = CLIPEmbedder(device=self.device)
                d_np = np.load(os.path.join(
                    self.paths["depths_dir"],
                    f"depth_{fidx:04d}.npy"
                )).astype(np.float64) * global_scale
                if d_np.shape != (H2, W2):
                    d_np = cv2.resize(d_np, (W2, H2))
                pose = np.array(
                    poses_data["poses"][fidx]["cam_to_world"],
                    dtype=np.float64
                )
                ug, vg = np.meshgrid(np.arange(W2), np.arange(H2))
                uf = ug.flatten().astype(np.float64)
                vf = vg.flatten().astype(np.float64)

                for msk in masks:
                    if msk.sum() < 100:
                        continue
                    emb = clip_emb.embed_masked_region(img_rgb, msk)
                    if emb is None:
                        continue
                    lbl = clip_emb.label_mask(img_rgb, msk)
                    mf  = msk.flatten()
                    dv  = d_np.flatten()[mf]
                    uv  = uf[mf]; vv = vf[mf]
                    ok  = (dv >= 0.1) & (dv <= 6.0) & np.isfinite(dv)
                    if ok.sum() < 10:
                        continue
                    xc = (uv[ok]-cx)/fx*dv[ok]
                    yc = (vv[ok]-cy)/fy*dv[ok]
                    zc = dv[ok]
                    pts_c = np.stack([xc,yc,zc,np.ones_like(zc)])
                    pts_w = (pose @ pts_c)[:3].T
                    step  = max(1, len(pts_w)//200)
                    for pt in pts_w[::step]:
                        _, nn, dsq = kd.search_knn_vector_3d(
                            pt.astype(np.float64), 1
                        )
                        if dsq[0]**0.5 < 0.15:
                            mask_ids[nn[0]] = global_mid
                    all_mask_embs.append(emb)
                    all_mask_labs.append(lbl["label"])
                    all_mask_confs.append(lbl["confidence"])
                    global_mid += 1

                del clip_emb; gc.collect()

            # Keep assigned + background
            assigned_idx   = np.where(mask_ids >= 0)[0]
            unassigned_idx = np.where(mask_ids < 0)[0]
            bg_n   = min(50_000, len(unassigned_idx))
            bg_idx = np.random.choice(unassigned_idx, bg_n,
                                      replace=False) if bg_n > 0 else []
            keep   = np.sort(
                np.concatenate([assigned_idx,
                                np.array(bg_idx, dtype=np.int32)])
            )

            np.savez_compressed(
                self.paths["query_npz"],
                points           = pts_ds[keep],
                colours          = cols_ds[keep],
                mask_ids         = mask_ids[keep],
                mask_embeddings  = np.array(all_mask_embs,
                                            dtype=np.float32),
                mask_labels      = np.array(all_mask_labs),
                mask_confidences = np.array(all_mask_confs,
                                            dtype=np.float32),
            )
            del pts_ds, cols_ds, mask_ids
            gc.collect()

        progress("CLIP", 80, "Embeddings ready")

        # ── Stage 7: Accuracy report ──────────────────────────────────────────
        progress("Accuracy", 82, "Computing accuracy metrics")
        from accuracy import run_full_accuracy_report

        # Load query engine for accuracy
        from query_engine  import QueryEngine
        from clip_embedder import CLIPEmbedder

        self._embedder = CLIPEmbedder(device=self.device)
        self._engine   = QueryEngine(
            clip_npz_path          = self.paths["query_npz"],
            scene_description_path = self.paths["scene_desc"],
        )

        test_queries  = ["floor", "wall", "navigable path",
                         "obstacle", "furniture"]
        query_results = self._engine.multi_query(
            test_queries, self._embedder
        )

        report = run_full_accuracy_report(
            poses_path    = self.paths["poses_json"],
            depths_dir    = self.paths["depths_dir"],
            npz_path      = self.paths["query_npz"],
            query_results = query_results,
            output_path   = self.paths["accuracy_json"],
        )

        t_total = time.time() - t_start
        progress("Done", 100,
                 f"Complete in {t_total:.1f}s — "
                 f"score={report['overall_score']}/100")

        return {
            "scene_description" : scene_desc,
            "accuracy_report"   : report,
            "output_paths"      : self.paths,
            "n_frames"          : meta["frames_kept"],
            "n_points"          : len(np.load(
                self.paths["query_npz"])["points"]),
            "n_masks"           : len(np.load(
                self.paths["query_npz"],
                allow_pickle=True)["mask_embeddings"]),
            "processing_time_s" : round(t_total, 1),
        }

    def query(self, text: str, top_k: int = 500) -> dict:
        """
        Query the scene after pipeline.run() has been called.

        Parameters
        ----------
        text  : natural language query
        top_k : number of points to return

        Returns
        -------
        result dict from QueryEngine.query()
        """
        if self._engine is None or self._embedder is None:
            raise RuntimeError(
                "Call pipeline.run(video_path) before query()"
            )
        return self._engine.query(text, self._embedder, top_k=top_k)

    def get_scene_summary(self) -> str:
        """Return a human-readable scene summary string."""
        if not os.path.exists(self.paths["scene_desc"]):
            return "No scene description available. Run pipeline first."

        with open(self.paths["scene_desc"]) as f:
            sd = json.load(f)

        objects  = [o["name"] for o in sd.get("objects", [])]
        layout   = sd.get("layout", "")
        nav      = sd.get("navigable_regions", [])
        obstacles = sd.get("obstacles", [])

        lines = [
            f"Layout   : {layout}",
            f"Objects  : {', '.join(objects[:10])}",
            f"Navigate : {', '.join(nav[:3])}",
            f"Obstacles: {', '.join(obstacles[:5])}",
        ]
        return "\n".join(lines)