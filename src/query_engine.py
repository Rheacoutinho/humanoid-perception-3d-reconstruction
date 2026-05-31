"""
query_engine.py
---------------
RAM-efficient language query engine.

Key insight: we do NOT store 512-dim CLIP embeddings per point.
Instead:
  - Keep embeddings only for unique mask regions (~500 masks, ~1MB)
  - Each point stores only a mask_id (int16) pointing to its mask
  - At query time: embed query text, compare against ~500 mask embeddings,
    find top matching masks, return all points belonging to those masks
  - RAM: 500 × 512 × 4 bytes = 1MB instead of 1.8M × 512 × 4 = 3.6GB
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import QUERY_TOP_K, QUERY_MIN_SIMILARITY


class QueryEngine:

    def __init__(
        self,
        clip_npz_path: str,
        scene_description_path: str = None,
    ):
        if not os.path.exists(clip_npz_path):
            raise FileNotFoundError(
                f"Query data not found: {clip_npz_path}"
            )

        print(f"Loading query engine data...")
        data = np.load(clip_npz_path, allow_pickle=True)

        self.points     = data["points"].astype(np.float32)   # (M, 3)
        self.colours    = data["colours"].astype(np.float32)  # (M, 3)
        self.mask_ids   = data["mask_ids"].astype(np.int32)   # (M,)

        # Small arrays — mask-level data
        self.mask_embeddings = data["mask_embeddings"].astype(np.float32)
        self.mask_labels     = data["mask_labels"].tolist()
        self.mask_confidences = data["mask_confidences"].astype(np.float32)

        self.M        = len(self.points)
        self.n_masks  = len(self.mask_embeddings)

        print(f"  Points      : {self.M:,}")
        print(f"  Mask regions: {self.n_masks}")
        print(f"  RAM for embs: "
              f"{self.mask_embeddings.nbytes/1e6:.1f} MB "
              f"(vs {self.M*512*4/1e6:.0f} MB per-point)")

        self.scene_description = None
        if scene_description_path and \
                os.path.exists(scene_description_path):
            with open(scene_description_path) as f:
                self.scene_description = json.load(f)
            print(f"  Scene objects: "
                  f"{len(self.scene_description.get('objects', []))}")

        print("✓ Query engine ready")

    def query(
        self,
        text_query: str,
        clip_embedder,
        top_k: int = None,
        min_similarity: float = None,
    ) -> dict:

        if top_k         is None: top_k         = QUERY_TOP_K
        if min_similarity is None: min_similarity = QUERY_MIN_SIMILARITY

        # Embed query text
        q_emb = clip_embedder.embed_text(text_query)  # (512,)

        # Compare against mask embeddings only (~500 masks)
        mask_sims = self.mask_embeddings @ q_emb  # (n_masks,)

        # Rank masks by similarity
        ranked     = np.argsort(mask_sims)[::-1]
        top_masks  = ranked[:20]  # top 20 matching masks

        # Collect all points belonging to top matching masks
        result_point_indices = []
        result_sims          = []

        for mask_id in top_masks:
            sim = float(mask_sims[mask_id])
            if sim < min_similarity:
                break
            pts_in_mask = np.where(self.mask_ids == mask_id)[0]
            result_point_indices.extend(pts_in_mask.tolist())
            result_sims.extend([sim] * len(pts_in_mask))

        if not result_point_indices:
            # Fallback: return top mask regardless of threshold
            best_mask    = int(ranked[0])
            pts_in_mask  = np.where(self.mask_ids == best_mask)[0]
            result_point_indices = pts_in_mask.tolist()
            result_sims  = [float(mask_sims[best_mask])] * len(pts_in_mask)

        # Cap at top_k
        if len(result_point_indices) > top_k:
            result_point_indices = result_point_indices[:top_k]
            result_sims          = result_sims[:top_k]

        idx_arr = np.array(result_point_indices, dtype=np.int32)
        sim_arr = np.array(result_sims,          dtype=np.float32)

        result_points  = self.points[idx_arr]
        result_colours = self.colours[idx_arr]

        # Bounding box
        bbox_min    = result_points.min(axis=0) if len(result_points) else np.zeros(3)
        bbox_max    = result_points.max(axis=0) if len(result_points) else np.zeros(3)
        bbox_centre = (bbox_min + bbox_max) / 2.0
        bbox_size   = bbox_max - bbox_min

        # Metrics
        mean_sim   = float(sim_arr.mean()) if len(sim_arr) else 0.0
        max_sim    = float(sim_arr.max())  if len(sim_arr) else 0.0
        confidence = float(
            np.clip((mean_sim - 0.15) / (0.40 - 0.15) * 100, 0, 100)
        )

        scene_size  = float(np.linalg.norm(
            self.points.max(axis=0) - self.points.min(axis=0)
        ))
        if len(result_points) > 1:
            centroid     = result_points.mean(axis=0)
            dists        = np.linalg.norm(result_points - centroid, axis=1)
            median_dist  = float(np.median(dists))
            compactness  = float(
                np.clip(1.0 - median_dist / (scene_size / 2.0 + 1e-8), 0, 1)
            )
            within       = (dists <= 2.0 * median_dist).sum()
            precision_at_k = float(within / len(result_points))
        else:
            compactness    = 0.0
            precision_at_k = 0.0

        vlm_context = self._get_vlm_context(text_query)

        # Per-point similarity for visualisation
        all_sims = np.full(self.M, -1.0, dtype=np.float32)
        all_sims[idx_arr] = sim_arr

        query_hash       = hash(text_query.lower().strip()) % 6
        highlight_colours = [
            [255,80,80],[80,200,80],[80,150,255],
            [255,200,50],[255,120,200],[120,220,220],
        ]

        return {
            "query"           : text_query,
            "top_k"           : len(idx_arr),
            "mean_similarity" : round(mean_sim, 4),
            "max_similarity"  : round(max_sim,  4),
            "confidence"      : round(confidence, 1),
            "points"          : result_points,
            "colours"         : result_colours,
            "highlight_colour": highlight_colours[query_hash],
            "bbox_min"        : bbox_min.tolist(),
            "bbox_max"        : bbox_max.tolist(),
            "bbox_centre"     : bbox_centre.tolist(),
            "bbox_size"       : bbox_size.tolist(),
            "compactness"     : round(compactness, 3),
            "precision_at_k"  : round(precision_at_k, 3),
            "vlm_context"     : vlm_context,
            "all_similarities": all_sims,
        }

    def _get_vlm_context(self, query: str) -> list:
        if not self.scene_description:
            return []
        query_words = set(query.lower().split())
        relevant    = []
        for obj in self.scene_description.get("objects", []):
            name = obj.get("name", "").lower()
            if query_words & set(name.split()):
                relevant.append(obj)
        nav_words = {"navigate","navigable","walk","path",
                     "floor","clear","safe","through"}
        if query_words & nav_words:
            for r in self.scene_description.get(
                "navigable_regions", []
            )[:3]:
                relevant.append({"name": r, "type": "navigable_region"})
        return relevant

    def build_highlighted_cloud(
        self,
        query_result: dict,
        highlight_alpha: float = 0.85,
    ) -> dict:

        all_sims = query_result["all_similarities"]
        h_colour = np.array(
            query_result["highlight_colour"], dtype=np.float32
        ) / 255.0

        vis_colours  = self.colours.copy() * 0.25
        result_mask  = all_sims >= QUERY_MIN_SIMILARITY

        if result_mask.sum() > 0:
            match_sims = all_sims[result_mask]
            sim_min    = match_sims.min()
            sim_max    = match_sims.max()
            intensity  = (
                (match_sims - sim_min) / (sim_max - sim_min + 1e-8)
            )
            match_idx  = np.where(result_mask)[0]
            for i, idx in enumerate(match_idx):
                blend = highlight_alpha * float(intensity[i])
                vis_colours[idx] = (
                    (1 - blend) * self.colours[idx] +
                    blend       * h_colour
                )

        return {
            "points"     : self.points,
            "colours"    : np.clip(vis_colours, 0, 1),
            "result_mask": result_mask,
        }

    def multi_query(self, queries: list, clip_embedder) -> list:
        results = []
        for q in queries:
            r = self.query(q, clip_embedder)
            results.append(r)
            print(f"  '{q}' → conf={r['confidence']:.1f}%  "
                  f"pts={r['top_k']:,}  "
                  f"compact={r['compactness']:.3f}")
        return results