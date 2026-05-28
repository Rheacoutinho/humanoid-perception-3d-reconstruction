"""
query_engine.py
---------------
Natural language query engine for the language-embedded 3D point cloud.

How it works:
  1. Load the CLIP-embedded point cloud (pointcloud_clip.npz)
  2. User types a query: "where is the chair?"
  3. Embed the query text using CLIP → 512-dim vector
  4. Compute cosine similarity between query vector and every point's embedding
  5. Return the top-k points with highest similarity
  6. Compute a 3D bounding box around the result region
  7. Return coloured point cloud with matching points highlighted

Why this is powerful for robotics:
  - No retraining needed — works on any query, any scene
  - Returns 3D coordinates directly usable for robot navigation
  - Bounding box gives the robot a target region to move toward
  - Confidence score tells the robot how certain the match is
  - Works in real-time: a single matrix multiply on 5M points
    takes <100ms on CPU

Accuracy metrics computed per query:
  - Mean similarity score of top-k results
  - Spatial compactness of results (are they clustered or scattered?)
  - Precision@k (what fraction of top-k are within the result cluster?)
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    QUERY_TOP_K,
    QUERY_MIN_SIMILARITY,
    CLIP_DIM,
)


class QueryEngine:
    """
    Language-queryable 3D point cloud search engine.
    """

    def __init__(
        self,
        clip_npz_path: str,
        scene_description_path: str = None,
    ):
        """
        Load the embedded point cloud and optional scene description.

        Parameters
        ----------
        clip_npz_path           : path to pointcloud_clip.npz
        scene_description_path  : path to scene_description.json (optional)
                                  Used to enrich query results with VLM context
        """
        if not os.path.exists(clip_npz_path):
            raise FileNotFoundError(
                f"CLIP point cloud not found: {clip_npz_path}\n"
                f"Run cloud_builder.embed_clip_features() first."
            )

        print(f"Loading CLIP point cloud: {clip_npz_path}")
        data = np.load(clip_npz_path)

        self.points       = data["points"].astype(np.float32)      # (M, 3)
        self.colours      = data["colours"].astype(np.float32)     # (M, 3)
        self.embeddings   = data["embeddings"].astype(np.float32)  # (M, 512)
        self.embed_counts = data["embed_counts"].astype(np.int32)  # (M,)

        self.M = len(self.points)

        # Only query points that actually have embeddings
        self.has_embedding = self.embed_counts > 0
        self.n_embedded    = self.has_embedding.sum()

        print(f"  Total points     : {self.M:,}")
        print(f"  Embedded points  : {self.n_embedded:,} "
              f"({self.n_embedded/self.M*100:.1f}%)")

        # Load scene description if available
        self.scene_description = None
        if scene_description_path and \
                os.path.exists(scene_description_path):
            with open(scene_description_path) as f:
                self.scene_description = json.load(f)
            n_obj = len(self.scene_description.get("objects", []))
            print(f"  Scene objects    : {n_obj}")

        print(f"✓ Query engine ready")

    def query(
        self,
        text_query: str,
        clip_embedder,
        top_k: int = None,
        min_similarity: float = None,
    ) -> dict:
        """
        Find 3D points matching a natural language query.

        Parameters
        ----------
        text_query     : natural language query string
                         e.g. "chair", "floor", "navigable path",
                              "obstacle blocking robot", "workbench"
        clip_embedder  : CLIPEmbedder instance for text encoding
        top_k          : return this many top matching points
                         None = use config default (500)
        min_similarity : minimum cosine similarity threshold
                         None = use config default (0.15)

        Returns
        -------
        result : dict with keys:
            "query"           : the original query string
            "top_k"           : number of results returned
            "mean_similarity" : mean cosine similarity of results
            "max_similarity"  : best match similarity score
            "confidence"      : 0-100 score for display
            "points"          : (K, 3) XYZ of matching points
            "colours"         : (K, 3) RGB of matching points
            "highlight_colour": suggested highlight colour [R, G, B]
            "bbox_min"        : (3,) bounding box minimum corner
            "bbox_max"        : (3,) bounding box maximum corner
            "bbox_centre"     : (3,) bounding box centre
            "bbox_size"       : (3,) bounding box dimensions in metres
            "compactness"     : 0-1 score (1=tight cluster, 0=scattered)
            "precision_at_k"  : fraction of top-k within result cluster
            "vlm_context"     : relevant VLM scene description objects
            "all_similarities": (M,) similarity scores for all points
                                (used for coloured visualisation)
        """
        if top_k is None:
            top_k = QUERY_TOP_K
        if min_similarity is None:
            min_similarity = QUERY_MIN_SIMILARITY

        # ── 1. Embed the query text ───────────────────────────────────────────
        query_embedding = clip_embedder.embed_text(text_query)  # (512,)

        # ── 2. Compute cosine similarity with all embedded points ─────────────
        # embeddings are already L2-normalised, so dot product = cosine sim
        # Only compute for points that have embeddings
        similarities = np.full(self.M, -1.0, dtype=np.float32)

        if self.n_embedded > 0:
            embedded_sims = (
                self.embeddings[self.has_embedding] @ query_embedding
            )  # (n_embedded,)
            similarities[self.has_embedding] = embedded_sims

        # ── 3. Get top-k results ──────────────────────────────────────────────
        # Only consider embedded points above the minimum threshold
        candidate_mask = (similarities >= min_similarity) & self.has_embedding
        n_candidates   = candidate_mask.sum()

        if n_candidates == 0:
            # No points above threshold — return top-k regardless
            candidate_mask = self.has_embedding
            n_candidates   = candidate_mask.sum()

        candidate_indices = np.where(candidate_mask)[0]
        candidate_sims    = similarities[candidate_indices]

        # Sort by similarity descending
        sorted_order    = np.argsort(candidate_sims)[::-1]
        top_k_actual    = min(top_k, len(sorted_order))
        top_k_order     = sorted_order[:top_k_actual]
        top_k_indices   = candidate_indices[top_k_order]
        top_k_sims      = candidate_sims[top_k_order]

        # ── 4. Extract matching points ────────────────────────────────────────
        result_points  = self.points[top_k_indices]    # (K, 3)
        result_colours = self.colours[top_k_indices]   # (K, 3)

        # ── 5. Compute bounding box ───────────────────────────────────────────
        bbox_min    = result_points.min(axis=0)
        bbox_max    = result_points.max(axis=0)
        bbox_centre = (bbox_min + bbox_max) / 2.0
        bbox_size   = bbox_max - bbox_min

        # ── 6. Compute accuracy metrics ───────────────────────────────────────
        mean_sim   = float(top_k_sims.mean())
        max_sim    = float(top_k_sims.max())

        # Confidence: scale mean similarity to 0-100
        # CLIP cosine similarities typically range 0.15-0.40
        # We map 0.15 → 0, 0.40 → 100
        confidence = float(
            np.clip((mean_sim - 0.15) / (0.40 - 0.15) * 100, 0, 100)
        )

        # Compactness: how tightly clustered are the results?
        # Compute as ratio of median distance to centroid vs scene size
        scene_size    = float(np.linalg.norm(
            self.points.max(axis=0) - self.points.min(axis=0)
        ))
        centroid      = result_points.mean(axis=0)
        dists_to_ctr  = np.linalg.norm(result_points - centroid, axis=1)
        median_dist   = float(np.median(dists_to_ctr))
        compactness   = float(
            np.clip(1.0 - (median_dist / (scene_size / 2.0 + 1e-8)), 0, 1)
        )

        # Precision@k: fraction of top-k within 2× median distance
        within_cluster  = (dists_to_ctr <= 2.0 * median_dist).sum()
        precision_at_k  = float(within_cluster / max(top_k_actual, 1))

        # ── 7. Get VLM context for this query ────────────────────────────────
        vlm_context = self._get_vlm_context(text_query)

        # ── 8. Suggest a highlight colour ────────────────────────────────────
        # Use a consistent colour per query based on hash
        # so repeated queries show the same colour
        query_hash      = hash(text_query.lower().strip()) % 6
        highlight_colours = [
            [255,  80,  80],   # red
            [ 80, 200,  80],   # green
            [ 80, 150, 255],   # blue
            [255, 200,  50],   # yellow
            [255, 120, 200],   # pink
            [120, 220, 220],   # cyan
        ]
        highlight_colour = highlight_colours[query_hash]

        result = {
            "query"           : text_query,
            "top_k"           : top_k_actual,
            "mean_similarity" : round(mean_sim, 4),
            "max_similarity"  : round(max_sim, 4),
            "confidence"      : round(confidence, 1),
            "points"          : result_points,
            "colours"         : result_colours,
            "highlight_colour": highlight_colour,
            "bbox_min"        : bbox_min.tolist(),
            "bbox_max"        : bbox_max.tolist(),
            "bbox_centre"     : bbox_centre.tolist(),
            "bbox_size"       : bbox_size.tolist(),
            "compactness"     : round(compactness, 3),
            "precision_at_k"  : round(precision_at_k, 3),
            "vlm_context"     : vlm_context,
            "all_similarities": similarities,
        }

        return result

    def _get_vlm_context(self, query: str) -> list:
        """
        Find VLM scene description objects relevant to the query.
        Simple keyword matching between query words and object names.
        """
        if not self.scene_description:
            return []

        query_words = set(query.lower().split())
        relevant    = []

        for obj in self.scene_description.get("objects", []):
            name       = obj.get("name", "").lower()
            name_words = set(name.split())

            # Check if any query word matches any object name word
            if query_words & name_words:
                relevant.append(obj)

        # Also check navigable regions if query mentions navigation
        nav_words = {"navigate", "navigable", "walk", "path",
                     "floor", "clear", "safe", "through"}
        if query_words & nav_words:
            for region in self.scene_description.get(
                "navigable_regions", []
            )[:3]:
                relevant.append({"name": region, "type": "navigable_region"})

        return relevant

    def build_highlighted_cloud(
        self,
        query_result: dict,
        highlight_alpha: float = 0.85,
    ) -> dict:
        """
        Build a point cloud coloured by query relevance.

        Background points are dimmed.
        Matching points are shown in the highlight colour,
        brighter for higher similarity scores.

        Parameters
        ----------
        query_result    : output of query()
        highlight_alpha : how strongly to colour matching points
                          0 = original colour, 1 = pure highlight colour

        Returns
        -------
        dict with:
            "points"        : (M, 3) all points
            "colours"       : (M, 3) relevance-coloured RGB in [0, 1]
            "result_mask"   : (M,) bool — True for top-k matching points
        """
        all_sims  = query_result["all_similarities"]  # (M,)
        h_colour  = np.array(
            query_result["highlight_colour"], dtype=np.float32
        ) / 255.0  # (3,) in [0, 1]

        # Start with original dimmed colours for all points
        vis_colours = self.colours.copy() * 0.25  # dim background

        # Find top-k indices
        top_k = query_result["top_k"]
        valid  = all_sims >= QUERY_MIN_SIMILARITY
        if valid.sum() < top_k:
            valid = np.ones(self.M, dtype=bool)

        top_k_indices = np.argsort(all_sims)[::-1][:top_k]
        result_mask   = np.zeros(self.M, dtype=bool)
        result_mask[top_k_indices] = True

        # Colour matching points by similarity strength
        if result_mask.sum() > 0:
            match_sims = all_sims[result_mask]

            # Normalise similarities to [0, 1] for colour intensity
            sim_min = match_sims.min()
            sim_max = match_sims.max()
            if sim_max > sim_min:
                intensity = (match_sims - sim_min) / (sim_max - sim_min)
            else:
                intensity = np.ones_like(match_sims)

            # Blend original colour with highlight colour
            orig_cols  = self.colours[result_mask]  # (K, 3)
            for i, (idx, intens) in enumerate(
                zip(np.where(result_mask)[0], intensity)
            ):
                blend = highlight_alpha * intens
                vis_colours[idx] = (
                    (1 - blend) * self.colours[idx] +
                    blend       * h_colour
                )

        return {
            "points"     : self.points,
            "colours"    : np.clip(vis_colours, 0, 1),
            "result_mask": result_mask,
        }

    def multi_query(
        self,
        queries: list,
        clip_embedder,
    ) -> list:
        """
        Run multiple queries at once and return all results.
        Useful for the Gradio app to show several queries simultaneously.

        Parameters
        ----------
        queries       : list of query strings
        clip_embedder : CLIPEmbedder instance

        Returns
        -------
        results : list of query result dicts (same order as queries)
        """
        results = []
        for q in queries:
            result = self.query(q, clip_embedder)
            results.append(result)
            print(f"  '{q}' → "
                  f"conf={result['confidence']:.1f}%  "
                  f"compact={result['compactness']:.3f}  "
                  f"bbox={[f'{x:.2f}m' for x in result['bbox_size']]}")
        return results

    def get_navigation_targets(self, clip_embedder) -> dict:
        """
        Automatically find key navigation-relevant regions.
        Called once per scene to pre-compute robot navigation targets.

        Returns dict mapping target name → 3D bounding box centre
        """
        nav_queries = [
            "navigable floor space",
            "obstacle blocking path",
            "clear corridor or walkway",
            "wall boundary",
            "doorway or exit",
        ]

        print("Computing navigation targets...")
        targets = {}

        for q in nav_queries:
            result = self.query(q, clip_embedder, top_k=300)
            if result["confidence"] > 10:
                targets[q] = {
                    "centre"    : result["bbox_centre"],
                    "size"      : result["bbox_size"],
                    "confidence": result["confidence"],
                }
                print(f"  ✓ '{q}' → "
                      f"centre={[f'{x:.2f}' for x in result['bbox_centre']]}  "
                      f"conf={result['confidence']:.1f}%")
            else:
                print(f"  ✗ '{q}' → low confidence ({result['confidence']:.1f}%)")

        return targets