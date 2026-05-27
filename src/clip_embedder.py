"""
clip_embedder.py
----------------
Wraps OpenAI CLIP ViT-B/32 for two purposes:

  1. Image embedding  — embed a masked region of a frame into 512-dim space
  2. Text embedding   — embed a natural language query into the same 512-dim space

Because both image and text embeddings live in the same vector space,
cosine similarity between a text query and an image region embedding
tells us how well that region matches the query.

This is the core mechanism behind language-queryable 3D:
  "where is the chair?" 
  → embed "chair" as text vector
  → find all 3D points whose image embedding has high cosine similarity
  → highlight those points

Design decisions:
  - ViT-B/32 is 150MB and runs in ~80ms on CPU — acceptable for offline processing
  - We embed the masked crop, not the full image — gives much more specific embeddings
  - We also embed the full image as context and average with the crop embedding
    to improve robustness on small or ambiguous regions
  - All embeddings are L2-normalised before storage so cosine similarity
    reduces to a simple dot product (faster at query time)
"""

import numpy as np
import cv2
import torch
import clip
from PIL import Image as PILImage
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLIP_MODEL, CLIP_DIM


class CLIPEmbedder:
    """
    CLIP ViT-B/32 wrapper for image region and text embedding.
    Load once, call embed_masked_region() and embed_text() as needed.
    """

    # Indoor scene vocabulary for zero-shot labelling
    # Used by label_masks() to assign text labels to detected regions
    INDOOR_VOCAB = [
        "floor", "wall", "ceiling", "door", "window",
        "chair", "table", "desk", "sofa", "couch",
        "bed", "pillow", "blanket", "shelf", "bookshelf",
        "monitor", "television", "laptop", "keyboard", "mouse",
        "lamp", "light", "curtain", "cabinet", "drawer",
        "plant", "bottle", "cup", "bowl", "box",
        "bag", "backpack", "clothes", "shoes",
        "staircase", "railing", "column", "beam",
        "robot", "machine", "tool", "equipment", "workbench",
        "corridor", "hallway", "empty space", "obstacle",
        "navigable floor", "blocked path",
    ]

    def __init__(self, device: str = None):
        """
        Load CLIP ViT-B/32.

        Parameters
        ----------
        device : "cuda", "cpu", or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading CLIP {CLIP_MODEL} on {device}...")
        print(f"  (~150MB download on first run, then cached)")

        self.model, self.preprocess = clip.load(CLIP_MODEL, device=device)
        self.model.eval()

        # Pre-compute and cache text embeddings for the indoor vocabulary
        # This is done once at load time so query() is fast
        print(f"  Pre-computing vocabulary embeddings "
              f"({len(self.INDOOR_VOCAB)} labels)...")
        self._vocab_embeddings = self._precompute_vocab()

        print(f"✓ CLIP embedder ready")
        print(f"  Embedding dim : {CLIP_DIM}")
        print(f"  Vocabulary    : {len(self.INDOOR_VOCAB)} indoor labels")

    def _precompute_vocab(self) -> np.ndarray:
        """
        Pre-compute CLIP text embeddings for the indoor vocabulary.

        Returns
        -------
        embeddings : (V, 512) float32 — one embedding per vocab label
                     All L2-normalised
        """
        with torch.no_grad():
            tokens = clip.tokenize(
                [f"a photo of a {label}" for label in self.INDOOR_VOCAB]
            ).to(self.device)

            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy().astype(np.float32)

    def embed_image_region(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Embed a full RGB image crop into CLIP space.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB numpy array

        Returns
        -------
        embedding : (512,) float32 L2-normalised CLIP embedding
                    or None if the image is too small to embed
        """
        h, w = image_rgb.shape[:2]
        if h < 8 or w < 8:
            return None

        pil_img = PILImage.fromarray(image_rgb)

        with torch.no_grad():
            tensor    = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().astype(np.float32).squeeze(0)

    def embed_masked_region(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        context_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Embed a masked region of an image.

        Strategy:
          1. Extract the tight bounding box crop around the mask
          2. Zero out pixels outside the mask (black background)
          3. Embed the masked crop — gives a region-specific embedding
          4. Also embed the full image as global context
          5. Return: (1 - context_weight) * crop_emb + context_weight * full_emb
             This blends specific region info with scene-level context

        Parameters
        ----------
        image_rgb      : (H, W, 3) uint8 RGB image
        mask           : (H, W) bool — True = pixels belonging to this region
        context_weight : how much weight to give the full image context
                         0.0 = crop only, 1.0 = full image only
                         0.3 = 70% region-specific, 30% global context

        Returns
        -------
        embedding : (512,) float32 L2-normalised, or None if region too small
        """
        if mask.sum() < 50:
            return None

        # Get bounding box of the mask
        rows    = np.where(mask.any(axis=1))[0]
        cols    = np.where(mask.any(axis=0))[0]
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()

        # Skip if bounding box is too small
        if (r_max - r_min) < 8 or (c_max - c_min) < 8:
            return None

        # Extract and mask the crop
        crop      = image_rgb[r_min:r_max+1, c_min:c_max+1].copy()
        mask_crop = mask[r_min:r_max+1, c_min:c_max+1]

        # Zero out pixels outside the mask
        # This forces CLIP to focus on the region, not background
        masked_crop = crop.copy()
        masked_crop[~mask_crop] = 0

        # Embed the masked crop
        crop_embedding = self.embed_image_region(masked_crop)
        if crop_embedding is None:
            return None

        # Embed full image for context
        full_embedding = self.embed_image_region(image_rgb)
        if full_embedding is None:
            return crop_embedding

        # Blend crop and context embeddings
        blended = ((1.0 - context_weight) * crop_embedding +
                   context_weight         * full_embedding)

        # Re-normalise after blending
        norm = np.linalg.norm(blended)
        if norm < 1e-8:
            return crop_embedding

        return (blended / norm).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a natural language query into CLIP space.

        Parameters
        ----------
        text : query string, e.g. "where is the chair?"
               or "navigable floor space"

        Returns
        -------
        embedding : (512,) float32 L2-normalised CLIP text embedding
        """
        # Clean and format the query
        text = text.strip().lower()

        # Add "a photo of" prefix — improves CLIP zero-shot performance
        # as the model was trained with this kind of prompt
        if not text.startswith("a photo of"):
            prompt = f"a photo of {text}"
        else:
            prompt = text

        with torch.no_grad():
            tokens    = clip.tokenize([prompt]).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().astype(np.float32).squeeze(0)

    def label_mask(self, image_rgb: np.ndarray, mask: np.ndarray) -> dict:
        """
        Assign the best matching vocabulary label to a masked region.

        Uses zero-shot CLIP classification:
          - Embed the masked region as an image
          - Compare against all pre-computed vocabulary embeddings
          - Return the top-3 matches with confidence scores

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB image
        mask      : (H, W) bool mask

        Returns
        -------
        result : dict with keys:
            "label"       : str — best matching label
            "confidence"  : float — cosine similarity (0–1)
            "top3"        : list of (label, score) tuples
        """
        embedding = self.embed_masked_region(image_rgb, mask)
        if embedding is None:
            return {"label": "unknown", "confidence": 0.0, "top3": []}

        # Cosine similarity = dot product (embeddings are L2-normalised)
        similarities = self._vocab_embeddings @ embedding  # (V,)

        # Get top 3
        top3_idx = np.argsort(similarities)[::-1][:3]
        top3     = [
            (self.INDOOR_VOCAB[i], float(similarities[i]))
            for i in top3_idx
        ]

        return {
            "label"      : top3[0][0],
            "confidence" : float(top3[0][1]),
            "top3"       : top3,
        }

    def label_all_masks(
        self,
        image_rgb: np.ndarray,
        masks: list,
    ) -> list:
        """
        Label all masks in a frame.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB image
        masks     : list of (H, W) bool masks

        Returns
        -------
        labels : list of result dicts (same order as masks)
        """
        return [self.label_mask(image_rgb, mask) for mask in masks]

    def batch_embed_texts(self, texts: list) -> np.ndarray:
        """
        Embed multiple text strings at once.
        More efficient than calling embed_text() in a loop.

        Parameters
        ----------
        texts : list of strings

        Returns
        -------
        embeddings : (N, 512) float32 L2-normalised embeddings
        """
        prompts = [
            f"a photo of {t}" if not t.startswith("a photo of") else t
            for t in texts
        ]

        with torch.no_grad():
            tokens     = clip.tokenize(prompts, truncate=True).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy().astype(np.float32)