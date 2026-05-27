"""
config.py
---------
Single source of truth for all configuration.
Change things here — nowhere else.
"""

import os

# ── VLM API ───────────────────────────────────────────────────────────────────
# Provider: "groq" or "huggingface"
# Get free Groq key at: https://console.groq.com (no credit card)
# Get free HF token at: https://huggingface.co/settings/tokens
VLM_PROVIDER    = "groq"
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
HF_API_TOKEN    = os.environ.get("HF_API_TOKEN", "")

# Groq vision model — free tier, open source
GROQ_VLM_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# HuggingFace fallback model
HF_VLM_MODEL    = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# ── Depth estimation ──────────────────────────────────────────────────────────
# Small = CPU-friendly (80MB), Large = GPU recommended (1.3GB)
DEPTH_MODEL     = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_INPUT_SIZE = 518       # native input size for DA2

# ── CLIP ──────────────────────────────────────────────────────────────────────
# ViT-B/32 = 150MB, fast on CPU, 512-dim embeddings
CLIP_MODEL      = "ViT-B/32"
CLIP_DIM        = 512

# ── Segmentation ─────────────────────────────────────────────────────────────
# FastSAM-s = 23MB, CPU-friendly
FASTSAM_MODEL   = "FastSAM-s.pt"
FASTSAM_CONF    = 0.4        # confidence threshold
FASTSAM_IOU     = 0.9        # IoU threshold for NMS

# ── Frame extraction ──────────────────────────────────────────────────────────
NUM_FRAMES          = 80
MAX_SIDE_PX         = 518
SHARPNESS_THRESHOLD = 30.0
MIN_FRAMES          = 40

# ── Point cloud ───────────────────────────────────────────────────────────────
VOXEL_SIZE          = 0.01   # 1cm voxel grid
MAX_DEPTH_M         = 6.0    # discard points beyond 6 metres
MIN_DEPTH_M         = 0.1    # discard points closer than 10cm
NEAR_ANCHOR_M       = 0.5    # assumed nearest depth for scale calibration
OUTLIER_NEIGHBORS   = 20
OUTLIER_STD_RATIO   = 2.0

# ── Pose estimation ───────────────────────────────────────────────────────────
ORB_FEATURES        = 3000
PNP_REPROJECTION_ERR = 4.0
PNP_CONFIDENCE      = 0.999
MIN_PNP_INLIERS     = 6

# ── Query engine ──────────────────────────────────────────────────────────────
QUERY_TOP_K         = 500    # return top 500 matching points per query
QUERY_MIN_SIMILARITY = 0.15  # cosine similarity threshold

# ── Accuracy ──────────────────────────────────────────────────────────────────
REPROJ_ERROR_GOOD   = 2.0    # pixels — good reprojection error threshold
DEPTH_CONSISTENCY_GOOD = 0.1 # metres — good depth consistency threshold

# ── Paths (relative to project root) ─────────────────────────────────────────
FRAMES_DIR          = "frames"
OUTPUT_DIR          = "output"
CHECKPOINTS_DIR     = "checkpoints"
DEPTHS_SUBDIR       = "depths"