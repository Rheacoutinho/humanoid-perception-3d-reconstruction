"""
vlm_describer.py
----------------
Uses a Vision Language Model to generate a structured scene description
from keyframes. This gives the system high-level semantic understanding
that CLIP alone cannot provide.

Why VLM description matters:
  CLIP can match "chair" to a region but cannot answer:
    - "How many chairs are there?"
    - "Is the path to the door clear?"
    - "What is on top of the table?"
    - "Which direction is the exit?"
  A VLM reads the actual scene and generates this structured understanding.

API options (all free, no credit card):
  1. Groq — llama-4-scout-17b (vision, fast LPU inference)
     Sign up: https://console.groq.com
  2. HuggingFace Inference API — Llama-3.2-11B-Vision
     Sign up: https://huggingface.co/settings/tokens
  3. Fallback — rule-based description from CLIP labels only
     Used automatically if no API key is provided

Output format (always JSON):
  {
    "objects": [
      {"name": "chair", "location": "left side", "navigable": false},
      {"name": "floor", "location": "centre", "navigable": true},
      ...
    ],
    "layout": "small room with desk on right, bed on left",
    "navigable_regions": ["centre floor", "doorway area"],
    "obstacles": ["chair near door", "box on floor"],
    "robot_notes": "clear path exists along left wall"
  }
"""

import os
import json
import base64
import re
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VLM_PROVIDER,
    GROQ_API_KEY,
    HF_API_TOKEN,
    GROQ_VLM_MODEL,
    HF_VLM_MODEL,
)

# ── Prompt used for all VLM calls ─────────────────────────────────────────────
SCENE_PROMPT = """You are analysing a frame from an indoor video for a humanoid robot navigation system.

Describe the scene in structured JSON format. Be specific about spatial locations.

Return ONLY valid JSON, no other text, no markdown, no backticks. Use this exact structure:
{
  "objects": [
    {"name": "object name", "location": "where in frame (left/right/centre/background/foreground)", "navigable": false}
  ],
  "layout": "one sentence describing the overall room layout",
  "navigable_regions": ["list of areas the robot could walk through"],
  "obstacles": ["list of objects blocking navigation"],
  "surfaces": ["list of visible surfaces like floor, wall, table top"],
  "robot_notes": "one sentence of advice for robot navigation in this scene"
}

Include 3-10 objects. Be concise. Focus on what matters for robot navigation."""


def _encode_image_to_base64(image_bgr: np.ndarray) -> str:
    """
    Encode a BGR numpy array to base64 JPEG string.
    Required format for sending images to vision APIs.
    """
    # Resize to max 512px to reduce API payload size
    h, w  = image_bgr.shape[:2]
    scale = min(1.0, 512 / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """
    Robustly parse JSON from VLM response.
    VLMs sometimes wrap JSON in markdown or add extra text.
    This handles all common failure modes.
    """
    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*",     "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Return empty structure if all parsing fails
    return {
        "objects"           : [],
        "layout"            : text[:200] if text else "unknown",
        "navigable_regions" : [],
        "obstacles"         : [],
        "surfaces"          : [],
        "robot_notes"       : "VLM response could not be parsed",
    }


def describe_with_groq(image_bgr: np.ndarray, api_key: str) -> dict:
    """
    Call Groq API with Llama 4 Scout Vision (free tier).

    Parameters
    ----------
    image_bgr : BGR frame to describe
    api_key   : Groq API key (from console.groq.com)

    Returns
    -------
    description : parsed JSON dict
    """
    from groq import Groq

    client    = Groq(api_key=api_key)
    b64_image = _encode_image_to_base64(image_bgr)

    response = client.chat.completions.create(
        model    = GROQ_VLM_MODEL,
        messages = [
            {
                "role"   : "user",
                "content": [
                    {
                        "type"      : "image_url",
                        "image_url" : {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": SCENE_PROMPT,
                    },
                ],
            }
        ],
        max_tokens  = 1024,
        temperature = 0.1,  # low temperature = more consistent structured output
    )

    raw_text = response.choices[0].message.content
    return _parse_json_response(raw_text)


def describe_with_huggingface(image_bgr: np.ndarray, hf_token: str) -> dict:
    """
    Call HuggingFace Inference API with Llama 3.2 11B Vision (free tier).

    Parameters
    ----------
    image_bgr : BGR frame to describe
    hf_token  : HuggingFace token (from huggingface.co/settings/tokens)

    Returns
    -------
    description : parsed JSON dict
    """
    import requests

    b64_image = _encode_image_to_base64(image_bgr)

    # HuggingFace inference API endpoint
    api_url = (
        f"https://api-inference.huggingface.co/models/{HF_VLM_MODEL}"
    )

    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": {
            "image"  : b64_image,
            "question": SCENE_PROMPT,
        },
        "parameters": {
            "max_new_tokens": 1024,
            "temperature"   : 0.1,
        },
    }

    response = requests.post(api_url, headers=headers, json=payload,
                             timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace API error {response.status_code}: "
            f"{response.text[:200]}"
        )

    result = response.json()

    # HF returns different formats depending on model
    if isinstance(result, list) and len(result) > 0:
        text = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        text = result.get("generated_text", str(result))
    else:
        text = str(result)

    return _parse_json_response(text)


def describe_with_fallback(
    image_bgr: np.ndarray,
    clip_labels: list = None,
) -> dict:
    """
    Rule-based fallback when no VLM API is available.
    Uses CLIP labels from the segmentor to build a basic description.

    Parameters
    ----------
    image_bgr   : BGR frame (used for basic colour/brightness analysis)
    clip_labels : list of label dicts from CLIPEmbedder.label_all_masks()

    Returns
    -------
    description : dict in the standard format
    """
    objects = []

    if clip_labels:
        # Count label occurrences
        label_counts = {}
        for lbl in clip_labels:
            name = lbl.get("label", "unknown")
            if lbl.get("confidence", 0) > 0.2:
                label_counts[name] = label_counts.get(name, 0) + 1

        # Build object list from CLIP labels
        locations = ["left side", "centre", "right side",
                     "background", "foreground", "top", "bottom"]
        for i, (name, count) in enumerate(label_counts.items()):
            objects.append({
                "name"      : name,
                "location"  : locations[i % len(locations)],
                "navigable" : name in ["floor", "navigable floor",
                                        "corridor", "hallway",
                                        "empty space"],
            })

    # Basic image analysis for navigability
    grey      = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w      = grey.shape
    # Bottom third of image is usually floor (navigable)
    floor_brightness = float(grey[2*h//3:, :].mean())

    navigable = ["floor area"] if floor_brightness > 50 else []
    surfaces  = ["floor", "wall"] if floor_brightness > 50 else ["wall"]

    return {
        "objects"           : objects,
        "layout"            : "Indoor scene — VLM not available, "
                              "using CLIP-based description",
        "navigable_regions" : navigable,
        "obstacles"         : [o["name"] for o in objects
                                if not o.get("navigable", False)][:3],
        "surfaces"          : surfaces,
        "robot_notes"       : "No VLM available — navigation based on "
                              "CLIP labels only",
    }


class VLMDescriber:
    """
    Generates structured scene descriptions from video frames.
    Automatically selects the best available VLM provider.
    """

    def __init__(self):
        """
        Detect which VLM provider is available based on env vars.
        Falls back gracefully if no keys are set.
        """
        self.groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        self.hf_token = os.environ.get("HF_API_TOKEN", "").strip()

        # Determine provider
        if self.groq_key and len(self.groq_key) > 10:
            self.provider = "groq"
            print(f"✓ VLM provider: Groq ({GROQ_VLM_MODEL})")
        elif self.hf_token and len(self.hf_token) > 10:
            self.provider = "huggingface"
            print(f"✓ VLM provider: HuggingFace ({HF_VLM_MODEL})")
        else:
            self.provider = "fallback"
            print("⚠ No VLM API key found — using CLIP-based fallback")
            print("  Set GROQ_API_KEY env var for full VLM capability")

    def describe_frame(
        self,
        image_bgr: np.ndarray,
        clip_labels: list = None,
    ) -> dict:
        """
        Generate a structured description of a single frame.

        Parameters
        ----------
        image_bgr   : BGR frame
        clip_labels : optional CLIP labels for fallback enrichment

        Returns
        -------
        description : JSON-serialisable dict
        """
        try:
            if self.provider == "groq":
                desc = describe_with_groq(image_bgr, self.groq_key)

            elif self.provider == "huggingface":
                desc = describe_with_huggingface(image_bgr, self.hf_token)

            else:
                desc = describe_with_fallback(image_bgr, clip_labels)

        except Exception as e:
            print(f"  VLM call failed: {e}")
            print(f"  Falling back to rule-based description")
            desc = describe_with_fallback(image_bgr, clip_labels)

        return desc

    def describe_scene(
        self,
        frame_paths: list,
        clip_labels_per_frame: dict = None,
        n_keyframes: int = 5,
        output_path: str = None,
    ) -> dict:
        """
        Generate a unified scene description from multiple keyframes.

        Selects n_keyframes evenly from the video, describes each,
        then merges into a single coherent scene description.

        Parameters
        ----------
        frame_paths           : all frame paths
        clip_labels_per_frame : dict mapping frame_idx → label list
        n_keyframes           : how many frames to describe (API calls)
                                5 is a good balance of cost vs coverage
        output_path           : if set, save the description as JSON

        Returns
        -------
        scene_description : merged dict with all unique objects,
                            navigable regions, obstacles etc.
        """
        indices = np.linspace(
            0, len(frame_paths) - 1, n_keyframes, dtype=int
        ).tolist()

        print(f"Generating VLM scene description...")
        print(f"  Provider   : {self.provider}")
        print(f"  Keyframes  : {n_keyframes} frames")
        print()

        all_descriptions = []

        for ki, frame_idx in enumerate(indices):
            fpath   = frame_paths[frame_idx]
            img_bgr = cv2.imread(fpath)

            clip_labels = None
            if clip_labels_per_frame and frame_idx in clip_labels_per_frame:
                clip_labels = clip_labels_per_frame[frame_idx]

            print(f"  Frame {frame_idx:3d} ({ki+1}/{n_keyframes})...",
                  end=" ", flush=True)

            desc = self.describe_frame(img_bgr, clip_labels)
            desc["frame_idx"] = frame_idx
            all_descriptions.append(desc)

            n_obj = len(desc.get("objects", []))
            layout = desc.get("layout", "")[:60]
            print(f"{n_obj} objects — '{layout}'")

        # Merge all descriptions into one unified scene description
        merged = self._merge_descriptions(all_descriptions)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"\n✓ Scene description saved: {output_path}")

        return merged

    def _merge_descriptions(self, descriptions: list) -> dict:
        """
        Merge multiple per-frame descriptions into one scene description.

        Strategy:
          - Objects: union of all unique object names across frames
          - Layout: from the most detailed description (most objects)
          - Navigable regions: union of all unique regions
          - Obstacles: union of all unique obstacles
          - Robot notes: from the frame with the most useful description
        """
        # Collect all unique objects
        seen_objects = {}
        for desc in descriptions:
            for obj in desc.get("objects", []):
                name = obj.get("name", "").strip().lower()
                if name and name not in seen_objects:
                    seen_objects[name] = obj

        # Collect unique navigable regions
        nav_regions = []
        for desc in descriptions:
            for r in desc.get("navigable_regions", []):
                if r and r not in nav_regions:
                    nav_regions.append(r)

        # Collect unique obstacles
        obstacles = []
        for desc in descriptions:
            for o in desc.get("obstacles", []):
                if o and o not in obstacles:
                    obstacles.append(o)

        # Collect unique surfaces
        surfaces = []
        for desc in descriptions:
            for s in desc.get("surfaces", []):
                if s and s not in surfaces:
                    surfaces.append(s)

        # Pick best layout description (longest = most detailed)
        layouts = [
            d.get("layout", "") for d in descriptions
            if d.get("layout", "")
        ]
        best_layout = max(layouts, key=len) if layouts else "unknown"

        # Pick best robot notes
        notes = [
            d.get("robot_notes", "") for d in descriptions
            if d.get("robot_notes", "")
        ]
        best_notes = max(notes, key=len) if notes else ""

        merged = {
            "objects"           : list(seen_objects.values()),
            "layout"            : best_layout,
            "navigable_regions" : nav_regions,
            "obstacles"         : obstacles,
            "surfaces"          : surfaces,
            "robot_notes"       : best_notes,
            "n_frames_analysed" : len(descriptions),
            "provider"          : self.provider,
        }

        print(f"\nMerged scene description:")
        print(f"  Unique objects      : {len(merged['objects'])}")
        print(f"  Navigable regions   : {len(merged['navigable_regions'])}")
        print(f"  Obstacles           : {len(merged['obstacles'])}")
        print(f"  Surfaces            : {len(merged['surfaces'])}")

        return merged