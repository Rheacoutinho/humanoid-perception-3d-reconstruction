"""
app.py
------
Gradio web application for language-queryable 3D scene understanding.
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Global state shared across all Gradio callbacks
_pipeline  = None
_result    = None
_embedder  = None


def render_pointcloud_image(
    points: np.ndarray,
    colours: np.ndarray,
    title: str = "",
    figsize: tuple = (12, 4),
    max_pts: int = 40_000,
) -> np.ndarray:
    """Render a 3D point cloud as a 2D PNG image (4 views)."""
    if len(points) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No points to display",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        N   = min(len(points), max_pts)
        idx = np.random.choice(len(points), N, replace=False)
        p   = points[idx]
        c   = np.clip(colours[idx], 0, 1)

        fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
        view_params = [
            (25, -60, "Perspective"),
            (90, -90, "Top-down"),
            (0,  -90, "Front"),
            (0,    0, "Side"),
        ]
        for i, (elev, azim, vtitle) in enumerate(view_params):
            ax = fig.add_subplot(1, 4, i + 1, projection="3d")
            ax.scatter(p[:, 0], p[:, 2], p[:, 1],
                       c=c, s=0.4, alpha=0.7)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(vtitle, color="white", fontsize=8)
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white", labelsize=6)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        if title:
            fig.suptitle(title, color="white", fontsize=10)
        plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    img  = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def render_query_result(engine, query_result: dict) -> np.ndarray:
    """Render a query result with highlighted matching points."""
    vis_data = engine.build_highlighted_cloud(query_result)
    return render_pointcloud_image(
        vis_data["points"],
        vis_data["colours"],
        title=(
            f"Query: '{query_result['query']}' -- "
            f"confidence {query_result['confidence']:.1f}%"
        ),
    )


def process_video(
    video_file,
    fastsam_ckpt_path: str,
    groq_key: str,
    output_dir_path: str,
    progress=gr.Progress(),
):
    """Gradio callback for Tab 1 -- Process Video button."""
    global _pipeline, _result, _embedder

    # Handle None or missing video
    if not video_file:
        return (
            None,
            "Please upload a video first.",
            "No scene processed yet.",
        )

    # Gradio may pass a dict or a string path
    if isinstance(video_file, dict):
        video_path = video_file.get("name", video_file.get("path", ""))
    else:
        video_path = str(video_file)

    if not os.path.exists(video_path):
        return (
            None,
            f"Video file not found: {video_path}",
            "",
        )

    try:
        progress(0, desc="Initialising pipeline...")

        from pipeline      import Pipeline
        from clip_embedder import CLIPEmbedder

        if not output_dir_path or not output_dir_path.strip():
            output_dir_path = tempfile.mkdtemp(prefix="scene3d_")

        os.makedirs(output_dir_path, exist_ok=True)

        ckpt = fastsam_ckpt_path.strip()
        if not os.path.exists(ckpt):
            search_paths = [
                "/content/drive/MyDrive/humanoid_3d_project/checkpoints/FastSAM-s.pt",
                "/content/FastSAM-s.pt",
                os.path.join(os.path.dirname(__file__), "..",
                             "checkpoints", "FastSAM-s.pt"),
            ]
            for sp in search_paths:
                if os.path.exists(sp):
                    ckpt = sp
                    break
            else:
                return (
                    None,
                    "FastSAM checkpoint not found. Please provide the path.",
                    "",
                )

        _pipeline = Pipeline(
            output_dir   = output_dir_path,
            fastsam_ckpt = ckpt,
            groq_api_key = groq_key.strip(),
        )

        def progress_cb(stage, pct, msg):
            progress(pct / 100, desc=f"{stage}: {msg}")

        progress(0.05, desc="Starting pipeline...")
        _result = _pipeline.run(
            video_path  = video_path,
            progress_cb = progress_cb,
        )

        _embedder = CLIPEmbedder()
        progress(0.95, desc="Generating visualisations...")

        import open3d as o3d
        ply_path = _result["output_paths"]["rgb_ply"]
        if os.path.exists(ply_path):
            pcd  = o3d.io.read_point_cloud(ply_path)
            pts  = np.asarray(pcd.points,  dtype=np.float32)
            cols = np.asarray(pcd.colors,  dtype=np.float32)
        else:
            data = np.load(
                _result["output_paths"]["query_npz"],
                allow_pickle=True,
            )
            pts  = data["points"]
            cols = data["colours"]

        cloud_img = render_pointcloud_image(
            pts, cols,
            title=f"Reconstructed scene -- {len(pts):,} points",
        )

        report = _result["accuracy_report"]
        summary = (
            f"Processing complete!\n\n"
            f"Accuracy Score: {report['overall_score']}/100\n\n"
            f"Pose quality     : {report['pose_quality']['score']:.1f}/100\n"
            f"Query quality    : {report['query_quality']['score']:.1f}/100\n"
            f"Depth consistency: {report['depth_consistency']['score']:.1f}/100\n"
            f"CLIP coverage    : {report['embedding_coverage']['score']:.1f}/100\n\n"
            f"Scene Stats\n"
            f"Frames : {_result['n_frames']}\n"
            f"Points : {_result['n_points']:,}\n"
            f"Masks  : {_result['n_masks']}\n"
            f"Time   : {_result['processing_time_s']}s\n"
        )

        scene_summary = _pipeline.get_scene_summary()
        progress(1.0, desc="Done!")
        return cloud_img, summary, scene_summary

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return (
            None,
            f"Error: {str(e)}\n\n{err}",
            "",
        )


def query_scene(query_text: str):
    """Gradio callback for Tab 2 -- Query button."""
    global _pipeline, _result, _embedder

    if _pipeline is None or _result is None:
        return (
            None,
            "No scene loaded. Process a video in Tab 1 first.",
            "",
        )

    if not query_text or not query_text.strip():
        return None, "Please enter a query.", ""

    try:
        from query_engine import QueryEngine

        engine = QueryEngine(
            clip_npz_path          = _result["output_paths"]["query_npz"],
            scene_description_path = _result["output_paths"]["scene_desc"],
        )

        result = engine.query(query_text.strip(), _embedder, top_k=500)
        vis_img = render_query_result(engine, result)

        vlm_ctx = result.get("vlm_context", [])
        vlm_str = (
            ", ".join(o["name"] for o in vlm_ctx)
            if vlm_ctx else "none"
        )

        result_text = (
            f"Query: '{result['query']}'\n\n"
            f"Confidence: {result['confidence']:.1f}%\n\n"
            f"Matched points  : {result['top_k']:,}\n"
            f"Max similarity  : {result['max_similarity']:.4f}\n"
            f"Compactness     : {result['compactness']:.3f}\n"
            f"Precision@k     : {result['precision_at_k']:.3f}\n\n"
            f"3D Location\n"
            f"Centre: ({result['bbox_centre'][0]:.2f}, "
            f"{result['bbox_centre'][1]:.2f}, "
            f"{result['bbox_centre'][2]:.2f}) m\n"
            f"Size  : {result['bbox_size'][0]:.2f} x "
            f"{result['bbox_size'][1]:.2f} x "
            f"{result['bbox_size'][2]:.2f} m\n\n"
            f"VLM Context: {vlm_str}\n"
        )

        robot_output = json.dumps({
            "query"      : result["query"],
            "confidence" : result["confidence"],
            "target_3d"  : {
                "centre": result["bbox_centre"],
                "size"  : result["bbox_size"],
            },
            "n_points"   : result["top_k"],
            "vlm_context": vlm_ctx,
        }, indent=2)

        return vis_img, result_text, robot_output

    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}\n{traceback.format_exc()}", ""


def get_scene_info():
    """Return full scene description and accuracy report."""
    global _result

    if _result is None:
        return "No scene loaded.", "No accuracy data."

    try:
        with open(_result["output_paths"]["scene_desc"]) as f:
            sd = json.load(f)

        objects   = sd.get("objects",            [])
        nav       = sd.get("navigable_regions",  [])
        obstacles = sd.get("obstacles",           [])
        layout    = sd.get("layout",             "")
        notes     = sd.get("robot_notes",        "")

        scene_text = (
            f"## Scene Layout\n{layout}\n\n"
            f"## Robot Navigation Note\n{notes}\n\n"
            f"## Objects Detected ({len(objects)})\n"
            + "\n".join(
                f"- {o['name']} -- {o.get('location', 'unknown')}  "
                f"{'navigable' if o.get('navigable') else 'obstacle'}"
                for o in objects
            )
            + f"\n\n## Navigable Regions ({len(nav)})\n"
            + "\n".join(f"- {r}" for r in nav)
            + f"\n\n## Obstacles ({len(obstacles)})\n"
            + "\n".join(f"- {o}" for o in obstacles)
        )

        with open(_result["output_paths"]["accuracy_json"]) as f:
            report = json.load(f)

        acc_text = (
            f"## Accuracy Report\n\n"
            f"Overall Score: {report['overall_score']}/100\n\n"
            f"### Depth Consistency\n"
            f"Score          : {report['depth_consistency']['score']}/100\n"
            f"Median error   : "
            f"{report['depth_consistency']['median_error_m']*100:.1f} cm\n"
            f"Under 10cm     : "
            f"{report['depth_consistency']['pct_under_10cm']:.1f}%\n\n"
            f"### Pose Quality\n"
            f"Score          : {report['pose_quality']['score']}/100\n"
            f"PnP success    : {report['pose_quality']['pnp_success_pct']}%\n"
            f"Smoothness     : {report['pose_quality']['smoothness_score']}/100\n"
            f"Trajectory     : {report['pose_quality']['trajectory_length']:.3f}m\n\n"
            f"### Query Quality\n"
            f"Score          : {report['query_quality']['score']}/100\n"
            f"Mean confidence: {report['query_quality']['mean_confidence']:.1f}%\n"
            f"Mean compactness: {report['query_quality']['mean_compactness']:.3f}\n\n"
            f"### Embedding Coverage\n"
            f"Score          : {report['embedding_coverage']['score']}/100\n"
            f"Coverage       : {report['embedding_coverage']['coverage_pct']:.1f}%\n"
            f"Mask regions   : {report['embedding_coverage']['n_mask_regions']}\n"
        )

        return scene_text, acc_text

    except Exception as e:
        return f"Error: {e}", ""


def compute_nav_targets():
    """Pre-compute navigation targets for the robot."""
    global _result, _embedder

    if _result is None or _embedder is None:
        return "No scene loaded. Process a video first.", None

    try:
        from query_engine import QueryEngine

        engine = QueryEngine(
            clip_npz_path          = _result["output_paths"]["query_npz"],
            scene_description_path = _result["output_paths"]["scene_desc"],
        )

        nav_queries = [
            "navigable floor space",
            "obstacle blocking path",
            "clear walkway or corridor",
            "wall or boundary",
            "furniture or large object",
            "workbench or table surface",
            "door or exit",
        ]

        nav_text = "## Navigation Targets\n\n"
        results  = []

        for q in nav_queries:
            r = engine.query(q, _embedder, top_k=300)
            results.append(r)
            nav_text += (
                f"### {q}\n"
                f"Confidence : {r['confidence']:.1f}%\n"
                f"Centre     : ({r['bbox_centre'][0]:.2f}, "
                f"{r['bbox_centre'][1]:.2f}, "
                f"{r['bbox_centre'][2]:.2f}) m\n"
                f"Size       : {r['bbox_size'][0]:.2f} x "
                f"{r['bbox_size'][1]:.2f} x "
                f"{r['bbox_size'][2]:.2f} m\n\n"
            )

        data     = np.load(
            _result["output_paths"]["query_npz"],
            allow_pickle=True,
        )
        nav_img  = render_pointcloud_image(
            data["points"],
            data["colours"],
            title="Point cloud -- run queries in Tab 2 for highlighted results",
        )

        return nav_text, nav_img

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None


def build_app(
    fastsam_default: str = "",
    output_default: str  = "",
    share: bool          = False,
):
    """Build and launch the Gradio app."""

    with gr.Blocks(
        title = "Language-Queryable 3D Scene Understanding",
        theme = gr.themes.Soft(primary_hue="teal"),
    ) as app:

        gr.Markdown(
            "# Language-Queryable 3D Scene Understanding\n"
            "**For Humanoid Robot Navigation** -- Upload a video, "
            "reconstruct the 3D scene, query it in natural language.\n\n"
            "*Depth-Anything V2 + CLIP ViT-B/32 + FastSAM + Llama 4 Vision*"
        )

        # Tab 1 -- Process Video
        with gr.Tab("Process Video"):
            gr.Markdown(
                "Upload a short indoor video (20-40 seconds). "
                "The system reconstructs the 3D scene."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    video_input  = gr.Video(label="Input Video", height=300)
                    ckpt_input   = gr.Textbox(
                        label       = "FastSAM checkpoint path",
                        value       = fastsam_default,
                        placeholder = "/path/to/FastSAM-s.pt",
                    )
                    groq_input   = gr.Textbox(
                        label       = "Groq API Key (optional)",
                        placeholder = "gsk_...",
                        type        = "password",
                    )
                    outdir_input = gr.Textbox(
                        label       = "Output directory",
                        value       = output_default,
                        placeholder = "/path/to/output",
                    )
                    process_btn  = gr.Button(
                        "Process Video",
                        variant = "primary",
                        size    = "lg",
                    )

                with gr.Column(scale=2):
                    cloud_output  = gr.Image(
                        label  = "3D Point Cloud",
                        height = 350,
                    )
                    status_output = gr.Markdown(
                        "Upload a video and click Process to begin."
                    )

            scene_preview = gr.Markdown("")

            process_btn.click(
                fn      = process_video,
                inputs  = [video_input, ckpt_input,
                           groq_input, outdir_input],
                outputs = [cloud_output, status_output, scene_preview],
            )

        # Tab 2 -- Query Scene
        with gr.Tab("Query Scene"):
            gr.Markdown(
                "Type a natural language query to find objects "
                "or regions in the 3D scene. Process a video first."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(
                        label       = "Query",
                        placeholder = (
                            "e.g. 'where is the chair?' or "
                            "'navigable floor' or 'obstacle'"
                        ),
                        lines = 2,
                    )
                    query_btn = gr.Button("Search", variant="primary")

                    gr.Markdown("**Example queries:**")
                    for eq in [
                        "floor", "chair", "navigable path",
                        "obstacle blocking robot", "wall",
                        "table or desk", "workbench",
                        "clear space to walk",
                    ]:
                        gr.Button(eq, size="sm").click(
                            fn      = lambda x=eq: x,
                            outputs = query_input,
                        )

                with gr.Column(scale=2):
                    query_image  = gr.Image(
                        label  = "Query Result",
                        height = 350,
                    )
                    query_result = gr.Markdown("")
                    robot_json   = gr.Code(
                        label    = "Robot-readable output (JSON)",
                        language = "json",
                    )

            query_btn.click(
                fn      = query_scene,
                inputs  = [query_input],
                outputs = [query_image, query_result, robot_json],
            )

        # Tab 3 -- Scene Info
        with gr.Tab("Scene Info"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="secondary")
            with gr.Row():
                with gr.Column():
                    scene_info_md = gr.Markdown(
                        "Process a video to see scene information."
                    )
                with gr.Column():
                    accuracy_md   = gr.Markdown(
                        "Accuracy report will appear here."
                    )
            refresh_btn.click(
                fn      = get_scene_info,
                inputs  = [],
                outputs = [scene_info_md, accuracy_md],
            )

        # Tab 4 -- Navigation
        with gr.Tab("Navigation"):
            gr.Markdown(
                "Pre-compute navigation targets for the robot."
            )
            nav_btn = gr.Button(
                "Compute Navigation Targets",
                variant = "primary",
            )
            with gr.Row():
                with gr.Column():
                    nav_text_out = gr.Markdown("")
                with gr.Column():
                    nav_img_out  = gr.Image(
                        label  = "Navigation overview",
                        height = 400,
                    )
            nav_btn.click(
                fn      = compute_nav_targets,
                inputs  = [],
                outputs = [nav_text_out, nav_img_out],
            )

        gr.Markdown(
            "---\n"
            "**Pipeline:** Frame extraction -> Depth-Anything V2 -> "
            "ORB+PnP pose estimation -> 3D fusion -> "
            "Llama 4 Vision -> FastSAM + CLIP -> Language query engine\n\n"
            "**Compute:** CPU-deployable, 4GB RAM minimum, GPU optional\n\n"
            "**Open source:** DA2 (Apache 2.0), CLIP (MIT), "
            "FastSAM (Apache 2.0), Llama 4 (Meta)"
        )

    app.launch(share=share, debug=False)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Language-queryable 3D scene understanding"
    )
    parser.add_argument("--fastsam", default="checkpoints/FastSAM-s.pt")
    parser.add_argument("--output",  default="output")
    parser.add_argument("--share",   action="store_true")
    args = parser.parse_args()

    build_app(
        fastsam_default = args.fastsam,
        output_default  = args.output,
        share           = args.share,
    )
