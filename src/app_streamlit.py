"""
app_streamlit.py
----------------
Streamlit web application for language-queryable 3D scene understanding.
Loads pre-processed outputs from Drive and provides a query interface.

Run with:
    streamlit run src/app_streamlit.py -- --output_dir /path/to/output
"""

import streamlit as st
import numpy as np
import json
import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "3D Scene Understanding",
    page_icon  = "🤖",
    layout     = "wide",
)


# ── Argument parsing ──────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default = "output",
        help    = "Path to pipeline output directory",
    )
    parser.add_argument(
        "--groq_key",
        default = "",
        help    = "Groq API key",
    )
    # Streamlit passes its own args so we strip those first
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        args = parser.parse_args([])
    return args


args = get_args()
OUTPUT_DIR = args.output_dir
if args.groq_key:
    os.environ["GROQ_API_KEY"] = args.groq_key


# ── Load pipeline outputs (cached with st.cache_resource) ─────────────────────
@st.cache_resource(show_spinner="Loading 3D scene data...")
def load_scene(output_dir: str):
    """
    Load all pipeline outputs once and cache them.
    st.cache_resource keeps this in memory across reruns.
    """
    from query_engine  import QueryEngine
    from clip_embedder import CLIPEmbedder

    npz_path   = os.path.join(output_dir, "pointcloud_query.npz")
    scene_path = os.path.join(output_dir, "scene_description.json")
    acc_path   = os.path.join(output_dir, "accuracy_report.json")
    ply_path   = os.path.join(output_dir, "pointcloud_rgb.ply")

    # Check required files exist
    missing = [p for p in [npz_path, scene_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required files: {missing}\n"
            f"Run the pipeline first."
        )

    embedder = CLIPEmbedder()
    engine   = QueryEngine(
        clip_npz_path          = npz_path,
        scene_description_path = scene_path,
    )

    with open(scene_path) as f:
        scene_desc = json.load(f)

    accuracy = {}
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            accuracy = json.load(f)

    # Load point cloud for display
    pts, cols = None, None
    if os.path.exists(ply_path):
        import open3d as o3d
        pcd  = o3d.io.read_point_cloud(ply_path)
        pts  = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float32)

    return engine, embedder, scene_desc, accuracy, pts, cols


# ── Rendering helper ──────────────────────────────────────────────────────────
def render_cloud(
    points  : np.ndarray,
    colours : np.ndarray,
    title   : str = "",
    max_pts : int = 30_000,
) -> plt.Figure:
    """Render point cloud as a 4-view matplotlib figure."""
    N   = min(len(points), max_pts)
    idx = np.random.choice(len(points), N, replace=False)
    p   = points[idx]
    c   = np.clip(colours[idx], 0, 1)

    fig = plt.figure(figsize=(14, 4), facecolor="#1a1a2e")
    for i, (elev, azim, t) in enumerate([
        (25, -60, "Perspective"),
        (90, -90, "Top-down"),
        (0,  -90, "Front"),
        (0,    0, "Side"),
    ]):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=c, s=0.3, alpha=0.7)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(t, color="white", fontsize=8)
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    if title:
        fig.suptitle(title, color="white", fontsize=10)
    plt.tight_layout()
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # Load everything
    try:
        engine, embedder, scene_desc, accuracy, pts, cols = load_scene(
            OUTPUT_DIR
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to load scene: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🤖 Language-Queryable 3D Scene Understanding")
    st.caption(
        "Depth-Anything V2 · CLIP ViT-B/32 · FastSAM · Llama 4 Vision"
    )

    score = accuracy.get("overall_score", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score",     f"{score}/100")
    c2.metric("Points",            f"{engine.M:,}")
    c3.metric("Mask Regions",      engine.n_masks)
    c4.metric("Pose Quality",
              f"{accuracy.get('pose_quality',{}).get('score',0):.0f}/100")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺 3D Scene",
        "🔍 Query Scene",
        "📋 Scene Info",
        "📊 Accuracy",
    ])

    # ── Tab 1: 3D Scene ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("Reconstructed 3D Point Cloud")
        if pts is not None:
            fig = render_cloud(
                pts, cols,
                title=f"Reconstructed scene — {len(pts):,} points",
            )
            st.pyplot(fig)
            plt.close(fig)

            col1, col2, col3 = st.columns(3)
            col1.metric("X range",
                        f"{pts[:,0].max()-pts[:,0].min():.2f} m")
            col2.metric("Y range",
                        f"{pts[:,1].max()-pts[:,1].min():.2f} m")
            col3.metric("Z range",
                        f"{pts[:,2].max()-pts[:,2].min():.2f} m")
        else:
            st.warning("Point cloud file not found.")

        layout = scene_desc.get("layout", "")
        if layout:
            st.info(f"**Scene layout:** {layout}")

        robot_note = scene_desc.get("robot_notes", "")
        if robot_note:
            st.success(f"**Robot navigation note:** {robot_note}")

    # ── Tab 2: Query Scene ────────────────────────────────────────────────────
    with tab2:
        st.subheader("Natural Language Query")
        st.caption(
            "Type any query to find the matching region in 3D space."
        )

        # Quick query buttons
        st.write("**Quick queries:**")
        quick_cols = st.columns(8)
        quick_queries = [
            "floor", "chair", "wall", "table",
            "navigable path", "obstacle", "workbench", "door",
        ]
        # Store selected quick query in session state
        if "query_text" not in st.session_state:
            st.session_state.query_text = ""

        for i, qq in enumerate(quick_queries):
            if quick_cols[i].button(qq, key=f"qq_{i}"):
                st.session_state.query_text = qq

        # Query input
        query_text = st.text_input(
            "Enter query",
            value       = st.session_state.query_text,
            placeholder = "e.g. 'where is the chair?' or 'navigable floor'",
            key         = "main_query_input",
        )

        search_btn = st.button("🔍 Search", type="primary")

        if search_btn and query_text.strip():
            with st.spinner(f"Searching for '{query_text}'..."):
                try:
                    result = engine.query(
                        query_text.strip(), embedder, top_k=500
                    )
                    vis    = engine.build_highlighted_cloud(result)

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Confidence",
                              f"{result['confidence']:.1f}%")
                    m2.metric("Matched Points",
                              f"{result['top_k']:,}")
                    m3.metric("Compactness",
                              f"{result['compactness']:.3f}")
                    m4.metric("Precision@k",
                              f"{result['precision_at_k']:.3f}")

                    # Render highlighted cloud
                    fig = render_cloud(
                        vis["points"],
                        vis["colours"],
                        title=(
                            "Query: '" + query_text + "' — conf "
                            + str(round(result["confidence"], 1)) + "%"
                        ),
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    # 3D bounding box
                    centre = result["bbox_centre"]
                    size   = result["bbox_size"]

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.subheader("3D Location")
                        st.write(
                            f"**Centre:** "
                            f"({centre[0]:.2f}, "
                            f"{centre[1]:.2f}, "
                            f"{centre[2]:.2f}) metres"
                        )
                        st.write(
                            f"**Size:** "
                            f"{size[0]:.2f} × "
                            f"{size[1]:.2f} × "
                            f"{size[2]:.2f} metres"
                        )

                        vlm_ctx = result.get("vlm_context", [])
                        if vlm_ctx:
                            st.write(
                                "**VLM context:** "
                                + ", ".join(
                                    o["name"] for o in vlm_ctx
                                )
                            )

                    with col_r:
                        st.subheader("Robot-readable output")
                        st.json({
                            "query"      : result["query"],
                            "confidence" : result["confidence"],
                            "centre_3d"  : [
                                round(x, 3)
                                for x in result["bbox_centre"]
                            ],
                            "size_3d"    : [
                                round(x, 3)
                                for x in result["bbox_size"]
                            ],
                            "n_points"   : result["top_k"],
                            "vlm_context": [
                                o["name"] for o in vlm_ctx
                            ],
                        })

                except Exception as e:
                    import traceback
                    st.error(f"Query failed: {e}")
                    st.code(traceback.format_exc())

    # ── Tab 3: Scene Info ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("VLM Scene Description")

        layout = scene_desc.get("layout", "")
        if layout:
            st.info(layout)

        robot_note = scene_desc.get("robot_notes", "")
        if robot_note:
            st.success(robot_note)

        col_l, col_r = st.columns(2)

        with col_l:
            st.write("### Objects Detected")
            objects = scene_desc.get("objects", [])
            for obj in objects:
                nav_icon = "✅" if obj.get("navigable") else "🚫"
                st.write(
                    f"{nav_icon} **{obj['name']}** — "
                    f"{obj.get('location', 'unknown')}"
                )

        with col_r:
            st.write("### Navigable Regions")
            for r in scene_desc.get("navigable_regions", []):
                st.write(f"✅ {r}")

            st.write("### Obstacles")
            for o in scene_desc.get("obstacles", []):
                st.write(f"🚫 {o}")

    # ── Tab 4: Accuracy ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("Accuracy Report")

        if not accuracy:
            st.warning("No accuracy report found.")
        else:
            overall = accuracy.get("overall_score", 0)
            st.metric("Overall Score", f"{overall}/100")

            a1, a2, a3, a4 = st.columns(4)

            depth = accuracy.get("depth_consistency", {})
            a1.metric(
                "Depth Consistency",
                f"{depth.get('score', 0):.1f}/100",
                delta=f"median err: "
                      f"{depth.get('median_error_m', 0)*100:.1f}cm",
            )

            pose = accuracy.get("pose_quality", {})
            a2.metric(
                "Pose Quality",
                f"{pose.get('score', 0):.1f}/100",
                delta=f"PnP: {pose.get('pnp_success_pct', 0)}%",
            )

            query_q = accuracy.get("query_quality", {})
            a3.metric(
                "Query Quality",
                f"{query_q.get('score', 0):.1f}/100",
                delta=f"conf: "
                      f"{query_q.get('mean_confidence', 0):.1f}%",
            )

            embed = accuracy.get("embedding_coverage", {})
            a4.metric(
                "CLIP Coverage",
                f"{embed.get('score', 0):.1f}/100",
                delta=f"{embed.get('coverage_pct', 0):.1f}% covered",
            )

            st.divider()
            st.write("### Per-Query Breakdown")
            per_q = query_q.get("per_query", [])
            if per_q:
                import pandas as pd
                df = pd.DataFrame(per_q)
                st.dataframe(df, use_container_width=True)

            st.divider()
            st.write("### Full Report (JSON)")
            st.json(accuracy)


if __name__ == "__main__":
    main()