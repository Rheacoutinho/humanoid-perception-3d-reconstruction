"""
mesh_reconstruct.py
-------------------
Takes the fused point cloud (output of depth_refine.py) and reconstructs
a clean triangle mesh using Open3D's Poisson surface reconstruction.

Steps:
  1. Load the fused point cloud
  2. Estimate surface normals for every point
  3. Orient normals consistently (all pointing outward)
  4. Run Poisson surface reconstruction
  5. Trim low-density regions (areas with sparse point coverage)
  6. Clean the mesh (remove degenerate triangles, small islands)
  7. Export as .obj, .glb, and a coloured .ply
  8. Generate multiple rendered preview images
"""

import os
import json
import numpy as np
import open3d as o3d


def load_and_prepare_pointcloud(
    ply_path: str,
    voxel_size: float = 0.01,
) -> o3d.geometry.PointCloud:
    """
    Load point cloud and prepare it for meshing.

    Preparation steps:
      - Voxel downsample to make point density uniform
        (Poisson works best with evenly distributed points)
      - Remove outliers one more time on the fused cloud
      - Estimate and orient surface normals

    Parameters
    ----------
    ply_path   : path to the fused .ply point cloud
    voxel_size : grid cell size for downsampling in metres
                 0.01 = 1cm resolution — good for indoor rooms

    Returns
    -------
    pcd : prepared Open3D PointCloud with normals estimated
    """
    print(f"Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"  Loaded: {len(pcd.points):,} points")

    # ── Voxel downsample for uniform density ─────────────────────────────────
    print(f"  Voxel downsampling (size={voxel_size}m)...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  After downsample: {len(pcd.points):,} points")

    # ── Remove outliers ───────────────────────────────────────────────────────
    print("  Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.0,
    )
    print(f"  After outlier removal: {len(pcd.points):,} points")

    # ── Estimate surface normals ──────────────────────────────────────────────
    # search_param defines how many neighbours to use for normal estimation
    # KDTreeSearchParamHybrid: use all neighbours within 0.1m, max 30
    print("  Estimating surface normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30,
        )
    )

    # ── Orient normals consistently ───────────────────────────────────────────
    # Without this, some normals point inward and some outward — the mesh
    # will have holes and flipped faces
    # orient_normals_consistent_tangent_plane propagates orientation from
    # neighbours so they all point the same direction
    print("  Orienting normals...")
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print("  ✓ Point cloud prepared for meshing")
    return pcd


def run_poisson_reconstruction(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    density_threshold_percentile: float = 5.0,
) -> tuple:
    """
    Run Poisson surface reconstruction on a prepared point cloud.

    How Poisson reconstruction works:
      - Treats the point cloud as samples of a surface
      - Builds an octree (a 3D grid that subdivides space)
      - Solves a Poisson equation to find the smoothest implicit
        surface that fits the points and their normals
      - Extracts the surface as a triangle mesh via marching cubes

    Parameters
    ----------
    depth : octree depth — controls resolution
            8 = coarser but faster (~500k triangles)
            9 = good balance (default, ~1–2M triangles)
            10 = very fine but slow and memory hungry
    density_threshold_percentile : remove triangles in regions where
            fewer than this percentile of points supported them
            5.0 = remove the least-supported 5% of triangles
            Higher = more aggressive trimming = cleaner but smaller mesh

    Returns
    -------
    mesh        : o3d.geometry.TriangleMesh — the reconstructed surface
    densities   : per-vertex density values (useful for trimming)
    """
    print(f"\nRunning Poisson reconstruction (depth={depth})...")
    print("This takes 1–3 minutes...")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False,
    )

    print(f"  Raw mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # ── Trim low-density regions ──────────────────────────────────────────────
    # Poisson fills in ALL space — even areas the camera never saw
    # get a surface. Density tells us how many points supported each vertex.
    # We remove vertices (and their triangles) below the density threshold.
    densities_np = np.asarray(densities)
    threshold    = np.percentile(densities_np, density_threshold_percentile)

    print(f"  Density range: {densities_np.min():.3f} – {densities_np.max():.3f}")
    print(f"  Trimming below {density_threshold_percentile}th percentile "
          f"(threshold={threshold:.3f})...")

    vertices_to_remove = densities_np < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"  After trimming: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    return mesh, densities


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Clean up the mesh by removing degenerate geometry.

    Cleaning steps:
      - remove_degenerate_triangles: triangles with zero area
      - remove_duplicated_triangles: exact duplicate faces
      - remove_duplicated_vertices: vertices at identical positions
      - remove_non_manifold_edges: edges shared by >2 triangles
        (these cause issues in downstream tools)
      - keep only the largest connected component
        (removes floating fragments from reconstruction artefacts)
    """
    print("\nCleaning mesh...")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    print(f"  After basic cleaning: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # ── Keep only the largest connected component ─────────────────────────────
    # cluster_connected_triangles returns:
    #   triangle_clusters : which cluster each triangle belongs to
    #   cluster_n_triangles: how many triangles in each cluster
    #   cluster_area       : total area of each cluster
    triangle_clusters, cluster_n_triangles, _ = (
        mesh.cluster_connected_triangles()
    )

    triangle_clusters  = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # Find the largest cluster
    largest_cluster_idx = cluster_n_triangles.argmax()

    # Remove all triangles NOT in the largest cluster
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()

    print(f"  After keeping largest component: "
          f"{len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # Recompute vertex normals for correct lighting in viewers
    mesh.compute_vertex_normals()

    return mesh


def transfer_colours_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
    k_neighbours: int = 3,
) -> o3d.geometry.TriangleMesh:
    """
    Transfer colours from the point cloud to the mesh vertices.

    Poisson reconstruction creates new vertices that don't correspond
    directly to input points, so we need to find the nearest point
    cloud neighbours for each vertex and copy their colour.

    Uses a KD-tree for fast nearest-neighbour lookup.

    Parameters
    ----------
    mesh         : the reconstructed mesh (no colours yet)
    pcd          : the coloured point cloud
    k_neighbours : number of nearest neighbours to average colour from
                   3 = smooth colour blending between nearby points

    Returns
    -------
    mesh with vertex colours assigned
    """
    print("\nTransferring colours from point cloud to mesh...")

    pcd_tree   = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)       # (N, 3) float in [0, 1]
    vertices   = np.asarray(mesh.vertices)    # (M, 3)

    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float64)

    for i, vertex in enumerate(vertices):
        # Find k nearest point cloud neighbours to this vertex
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, k_neighbours)
        # Average their colours
        vertex_colors[i] = pcd_colors[np.asarray(idx)].mean(axis=0)

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    print(f"  ✓ Colours transferred to {len(vertices):,} vertices")

    return mesh


def save_mesh_formats(
    mesh: o3d.geometry.TriangleMesh,
    pcd_clean: o3d.geometry.PointCloud,
    output_dir: str,
) -> dict:
    """
    Save the mesh in multiple formats.

    Formats:
      .ply  — coloured point cloud (universal, open in MeshLab/CloudCompare)
      .obj  — mesh with material file (universal, opens in Blender/MeshLab)
      .glb  — binary glTF mesh (opens in web browsers, Windows 3D Viewer)
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    # ── Coloured point cloud ──────────────────────────────────────────────────
    ply_path = os.path.join(output_dir, "pointcloud_clean.ply")
    o3d.io.write_point_cloud(ply_path, pcd_clean)
    saved["pointcloud_clean_ply"] = ply_path
    print(f"  ✓ Point cloud (.ply) : {os.path.getsize(ply_path)/1e6:.1f} MB")

    # ── OBJ mesh ──────────────────────────────────────────────────────────────
    obj_path = os.path.join(output_dir, "mesh.obj")
    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_colors=True)
    saved["mesh_obj"] = obj_path
    print(f"  ✓ Mesh (.obj)        : {os.path.getsize(obj_path)/1e6:.1f} MB")

    # ── GLB mesh ──────────────────────────────────────────────────────────────
    glb_path = os.path.join(output_dir, "mesh.glb")
    o3d.io.write_triangle_mesh(glb_path, mesh)
    saved["mesh_glb"] = glb_path
    print(f"  ✓ Mesh (.glb)        : {os.path.getsize(glb_path)/1e6:.1f} MB")

    return saved


def render_mesh_previews(
    mesh: o3d.geometry.TriangleMesh,
    output_dir: str,
) -> list:
    """
    Render preview images of the mesh from multiple angles.

    Since Open3D's interactive viewer doesn't work in Colab,
    we use matplotlib to render 2D projections of the mesh
    by plotting the triangle edges as lines.

    Views rendered:
      - Perspective (isometric-style)
      - Top-down
      - Front
      - Side
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    print("\nRendering mesh previews...")

    vertices  = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors    = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    # Subsample triangles for faster rendering
    # matplotlib can't handle millions of polygons
    max_tris = 15_000
    if len(triangles) > max_tris:
        idx       = np.random.choice(len(triangles), max_tris, replace=False)
        triangles = triangles[idx]

    # Build triangle vertex arrays
    tri_verts = vertices[triangles]  # (T, 3, 3)

    # Get per-triangle colour (average of 3 vertex colours)
    if colors is not None:
        tri_colors = colors[triangles].mean(axis=1)  # (T, 3)
        tri_colors = np.clip(tri_colors, 0, 1)
    else:
        tri_colors = np.ones((len(triangles), 3)) * 0.7

    view_params = [
        (25,  -60, "Perspective"),
        (90,  -90, "Top-down"),
        (0,   -90, "Front"),
        (0,     0, "Side"),
    ]

    preview_paths = []
    fig = plt.figure(figsize=(18, 5))

    for i, (elev, azim, title) in enumerate(view_params):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')

        poly = Poly3DCollection(
            tri_verts,
            facecolors=tri_colors,
            edgecolors='none',
            alpha=0.95,
        )
        ax.add_collection3d(poly)

        # Set axis limits
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.grid(False)

    plt.suptitle(
        f"Reconstructed mesh — {len(np.asarray(mesh.vertices)):,} vertices  "
        f"{len(np.asarray(mesh.triangles)):,} triangles",
        fontsize=12,
    )
    plt.tight_layout()

    preview_path = os.path.join(output_dir, "mesh_preview.png")
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.show()
    preview_paths.append(preview_path)
    print(f"  ✓ Preview saved: {preview_path}")

    return preview_paths


def reconstruct_mesh(
    fused_ply_path: str,
    output_dir: str,
    poisson_depth: int = 9,
    density_threshold_percentile: float = 5.0,
    voxel_size: float = 0.01,
) -> dict:
    """
    Master function — runs the full mesh reconstruction pipeline.
    Call this from Colab.
    """
    print("=" * 50)
    print("TASK 4 — MESH RECONSTRUCTION")
    print("=" * 50)

    # Step 1 — Load and prepare point cloud
    pcd = load_and_prepare_pointcloud(fused_ply_path, voxel_size=voxel_size)

    # Step 2 — Poisson reconstruction
    mesh, densities = run_poisson_reconstruction(
        pcd,
        depth=poisson_depth,
        density_threshold_percentile=density_threshold_percentile,
    )

    # Step 3 — Clean mesh
    mesh = clean_mesh(mesh)

    # Step 4 — Transfer colours
    mesh = transfer_colours_to_mesh(mesh, pcd, k_neighbours=3)

    # Step 5 — Save all formats
    print("\nSaving outputs...")
    saved_files = save_mesh_formats(mesh, pcd, output_dir)

    # Step 6 — Render previews
    preview_paths = render_mesh_previews(mesh, output_dir)

    # Step 7 — Save summary
    summary = {
        "num_vertices":  len(mesh.vertices),
        "num_triangles": len(mesh.triangles),
        "poisson_depth": poisson_depth,
        "voxel_size":    voxel_size,
        "output_files":  saved_files,
        "previews":      preview_paths,
    }

    summary_path = os.path.join(output_dir, "mesh_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Mesh Reconstruction Summary ===")
    print(f"  Vertices  : {summary['num_vertices']:,}")
    print(f"  Triangles : {summary['num_triangles']:,}")
    for fmt, path in saved_files.items():
        print(f"  {fmt:25s}: {path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fused_ply", type=str)
    parser.add_argument("--output_dir",    type=str,   default="output")
    parser.add_argument("--poisson_depth", type=int,   default=9)
    parser.add_argument("--density_pct",   type=float, default=5.0)
    parser.add_argument("--voxel_size",    type=float, default=0.01)
    args = parser.parse_args()

    reconstruct_mesh(
        fused_ply_path               = args.fused_ply,
        output_dir                   = args.output_dir,
        poisson_depth                = args.poisson_depth,
        density_threshold_percentile = args.density_pct,
        voxel_size                   = args.voxel_size,
    )