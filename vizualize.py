#!/usr/bin/env python3
"""
mesh_viz.py — A handy PyVista mesh visualization script

Features
- Load a mesh from .vt*, .ply, .stl, .obj, .gltf/.glb, .3ds, .off, etc.
- Optional derived scalars (curvature, cell quality) and smoothing.
- Interactive widgets: threshold slider, isosurface slider (for volumes), opacity transfer toggle.
- Scene goodies: axes, orientation widget, EDL, scalar bar, edge display, grid, camera presets.
- Headless/offscreen screenshot export (PNG) and turntable GIF/MP4 export.

Examples
  # Quick look with curvature coloring
  python mesh_viz.py bunny.ply --curvature mean

  # Threshold interactively and export a screenshot
  python mesh_viz.py artery.vtu --threshold 0.2:0.9 --screenshot out.png

  # Offscreen render for CI
  python mesh_viz.py model.stl --offscreen --screenshot shot.png --camera iso

  # Volume: isosurface slider
  python mesh_viz.py ct_head.vti --isosurface --scalar "image"

Dependencies
  pip install pyvista vtk imageio

"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pyvista as pv

# ----------------------------
# Helpers
# ----------------------------

def load_dataset(path: Optional[str]) -> pv.DataSet:
    """Load a mesh/volume if path given; otherwise create a demo surface.

    Supports any format readable by PyVista/VTK. Falls back to a parametric
    surface when no file is provided, so the script is immediately usable.
    """
    if path is None:
        # A slightly interesting implicit surface for a nice demo
        sphere = pv.Sphere(theta_resolution=200, phi_resolution=200)
        bumps = np.sin(6*sphere.points[:, 0]) * np.cos(6*sphere.points[:, 1])
        sphere["bumps"] = (bumps - bumps.min()) / (bumps.ptp() + 1e-12)
        return sphere

    if not os.path.exists(path):
        sys.exit(f"Input not found: {path}")

    data = pv.read(path)
    if not isinstance(data, (pv.PolyData, pv.UnstructuredGrid, pv.ImageData, pv.StructuredGrid)):
        # Some formats return a MultiBlock; extract the first non-empty block.
        if isinstance(data, pv.MultiBlock) and len(data) > 0:
            for block in data:
                if block is not None and block.n_points > 0:
                    data = block
                    break
        else:
            sys.exit("Unsupported data type or empty dataset.")
    return data


def ensure_normals(mesh: pv.PolyData) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData) and "Normals" not in mesh.array_names:
        mesh = mesh.compute_normals(split_vertices=True, auto_orient_normals=True)
    return mesh


def add_curvature(mesh: pv.PolyData, mode: str = "mean") -> str:
    mode = mode.lower()
    if mode not in {"mean", "gaussian"}:
        sys.exit("--curvature must be 'mean' or 'gaussian'")
    curv = mesh.curvature(curv_type=mode)
    name = f"curvature_{mode}"
    mesh[name] = curv
    return name


def add_cell_quality(grid: pv.UnstructuredGrid, metric: str = "scaled_jacobian") -> str:
    metric = metric.lower()
    q = grid.compute_cell_quality(quality_measure=metric)
    name = f"quality_{metric}"
    grid.cell_data[name] = q["CellQuality"]
    return name


def parse_range(r: Optional[str]) -> Optional[Tuple[float, float]]:
    if not r:
        return None
    try:
        lo, hi = r.split(":")
        return float(lo), float(hi)
    except Exception:
        sys.exit("--threshold must be of the form 'min:max', e.g. 0.2:0.8")


# ----------------------------
# Main plotting logic
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="PyVista mesh/volume viewer")
    parser.add_argument("input", nargs="?", help="Path to mesh/volume (optional)")

    # Scene switches
    parser.add_argument("--background", default="#0b1021", help="Background color")
    parser.add_argument("--edl", action="store_true", help="Enable eye-dome lighting")
    parser.add_argument("--grid", action="store_true", help="Show XY grid")
    parser.add_argument("--edges", action="store_true", help="Show mesh edges")
    parser.add_argument("--camera", choices=["iso", "xy", "xz", "yz"], default="iso")
    parser.add_argument("--ortho", action="store_true", help="Orthographic projection")

    # Data/filters
    parser.add_argument("--scalar", help="Name of scalar array to color by")
    parser.add_argument("--curvature", choices=["mean", "gaussian"], help="Compute curvature on surfaces")
    parser.add_argument("--quality", default=None, help="Compute cell quality metric (unstructured grids)")
    parser.add_argument("--smooth", type=int, default=0, help="Laplacian smooth iterations (PolyData)")
    parser.add_argument("--threshold", help="Initial threshold range 'min:max' for threshold widget")
    parser.add_argument("--isosurface", action="store_true", help="Enable isosurface slider (volumes)")

    # Rendering & export
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--opacity", choices=["linear", "geom", "sigmoid"], help="Opacity transfer for volumes")
    parser.add_argument("--screenshot", help="Save a screenshot to this file (PNG)")
    parser.add_argument("--turntable", help="Export a turntable animation (GIF or MP4)")
    parser.add_argument("--frames", type=int, default=180, help="Frames for animation")
    parser.add_argument("--offscreen", action="store_true", help="Offscreen/headless rendering")

    args = parser.parse_args()

    if args.offscreen:
        pv.OFF_SCREEN = True

    data = load_dataset(args.input)

    # Optional smoothing for PolyData
    if isinstance(data, pv.PolyData) and args.smooth > 0:
        data = data.smooth(n_iter=args.smooth, relaxation_factor=0.01)
        data = ensure_normals(data)

    # Derived scalars
    scalar_name = args.scalar
    if isinstance(data, pv.PolyData) and args.curvature:
        scalar_name = add_curvature(data, args.curvature)
    if isinstance(data, pv.UnstructuredGrid) and args.quality:
        scalar_name = add_cell_quality(data, args.quality)

    # Plotter setup
    pl = pv.Plotter(window_size=(1280, 800))
    pl.set_background(args.background)
    if args.grid:
        pl.show_grid(color="white", location="outer")

    # Decide how to display based on data type
    display_kwargs = dict(cmap=args.cmap, scalar_bar_args={"title": scalar_name or ""})

    if isinstance(data, (pv.PolyData, pv.UnstructuredGrid)):
        # Try to color by a reasonable scalar if none specified
        if scalar_name is None:
            # Prefer point_data then cell_data
            if data.point_data and len(data.point_data.keys()) > 0:
                scalar_name = list(data.point_data.keys())[0]
            elif data.cell_data and len(data.cell_data.keys()) > 0:
                scalar_name = list(data.cell_data.keys())[0]
        if args.edges:
            display_kwargs["show_edges"] = True
        pl.add_mesh(data, scalars=scalar_name, **display_kwargs)

        # Threshold widget (if scalar available)
        if scalar_name is not None:
            rng = data.get_data_range(scalar_name)
            init = parse_range(args.threshold) or rng

            def _threshold_callback(lo, hi):
                # Remove any previous threshold actor and add a new one
                pl.remove_actor("threshold")
                th = data.threshold((lo, hi), scalars=scalar_name)
                pl.add_mesh(th, name="threshold", opacity=0.65)

            pl.add_slider_widget(
                _threshold_callback,
                rng=rng,
                value=init,
                title=f"Threshold: {scalar_name}",
                style="modern",
                pointa=(0.025, 0.1),
                pointb=(0.31, 0.1),
                event_type='always',
            )

    elif isinstance(data, (pv.ImageData, pv.StructuredGrid)):
        # Volume rendering or slices
        if args.opacity:
            if args.opacity == "linear":
                opacity = [0, 0.2, 0.4, 0.6, 0.8, 1]
            elif args.opacity == "geom":
                opacity = np.geomspace(0.01, 1, num=6).tolist()
            else:  # sigmoid-like
                x = np.linspace(-6, 6, 6)
                opacity = (1 / (1 + np.exp(-x))).tolist()
        else:
            opacity = None

        if args.isosurface:
            # Add an isovalue slider to extract a marching cubes surface
            sname = scalar_name or (list(data.point_data.keys())[0] if data.point_data else None)
            if sname is None:
                # Create a synthetic scalar field if none exists
                center = np.array(data.center)
                pts = data.points
                vals = np.linalg.norm(pts - center, axis=1)
                data.point_data["radial"] = vals
                sname = "radial"

            rng = data.get_data_range(sname)

            def _iso_callback(val):
                pl.remove_actor("iso")
                iso = data.contour(isosurfaces=[val], scalars=sname)
                pl.add_mesh(iso, name="iso", smooth_shading=True, **display_kwargs)

            pl.add_volume(data, scalars=sname, opacity=opacity, **display_kwargs)
            pl.add_slider_widget(
                lambda v: _iso_callback(v),
                rng=rng,
                value=sum(rng)/2,
                title=f"Isosurface: {sname}",
                style="modern",
                pointa=(0.69, 0.1),
                pointb=(0.975, 0.1),
            )
        else:
            # Orthogonal slices
            pl.add_mesh_slice(data, normal='x', **display_kwargs)
            pl.add_mesh_slice(data, normal='y', **display_kwargs)
            pl.add_mesh_slice(data, normal='z', **display_kwargs)
            if opacity is not None:
                pl.add_volume(data, opacity=opacity)

    # Camera & widgets
    if args.ortho:
        pl.enable_parallel_projection()

    if args.edl:
        pl.enable_eye_dome_lighting()

    pl.add_axes()
    pl.add_orientation_widget(pv.CubeAxesActor())

    if args.camera == "iso":
        pl.camera_position = "iso"
    elif args.camera == "xy":
        pl.view_xy()
    elif args.camera == "xz":
        pl.view_xz()
    elif args.camera == "yz":
        pl.view_yz()

    # Show or export
    if args.turntable:
        # Choose writer by extension
        ext = os.path.splitext(args.turntable)[1].lower()
        try:
            pl.open_movie(args.turntable, framerate=30, quality=7)
            pl.camera_position = pl.camera_position  # ensure initialized
            pl.add_text("Turntable", font_size=10)
            pl.show(auto_close=False)
            pl.write_frame()
            pl.plotter.store_image = True
            pl.camera.Zoom(1.0)
            for i in range(args.frames):
                pl.camera.Azimuth(360.0 / args.frames)
                pl.write_frame()
            pl.close()
            print(f"Saved turntable: {args.turntable}")
        except Exception as e:
            sys.exit(f"Failed to write animation: {e}")

    if args.screenshot:
        try:
            pl.show(screenshot=args.screenshot, auto_close=True)
            print(f"Saved screenshot: {args.screenshot}")
            return
        except Exception as e:
            sys.exit(f"Failed to save screenshot: {e}")

    # Interactive session if not offscreen or exporting only
    pl.show()


if __name__ == "__main__":
    main()
