#!/usr/bin/env python3
"""Visualize a COLMAP reconstruction using Vuer.

Shows the sparse 3D point cloud alongside camera frustums.
Click a frustum to jump to that camera's viewpoint.

Usage:
    python main.py --data-dir /path/to/colmap/project
"""

import colorsys
import re
from asyncio import sleep
from pathlib import Path

import numpy as np

from params_proto import proto

from vuer import Vuer
from vuer.schemas import (
    DefaultScene,
    Frustum,
    Group,
    OrbitControls,
    PerspectiveCamera,
    PointCloud,
)

from colmap_utils import (
    c2w_to_threejs,
    get_c2w,
    get_fov_aspect,
    mat4_to_column_major,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def camera_prefix(name: str) -> str:
    """Extract camera prefix from image name (everything before final _NNN.ext)."""
    m = re.match(r"^(.+?)_\d+\.", name)
    return m.group(1) if m else name.rsplit(".", 1)[0]


def assign_frustum_colors(prefixes: list) -> dict:
    """Map each unique prefix to a distinct hex color."""
    unique = sorted(set(prefixes))
    n = max(len(unique), 1)
    out = {}
    for i, pfx in enumerate(unique):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.8, 0.9)
        out[pfx] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return out


@proto.cli
def main(
    data_dir: Path = None,  # COLMAP project directory
):
    """Visualize COLMAP reconstruction with Vuer."""
    data_dir = Path(data_dir)
    sparse_dir = data_dir / "sparse" / "0"

    # ── 1. Read COLMAP reconstruction ───────────────────────────────────────

    print("Reading COLMAP data...")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    cam0 = list(cameras.values())[0]
    fov, aspect = get_fov_aspect(cam0)
    print(f"  {len(cameras)} camera(s), {len(images)} image(s) | FOV={fov:.1f} aspect={aspect:.2f}")

    # ── 2. Load point cloud ─────────────────────────────────────────────────

    pts3d = read_points3D_binary(sparse_dir / "points3D.bin")
    ids = list(pts3d.keys())
    vertices = np.array([pts3d[i].xyz for i in ids], dtype=np.float32)
    colors = np.array([pts3d[i].rgb for i in ids], dtype=np.float32) / 255.0
    print(f"  {len(vertices):,} points loaded")

    # ── 3. Prepare camera frustum colors ────────────────────────────────────

    sorted_img_ids = sorted(images)
    prefix_colors = assign_frustum_colors([camera_prefix(images[i].name) for i in sorted_img_ids])
    print(f"  {len(sorted_img_ids)} frustums")

    # ── 4. Launch Vuer ──────────────────────────────────────────────────────

    print("Launching visualization...")
    app = Vuer()

    @app.spawn(start=True)
    async def run(session):
        session.set @ DefaultScene(
            rawChildren=[
                PerspectiveCamera(
                    key="main-camera",
                    fov=60,
                    aspect=1.6,
                    near=0.1,
                    far=1000,
                    position=[0, 2, 5],
                    makeDefault=True,
                ),
            ],
            bgChildren=[OrbitControls(key="orbit")],
            up=[0, 1, 0],
            grid=False,
        )

        # Camera frustums
        frustum_children = []
        for img_id in sorted_img_ids:
            image = images[img_id]
            c2w = c2w_to_threejs(get_c2w(image))
            color = prefix_colors[camera_prefix(image.name)]

            frustum_children.append(
                Frustum(
                    key=f"cam_{img_id}",
                    matrix=mat4_to_column_major(c2w),
                    fov=fov,
                    aspect=aspect,
                    scale=0.15,
                    near=0.01,
                    far=0.1,
                    showFrustum=True,
                    showImagePlane=True,
                    showUp=True,
                    colorFrustum=color,
                    colorCone=color,
                    colorImagePlane=color,
                    colorUp="white",
                )
            )

        session.upsert @ Group(key="frustums", children=frustum_children)

        session.upsert @ PointCloud(
            key="pointcloud",
            vertices=vertices,
            colors=colors,
            size=0.01,
        )

        print("Visualization ready! Open the URL above in your browser.")

        while True:
            await sleep(1)


if __name__ == "__main__":
    main()
