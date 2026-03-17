#!/usr/bin/env python3
"""Visualize VGGT 4D reconstruction outputs using Vuer.

Shows a dense 3D point cloud colored by the original RGB images alongside
camera frustums. Supports frame-by-frame animation.

Usage:
    python main.py --data-dir '/path/to/vggt_outputs/kitchen'
    python main.py --data-dir '/path/to/vggt_outputs/kitchen' --animate
"""

import base64
import io
import os
from asyncio import sleep
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from params_proto import proto

from vuer import Vuer
from vuer.schemas import (
    DefaultScene,
    Img,
    PerspectiveFrustum,
    Group,
    OrbitControls,
    PerspectiveCamera,
    PointCloud,
)

# Flip Y and Z: OpenCV [right, down, forward] -> Three.js [right, up, backward]
# Applied to both world points and camera matrices for consistent convention.
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

CMAP = matplotlib.colormaps.get_cmap("turbo")


def mat4_to_column_major(m: np.ndarray) -> list:
    """4x4 numpy matrix to Three.js column-major flat list."""
    return m.T.flatten().tolist()


def w2c_to_c2w(w2c_3x4: np.ndarray) -> np.ndarray:
    """Convert VGGT world-to-camera (3x4) to camera-to-world (4x4)."""
    w2c = np.eye(4)
    w2c[:3, :4] = w2c_3x4
    return np.linalg.inv(w2c)


def get_fov_from_intrinsics(K: np.ndarray, height: int) -> float:
    """Compute vertical FOV in degrees from intrinsic matrix."""
    fy = K[1, 1]
    return float(np.degrees(2 * np.arctan(height / (2 * fy))))


def frame_color_hex(frame_idx: int, num_frames: int) -> str:
    """Get hex color for a frame index using turbo colormap."""
    rgba = CMAP(frame_idx / max(num_frames - 1, 1))
    return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"


def load_images(image_dir: Path, S: int, H: int, W: int) -> np.ndarray:
    """Load and resize input images to match point map resolution.

    Returns (S, H, W, 3) float32 array in [0, 1].
    """
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    image_files = sorted(image_files)[:S]
    assert len(image_files) == S, f"Expected {S} images, found {len(image_files)}"

    images = np.empty((S, H, W, 3), dtype=np.float32)
    for i, path in enumerate(image_files):
        img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
        images[i] = np.array(img, dtype=np.float32) / 255.0
    return images


def load_and_filter_points(
    data_dir: Path,
    image_dir: Path,
    use_depth_points: bool,
    conf_percentile: float,
    max_points: int,
):
    """Load VGGT outputs and filter points by confidence.

    Returns (vertices, colors, frame_ids, S, H, W) where colors are
    per-point RGB sampled from the input images.
    """
    if use_depth_points:
        world_points = np.load(data_dir / "point_map_from_depth.npy")
        conf = np.load(data_dir / "depth_conf.npy")
    else:
        world_points = np.load(data_dir / "world_points.npy")
        conf = np.load(data_dir / "world_points_conf.npy")

    S, H, W = conf.shape

    # Load RGB images resized to match point map
    images = load_images(image_dir, S, H, W)  # (S, H, W, 3)

    # Confidence filtering
    conf_flat = conf.reshape(-1)
    threshold = np.percentile(conf_flat, conf_percentile)
    mask = (conf_flat >= threshold) & (conf_flat > 1e-5)

    frame_indices = np.repeat(np.arange(S), H * W)

    pts = world_points.reshape(-1, 3)[mask]
    colors = images.reshape(-1, 3)[mask]
    frame_ids = frame_indices[mask]

    print(f"  {len(pts):,} points after {conf_percentile}% confidence filter")

    # Subsample if too many points
    if len(pts) > max_points:
        idx = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        colors = colors[idx]
        frame_ids = frame_ids[idx]
        print(f"  Subsampled to {max_points:,} points")

    pts = pts.astype(np.float32)

    # Transform points from OpenCV world (Y-down, Z-forward) to Three.js (Y-up, Z-backward)
    pts[:, 1] *= -1
    pts[:, 2] *= -1

    return pts, colors, frame_ids, S, H, W


@proto.cli
def main(
    data_dir: Path = "${DROPBOX}/data_collection/outputs/vggt/fps_5_frames_113",  # VGGT output directory
    conf_percentile: float = 90.0,  # Filter out lowest N% confidence points
    use_depth_points: bool = True,  # Use depth-unprojected points (more accurate)
    point_size: float = 0.005,  # Point size for rendering
    frustum_scale: float = 0.15,  # Camera frustum size
    max_points: int = 1000_000,  # Max points to render
    animate: bool = True,  # Animate frame-by-frame
    fps: float = 10,  # Animation frames per second
):
    """Visualize VGGT 4D reconstruction."""
    data_dir = Path(os.path.expandvars(data_dir))
    image_dir = data_dir / "images"

    # ── 1. Load VGGT outputs ────────────────────────────────────────────────

    print("Loading VGGT outputs...")
    extrinsics = np.load(data_dir / "extrinsic.npy")  # (S, 3, 4) w2c
    intrinsics = np.load(data_dir / "intrinsic.npy")  # (S, 3, 3)

    pts, colors, frame_ids, S, H, W = load_and_filter_points(
        data_dir, image_dir, use_depth_points, conf_percentile, max_points
    )  # (N, 3), (N, 3), (N,)

    print(f"  {S} frames, {H}x{W} resolution")

    # ── 2. Compute camera transforms ────────────────────────────────────────

    fov = get_fov_from_intrinsics(intrinsics[0], H)
    aspect = W / H
    print(f"  FOV={fov:.1f}°, aspect={aspect:.2f}")

    cam_c2w_threejs = []
    cam_positions = []
    cam_rotations = []
    for i in range(S):
        c2w = w2c_to_c2w(extrinsics[i])
        # Transform both world frame and camera local axes: FLIP @ c2w @ FLIP
        c2w_th = FLIP @ c2w @ FLIP
        cam_c2w_threejs.append(c2w_th)
        cam_positions.append(c2w_th[:3, 3].tolist())
        cam_rotations.append(R.from_matrix(c2w_th[:3, :3]).as_euler('xyz').tolist())

    # ── 3. Precompute base64 data URIs for source image overlay ────────────
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    image_files = sorted(image_files)[:S]

    image_data_uris = []
    for path in image_files:
        img = Image.open(path).convert("RGB").resize((300, int(300 / aspect)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_data_uris.append(f"data:image/jpeg;base64,{b64}")
    print(f"  Encoded {len(image_data_uris)} source images as data URIs")

    SOURCE_IMG_STYLE = {
        "position": "fixed", "bottom": "240px", "left": "10px",
        "width": "354px", "borderRadius": "8px",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.5)", "border": "2px solid #333",
    }

    # ── 4. Launch Vuer ──────────────────────────────────────────────────────
    app = Vuer()

    @app.spawn(start=True)
    async def run(session):
        session.set @ DefaultScene(
            bgChildren=[OrbitControls(key="orbit")],
            up=[0, 1, 0],
            grid=False,
        )

        if not animate:
            # Build all camera frustums
            frustum_children = []
            for i in range(S):
                color = frame_color_hex(i, S)
                frustum_children.append(
                    PerspectiveFrustum(
                        key=f"cam_{i}",
                        position=cam_positions[i],
                        rotation=cam_rotations[i],
                        fov=fov,
                        aspect=aspect,
                        scale=frustum_scale,
                        near=0.01,
                        far=0.1,
                        showFrustum=True,
                        showImagePlane=False,
                        showUp=True,
                        colorFrustum=color,
                        colorCone=color,
                        colorImagePlane=color,
                        colorUp="white",
                    )
                )

            session.upsert @ Group(key="frustums", children=frustum_children)

            # Static mode: show all points at once
            session.upsert @ PointCloud(
                key="pointcloud",
                vertices=pts,
                colors=colors,
                size=point_size,
            )
            while True:
                await sleep(1)
        else:
            # Animation mode: accumulate points frame-by-frame
            print("Animating frame-by-frame... Open the URL above in your browser.")
            frame = 0
            while True:
                # Show points up to current frame (accumulative)
                mask = (frame_ids <= frame) & (frame_ids > frame - 1)
                session.upsert @ PointCloud(
                    key="pointcloud",
                    vertices=pts[mask],
                    colors=colors[mask],
                    size=point_size,
                )

                # Move main camera to follow the current frame's camera
                session.upsert @ PerspectiveCamera(
                    key="main-camera",
                    fov=fov,
                    aspect=aspect,
                    near=0.01,
                    far=2,
                    position=cam_positions[frame],
                    rotation=cam_rotations[frame],
                    makeDefault=True,
                    showImagePlane=False,
                    previewInScene=False,
                )

                # Show source image for current frame
                session.upsert(
                    Img(src=image_data_uris[frame], key="source-image", style=SOURCE_IMG_STYLE),
                    to="htmlChildren",
                )

                frame = (frame + 1) % S
                await sleep(1.0 / fps)


if __name__ == "__main__":
    main()
