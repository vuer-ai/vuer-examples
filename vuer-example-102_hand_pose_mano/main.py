#!/usr/bin/env python3
"""Visualize Dyn-HaMR MANO hand pose reconstruction using Vuer.

Shows animated 3D hand meshes with skeleton overlay and interactive orbit
controls, replacing the pre-rendered video output from Dyn-HaMR.

The original Dyn-HaMR pipeline renders fixed-viewpoint videos (src_cam,
above, front, side) using PyRender. This script loads the same OBJ mesh
outputs and displays them in an interactive 3D viewer where the user can
freely orbit, zoom, and pan.

Usage:
    python main.py --data-dir '/path/to/dyn-hamr/output'
    python main.py --data-dir '/path/to/dyn-hamr/output' --fps 15
    python main.py --show-skeleton False  # mesh only
    python main.py --show-mesh False      # skeleton only
"""

import base64
import io
import os
from asyncio import sleep
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from params_proto import proto

from vuer import Vuer
from vuer.schemas import (
    DefaultScene,
    Img,
    TriMesh,
    Line,
    Sphere,
    OrbitControls,
    PerspectiveCamera,
)

ASSETS_DIR = Path(__file__).parent / "assets"

# MANO skeleton: kintree_table[0] gives parent of each joint
# Joint 0 = wrist, 1-3 = index, 4-6 = middle, 7-9 = pinky/ring3(?), 10-12 = ring, 13-15 = thumb
# Bones derived from kintree_table parent->child relationships:
MANO_BONES = [
    (0, 1), (1, 2), (2, 3),     # Index finger
    (0, 4), (4, 5), (5, 6),     # Middle finger
    (0, 7), (7, 8), (8, 9),     # Pinky
    (0, 10), (10, 11), (11, 12),  # Ring
    (0, 13), (13, 14), (14, 15),  # Thumb
]

# Color each finger differently
BONE_COLORS = [
    "#ff4444", "#ff4444", "#ff4444",   # Index - red
    "#44cc44", "#44cc44", "#44cc44",   # Middle - green
    "#4488ff", "#4488ff", "#4488ff",   # Pinky - blue
    "#ffaa44", "#ffaa44", "#ffaa44",   # Ring - orange
    "#cc44cc", "#cc44cc", "#cc44cc",   # Thumb - magenta
]


def load_obj_vertices(path):
    """Parse OBJ file returning vertices (Nx3) and faces (Mx3)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line[0] == "v" and line[1] == " ":
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line[0] == "f":
                parts = line.split()
                faces.append((
                    int(parts[1].split("/")[0]) - 1,
                    int(parts[2].split("/")[0]) - 1,
                    int(parts[3].split("/")[0]) - 1,
                ))
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def discover_sequence(data_dir: Path):
    """Auto-discover mesh directory and NPZ from Dyn-HaMR output."""
    smooth_dir = data_dir / "smooth_fit"
    mesh_dirs = sorted(smooth_dir.glob("*_meshes"))
    if not mesh_dirs:
        raise FileNotFoundError(f"No mesh directories found in {smooth_dir}")
    mesh_dir = mesh_dirs[-1]
    stem = mesh_dir.name.replace("_meshes", "")
    npz_path = smooth_dir / f"{stem}_world_results.npz"
    return mesh_dir, npz_path


def load_joint_regressors():
    """Load pre-extracted MANO joint regressors (16x778 matrices)."""
    regressor_path = ASSETS_DIR / "mano_joint_regressor.npz"
    data = np.load(regressor_path)
    return data["J_regressor_right"], data["J_regressor_left"]


@proto.cli
def main(
    data_dir: Path = "${DROPBOX}/data_collection/outputs/dyn-hamr",
    fps: float = 30.0,
    show_skeleton: bool = True,
    show_mesh: bool = True,
):
    """Visualize Dyn-HaMR MANO hand meshes in interactive 3D."""
    data_dir = Path(os.path.expandvars(data_dir))
    mesh_dir, npz_path = discover_sequence(data_dir)

    # Load metadata
    npz_data = np.load(npz_path)
    is_right = npz_data["is_right"]  # (num_people, num_frames)
    num_people = is_right.shape[0]

    # ── Camera parameters ────────────────────────────────────────────────
    # cam_R/cam_t are identical for both people (same physical camera).
    # They represent the world-to-camera transform in the NPZ world frame.
    cam_R_w2c = npz_data["cam_R"][0]  # (num_frames, 3, 3)
    cam_t_w2c = npz_data["cam_t"][0]  # (num_frames, 3)
    intrins = npz_data["intrins"]     # [fx, fy, cx, cy]

    # Compute vertical FOV from intrinsics: fov_y = 2 * atan(cy / fy)
    fy, cy = float(intrins[1]), float(intrins[3])
    fov = float(np.degrees(2 * np.arctan(cy / fy)))
    aspect = float(intrins[2] / intrins[3])  # cx/cy ≈ width/height
    print(f"Camera: FOV={fov:.1f}°, aspect={aspect:.2f}")

    # Load joint regressors if skeleton is enabled
    J_right, J_left = None, None
    if show_skeleton:
        J_right, J_left = load_joint_regressors()
        print(f"Loaded joint regressors: right {J_right.shape}, left {J_left.shape}")

    # ── Load meshes ──────────────────────────────────────────────────────
    obj_files = sorted(mesh_dir.glob("*.obj"))
    frame_set = sorted(set(int(f.stem.split("_")[0]) for f in obj_files))
    num_frames = len(frame_set)
    print(f"Found {num_frames} frames, {len(obj_files)} mesh files, {num_people} people")

    print("Loading meshes...")
    all_verts = {}
    faces_cache = {}

    for i, f in enumerate(obj_files):
        parts = f.stem.split("_")
        frame_idx, person_idx = int(parts[0]), int(parts[1])
        verts, faces = load_obj_vertices(f)
        all_verts[(frame_idx, person_idx)] = verts
        if person_idx not in faces_cache:
            faces_cache[person_idx] = faces
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(obj_files)}")

    print(f"Loaded {len(all_verts)} meshes ({num_frames} frames x up to {num_people} hands)")

    # ── Camera coordinate transform ──────────────────────────────────────
    # The OBJ meshes are NOT in the NPZ world frame. Dyn-HaMR's run_vis.py
    # applies two transforms before saving OBJ files:
    #   1. Subtract init_trans (centers scene on joint-9 midpoint)
    #   2. Apply 180° rotation around X: Rx = diag(1, -1, -1)
    # To place the camera in OBJ space: c2w_obj = world_to_obj @ c2w_world
    Rx = np.diag([1.0, -1.0, -1.0])

    # Solve init_trans to transform the NPZ camera into OBJ mesh space.
    # Dyn-HaMR's run_vis.py centers meshes at init_trans (joint-9 midpoint in
    # world frame) then rotates by Rx=diag(1,-1,-1). The NPZ camera is in the
    # original world frame (≈ OpenCV camera frame: X-right, Y-down, Z-forward).
    #
    # For X: use the projection constraint (scene center projects to image center
    #   horizontally). This works because root_orient mainly rotates in the XZ
    #   plane, so trans_x is a poor proxy but the projection constraint is exact.
    # For Y: use the NPZ trans average. In the Y-down world frame, trans_y gives
    #   the hand's vertical position below the camera center. root_orient barely
    #   shifts the Y component, so trans_y is a reliable proxy for init_trans_y.
    # For Z: use the projection depth constraint.
    fi0 = frame_set[0]
    first_centroids = [all_verts[(fi0, p)].mean(axis=0)
                       for p in range(num_people) if (fi0, p) in all_verts]
    centroid_obj_0 = np.mean(first_centroids, axis=0)
    rc = Rx @ centroid_obj_0

    trans = npz_data["trans"]  # (num_people, num_frames, 3)
    init_trans = np.zeros(3)
    init_trans[0] = -rc[0] - cam_t_w2c[fi0, 0]           # from projection
    init_trans[1] = float(trans[:, 0, 1].mean())           # from NPZ trans Y
    init_trans[2] = -rc[2] - cam_t_w2c[fi0, 2] + 0.44     # from projection depth
    print(f"Solved init_trans: {init_trans}")

    # Build 4x4 world-to-OBJ transform
    world_to_obj = np.eye(4)
    world_to_obj[:3, :3] = Rx
    world_to_obj[:3, 3] = -Rx @ init_trans

    # Flip camera local axes: OpenCV looks along +Z, Three.js along -Z.
    FLIP_CAM = np.diag([1.0, -1.0, -1.0, 1.0])

    # Precompute camera poses in OBJ space for each frame
    num_cam_frames = cam_R_w2c.shape[0]
    cam_positions = []
    cam_rotations = []
    for i in range(num_cam_frames):
        w2c = np.eye(4)
        w2c[:3, :3] = cam_R_w2c[i]
        w2c[:3, 3] = cam_t_w2c[i]
        c2w_world = np.linalg.inv(w2c)
        c2w_obj = world_to_obj @ c2w_world @ FLIP_CAM
        cam_positions.append(c2w_obj[:3, 3].tolist())
        cam_rotations.append(R.from_matrix(c2w_obj[:3, :3]).as_euler("xyz").tolist())

    print(f"Camera frame 0 position (OBJ): {cam_positions[0]}, "
          f"dist={np.linalg.norm(cam_positions[0]):.3f}")

    hand_color = np.array([250, 128, 114], dtype=np.uint8)  # salmon

    # ── Precompute source image data URIs ───────────────────────────────
    # Find the src_cam rendered frames (original RGB with mesh overlay)
    seq_name = mesh_dir.name.rsplit("_meshes", 1)[0]  # e.g. camera_1_episode_0001_000300
    phase_stem = seq_name.rsplit("_", 1)[0]  # e.g. camera_1_episode_0001
    src_cam_dir = data_dir / f"{phase_stem}_smooth_fit_final_{seq_name.split('_')[-1]}_src_cam"
    if not src_cam_dir.exists():
        # Fallback: try input video frames directory
        src_cam_dir = None

    image_data_uris = []
    if src_cam_dir and src_cam_dir.is_dir():
        image_files = sorted(src_cam_dir.glob("*.jpg")) + sorted(src_cam_dir.glob("*.png"))
        image_files = sorted(image_files)[:num_frames]
        print(f"Loading {len(image_files)} source images from {src_cam_dir.name}...")
        for path in image_files:
            img = Image.open(path).convert("RGB").resize(
                (300, int(300 / aspect)), Image.LANCZOS
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_data_uris.append(f"data:image/jpeg;base64,{b64}")
        print(f"Encoded {len(image_data_uris)} source images as data URIs")

    SOURCE_IMG_STYLE = {
        "position": "fixed", "bottom": "240px", "left": "10px",
        "width": "354px", "borderRadius": "8px",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.5)", "border": "2px solid #333",
    }

    # ── Launch Vuer ──────────────────────────────────────────────────────
    app = Vuer()

    @app.spawn(start=True)
    async def run(session):
        session.set @ DefaultScene(
            bgChildren=[OrbitControls(key="orbit")],
            up=[0, 1, 0],
            grid=False,
        )

        frame = 0
        while True:
            fi = frame_set[frame]

            # ── Move camera to follow original camera trajectory ─────
            session.upsert @ PerspectiveCamera(
                key="main-camera",
                fov=fov,
                aspect=aspect,
                near=0.001,
                far=100,
                position=cam_positions[fi],
                rotation=cam_rotations[fi],
                makeDefault=True,
                showImagePlane=False,
                previewInScene=False,
            )

            # ── Update hand meshes ───────────────────────────────────
            for person_idx in range(num_people):
                if (fi, person_idx) not in all_verts:
                    if show_mesh:
                        session.upsert @ TriMesh(
                            key=f"hand_{person_idx}",
                            vertices=np.zeros((3, 3), dtype=np.float32),
                            faces=np.zeros((1, 3), dtype=np.int32),
                        )
                    continue

                verts = all_verts[(fi, person_idx)]
                faces = faces_cache[person_idx]
                hand_right = bool(is_right[person_idx, fi] > 0.5)

                if show_mesh:
                    vert_colors = np.tile(hand_color, (len(verts), 1))
                    session.upsert @ TriMesh(
                        key=f"hand_{person_idx}",
                        vertices=verts,
                        faces=faces,
                        colors=vert_colors,
                    )

                if show_skeleton:
                    J = J_right if hand_right else J_left
                    joints = J @ verts  # (16, 3)

                    for bone_idx, (parent, child) in enumerate(MANO_BONES):
                        session.upsert @ Line(
                            key=f"bone_{person_idx}_{bone_idx}",
                            points=[joints[parent].tolist(), joints[child].tolist()],
                            color=BONE_COLORS[bone_idx],
                            lineWidth=3,
                        )

                    for jidx in [0, 3, 6, 9, 12, 15]:
                        session.upsert @ Sphere(
                            key=f"joint_{person_idx}_{jidx}",
                            args=[0.003, 8, 8],
                            position=joints[jidx].tolist(),
                            materialType="standard",
                            color="#ffffff",
                        )


            # Show source image for current frame
            if image_data_uris and frame < len(image_data_uris):
                session.upsert(
                    Img(src=image_data_uris[frame], key="source-image", style=SOURCE_IMG_STYLE),
                    to="htmlChildren",
                )

            frame = (frame + 1) % num_frames
            await sleep(1.0 / fps)


if __name__ == "__main__":
    main()
