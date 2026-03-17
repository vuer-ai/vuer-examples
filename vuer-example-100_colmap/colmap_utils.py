"""Utilities for reading COLMAP reconstruction data (binary format)."""

import struct
import numpy as np
from pathlib import Path
from collections import namedtuple
from typing import Dict, Tuple

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODEL_NAMES = [
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL",
    "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
]
CAMERA_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5,
    "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12,
}

# COLMAP: [right, down, forward] -> Three.js: [right, up, backward]
COLMAP_TO_THREEJS = np.diag([1.0, -1.0, -1.0, 1.0])


def _unpack(fid, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


# ── Binary readers ──────────────────────────────────────────────────────────


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        (n,) = _unpack(f, 8, "Q")
        for _ in range(n):
            cam_id, model_id, w, h = _unpack(f, 24, "iiQQ")
            model = CAMERA_MODEL_NAMES[model_id]
            num_p = CAMERA_MODEL_PARAMS[model]
            params = _unpack(f, 8 * num_p, "d" * num_p)
            cameras[cam_id] = Camera(cam_id, model, w, h, params)
    return cameras


def read_images_binary(path: Path) -> Dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        (n,) = _unpack(f, 8, "Q")
        for _ in range(n):
            props = _unpack(f, 64, "idddddddi")
            img_id, qvec, tvec, cam_id = props[0], np.array(props[1:5]), np.array(props[5:8]), props[8]

            name = b""
            ch = f.read(1)
            while ch != b"\x00":
                name += ch
                ch = f.read(1)
            name = name.decode("utf-8")

            (n2d,) = _unpack(f, 8, "Q")
            if n2d:
                raw = _unpack(f, 24 * n2d, "ddq" * n2d)
                xys = np.column_stack([raw[0::3], raw[1::3]]).astype(np.float64)
                pt3d_ids = np.array(raw[2::3], dtype=np.int64)
            else:
                xys, pt3d_ids = np.zeros((0, 2)), np.zeros(0, dtype=np.int64)

            images[img_id] = Image(img_id, qvec, tvec, cam_id, name, xys, pt3d_ids)
    return images


def read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    points = {}
    with open(path, "rb") as f:
        (n,) = _unpack(f, 8, "Q")
        for _ in range(n):
            props = _unpack(f, 43, "QdddBBBd")
            pid, xyz, rgb, err = props[0], np.array(props[1:4]), np.array(props[4:7]), props[7]
            (track_len,) = _unpack(f, 8, "Q")
            track = _unpack(f, 8 * track_len, "ii" * track_len)
            points[pid] = Point3D(pid, xyz, rgb, err, np.array(track[0::2]), np.array(track[1::2]))
    return points


# ── PLY reader ──────────────────────────────────────────────────────────────


def read_ply_point_cloud(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a binary PLY point cloud. Returns (vertices, colors) as Nx3 float32, colors in [0,1]."""
    dtype_map = {"float": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4"}

    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        num_vertices, props, in_vertex = 0, [], False

        for line in f:
            line = line.decode("utf-8").strip()
            if line.startswith("element vertex"):
                num_vertices, in_vertex = int(line.split()[-1]), True
            elif line.startswith("element"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                _, ptype, pname = line.split()
                props.append((pname, dtype_map.get(ptype, "f4")))
            elif line == "end_header":
                break

        data = np.fromfile(f, dtype=props, count=num_vertices)

    verts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    if "red" in data.dtype.names:
        colors = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.float32)
        if colors.max() > 1.0:
            colors /= 255.0
    else:
        colors = np.ones_like(verts)
    return verts, colors


# ── Geometry helpers ────────────────────────────────────────────────────────


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ])


def get_c2w(image: Image) -> np.ndarray:
    """Camera-to-world 4x4 from a COLMAP Image (COLMAP stores w2c, this inverts it)."""
    R = qvec2rotmat(image.qvec)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ image.tvec
    return T


def c2w_to_threejs(c2w: np.ndarray) -> np.ndarray:
    """Convert COLMAP camera-to-world to Three.js convention."""
    return c2w @ COLMAP_TO_THREEJS


def mat4_to_column_major(m: np.ndarray) -> list:
    """4x4 numpy matrix to Three.js column-major flat list."""
    return m.T.flatten().tolist()


def get_fov_aspect(camera: Camera) -> Tuple[float, float]:
    """Vertical FOV (degrees) and aspect ratio from a COLMAP Camera."""
    if camera.model == "PINHOLE":
        fy = camera.params[1]
    elif camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fy = camera.params[0]
    elif camera.model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fy = camera.params[1]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    fov = np.degrees(2 * np.arctan(camera.height / (2 * fy)))
    return fov, camera.width / camera.height
