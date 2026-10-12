import numpy as np
from pathlib import Path
from typing import Tuple, Union

import trimesh
from scipy.spatial.transform import Rotation


def parse_poses(path: str) -> dict:
    """GT_pose.txt (w2c convention) -> {name: 4x4 w2c ndarray}.

  Format: ``<name> <qw> <qx> <qy> <qz> <tx> <ty> <tz>`` where R,t are the
  world-to-camera transform (UAVD4L-LoD).
    """
    poses = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            name = parts[0]
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz      = map(float, parts[5:8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = R
            w2c[:3,  3] = [tx, ty, tz]
            poses[name] = w2c
    return poses


def load_intrinsics(path: str, image_name: str):
    """intrinsic.txt -> (fx, fy, cx, cy, W, H)."""
    stem = Path(image_name).stem
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8:
                continue
            if p[0] == image_name or Path(p[0]).stem == stem:
                W, H = int(p[2]), int(p[3])
                fx, fy, cx, cy = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                return fx, fy, cx, cy, W, H
    raise ValueError(f"No intrinsics for {image_name}")


def scale_K(fx, fy, cx, cy, src_W, src_H, dst_W, dst_H) -> np.ndarray:
    sx, sy = dst_W / src_W, dst_H / src_H
    return np.array([[fx * sx, 0, cx * sx],
                     [0, fy * sy, cy * sy],
                     [0, 0, 1]], dtype=np.float64)

def load_mesh_vertices_faces(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices (V, 3) float64 and faces (F, 3) int64 from .obj or .ply."""
    path = Path(path)
    if path.suffix.lower() not in {".obj", ".ply"}:
        raise ValueError(f"Unsupported mesh format: {path.suffix} (use .obj or .ply)")
    loaded = trimesh.load(str(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in loaded.geometry.values()
            ]
        )
    else:
        mesh = loaded
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def weld_mesh_vertices_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    digits_vertex: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices so edge-connected instancing is topologically correct."""
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.merge_vertices(digits_vertex=int(digits_vertex))
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _explode_face_colors_to_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate vertices per face so each face keeps a distinct color."""
    verts = vertices[faces.reshape(-1)].reshape(-1, 3)
    new_faces = np.arange(len(faces) * 3, dtype=np.int64).reshape(-1, 3)
    vertex_colors = np.repeat(face_colors, 3, axis=0)
    return verts, new_faces, vertex_colors


def _save_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> None:
    verts, faces, vertex_colors = _explode_face_colors_to_vertices(vertices, faces, face_colors)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = vertex_colors
    mesh.export(str(path))


def _save_colored_obj_mtl(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> None:
    """Write OBJ + MTL with one material per instance color (Blender / MeshLab)."""
    mtl_path = path.with_suffix(".mtl")
    rgb_to_mat: dict[tuple[int, int, int], str] = {}
    mat_lines = []

    unique_rgbs = np.unique(face_colors[:, :3], axis=0)
    for idx, rgb in enumerate(unique_rgbs):
        key = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        name = f"inst_{idx:06d}"
        rgb_to_mat[key] = name
        mat_lines.append(
            f"newmtl {name}\n"
            f"Kd {key[0] / 255.0:.6f} {key[1] / 255.0:.6f} {key[2] / 255.0:.6f}\n"
            "illum 1\n"
        )
    mtl_path.write_text("".join(mat_lines))

    with path.open("w", encoding="ascii") as obj_f:
        obj_f.write(f"mtllib {mtl_path.name}\n")
        for v in vertices:
            obj_f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        current_mat = None
        for fi, face in enumerate(faces):
            key = tuple(int(x) for x in face_colors[fi, :3])
            mat = rgb_to_mat[key]
            if mat != current_mat:
                obj_f.write(f"usemtl {mat}\n")
                current_mat = mat
            a, b, c = face + 1
            obj_f.write(f"f {a} {b} {c}\n")


def save_instance_colored_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    path: Union[str, Path],
) -> None:
    """Save mesh colored by building instance (.ply vertex colors, .obj + .mtl)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_colors = np.asarray(face_colors, dtype=np.uint8)

    if suf == ".ply":
        _save_colored_ply(path, vertices, faces, face_colors)
    elif suf == ".obj":
        _save_colored_obj_mtl(path, vertices, faces, face_colors)
    else:
        raise ValueError(f"Unsupported mesh format: {suf} (use .obj or .ply)")
