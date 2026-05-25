#!/usr/bin/env python3
"""Convert the bluerov2_gz Heavy SDF/Collada model to OpenUSD.

This is intentionally narrow: it supports the SDF + DAE shape used by the
clydemcqueen/bluerov2_gz models/bluerov2_heavy asset so the real Heavy visual
can be used as an optional Isaac Sim demo overlay without adding a dependency.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

DEFAULT_MODEL_DIR = Path("/tmp/oceanscale-asset-candidates/bluerov2_gz/models/bluerov2_heavy")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "isaacsim"
    / "external_assets"
    / "bluerov2_gz_heavy"
    / "bluerov2_gz_heavy.usdc"
)
ROOT_PRIM = "/BlueROV2HeavyGz"


@dataclass(frozen=True)
class VisualMesh:
    link_name: str
    visual_name: str
    mesh_path: Path
    link_pose: tuple[float, float, float, float, float, float]
    visual_pose: tuple[float, float, float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class ColladaMesh:
    points: list[Any]
    normals: list[Any]
    face_vertex_counts: list[int]
    face_vertex_indices: list[int]
    material_face_indices: dict[str, list[int]]
    material_colors: dict[str, tuple[float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bluerov2_gz BlueROV2 Heavy SDF/DAE assets to OpenUSD.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to bluerov2_gz/models/bluerov2_heavy.",
    )
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output USD/USDC path.",
    )
    parser.add_argument(
        "--source-repo-url",
        default="https://github.com/clydemcqueen/bluerov2_gz",
        help="Source repository URL written into USD provenance.",
    )
    return parser.parse_args()


def _xml_namespace(root: ET.Element) -> dict[str, str]:
    if root.tag.startswith("{"):
        return {"c": root.tag.split("}", maxsplit=1)[0].strip("{")}
    return {"c": ""}


def _child_text(element: ET.Element, path: str, default: str = "") -> str:
    found = element.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _parse_pose(text: str | None) -> tuple[float, float, float, float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [float(value) for value in text.split()]
    if len(values) != 6:
        raise ValueError(f"SDF pose must have 6 values, got {len(values)}: {text!r}")
    return tuple(values)  # type: ignore[return-value]


def _parse_vec3(text: str | None, default: float = 1.0) -> tuple[float, float, float]:
    if not text:
        return (default, default, default)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"Vector must have 3 values, got {len(values)}: {text!r}")
    return tuple(values)  # type: ignore[return-value]


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    if not cleaned:
        return "Prim"
    if cleaned[0].isdigit():
        return f"_{cleaned}"
    return cleaned


def _find_repo_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_head(path: Path) -> str | None:
    repo_root = _find_repo_root(path)
    if repo_root is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_license(model_dir: Path) -> str | None:
    repo_root = _find_repo_root(model_dir)
    if repo_root is None:
        return None
    package_xml = repo_root / "package.xml"
    if not package_xml.exists():
        return None
    root = ET.parse(package_xml).getroot()
    return _child_text(root, "license") or None


def _mesh_path_from_uri(model_dir: Path, uri: str) -> Path:
    prefix = "model://bluerov2_heavy/"
    if uri.startswith(prefix):
        return model_dir / uri.removeprefix(prefix)
    path = Path(uri)
    if path.is_absolute():
        return path
    return model_dir / path


def parse_sdf_visual_meshes(model_dir: Path) -> list[VisualMesh]:
    sdf_path = model_dir / "model.sdf"
    root = ET.parse(sdf_path).getroot()
    visual_meshes: list[VisualMesh] = []
    for link in root.findall(".//link"):
        link_name = link.attrib["name"]
        link_pose = _parse_pose(_child_text(link, "pose"))
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            uri = _child_text(mesh, "uri")
            if not uri:
                continue
            visual_meshes.append(
                VisualMesh(
                    link_name=link_name,
                    visual_name=visual.attrib.get("name", "visual"),
                    mesh_path=_mesh_path_from_uri(model_dir, uri),
                    link_pose=link_pose,
                    visual_pose=_parse_pose(_child_text(visual, "pose")),
                    scale=_parse_vec3(_child_text(mesh, "scale"), default=1.0),
                )
            )
    return visual_meshes


def _float_source(mesh: ET.Element, ns: dict[str, str], source_id: str) -> list[tuple[float, ...]]:
    source = mesh.find(f"c:source[@id='{source_id}']", ns)
    if source is None:
        raise ValueError(f"Missing Collada source: {source_id}")
    float_array = source.find("c:float_array", ns)
    if float_array is None or float_array.text is None:
        raise ValueError(f"Missing float_array for Collada source: {source_id}")
    accessor = source.find("c:technique_common/c:accessor", ns)
    stride = int(accessor.attrib.get("stride", "1")) if accessor is not None else 1
    values = [float(value) for value in float_array.text.split()]
    return [tuple(values[index : index + stride]) for index in range(0, len(values), stride)]


def _vertices_position_source(mesh: ET.Element, ns: dict[str, str], vertices_id: str) -> str:
    vertices = mesh.find(f"c:vertices[@id='{vertices_id}']", ns)
    if vertices is None:
        raise ValueError(f"Missing Collada vertices: {vertices_id}")
    position_input = vertices.find("c:input[@semantic='POSITION']", ns)
    if position_input is None:
        raise ValueError(f"Missing POSITION input for Collada vertices: {vertices_id}")
    return position_input.attrib["source"].lstrip("#")


def _node_matrix(root: ET.Element, ns: dict[str, str]) -> tuple[float, ...]:
    matrix = root.find(".//c:library_visual_scenes//c:node/c:matrix", ns)
    if matrix is None or matrix.text is None:
        return (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    values = tuple(float(value) for value in matrix.text.split())
    if len(values) != 16:
        raise ValueError(f"Collada matrix must have 16 values, got {len(values)}")
    return values


def _transform_point(point: tuple[float, ...], matrix: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z = point[:3]
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def _transform_normal(normal: tuple[float, ...], matrix: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z = normal[:3]
    transformed = (
        matrix[0] * x + matrix[1] * y + matrix[2] * z,
        matrix[4] * x + matrix[5] * y + matrix[6] * z,
        matrix[8] * x + matrix[9] * y + matrix[10] * z,
    )
    length = math.sqrt(sum(value * value for value in transformed))
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(value / length for value in transformed)  # type: ignore[return-value]


def _effect_colors(root: ET.Element, ns: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    colors: dict[str, tuple[float, float, float]] = {}
    for effect in root.findall(".//c:library_effects/c:effect", ns):
        effect_id = effect.attrib.get("id")
        if effect_id is None:
            continue
        color = effect.find(".//c:diffuse/c:color", ns)
        if color is None or color.text is None:
            colors[effect_id] = (0.72, 0.74, 0.72)
            continue
        values = [float(value) for value in color.text.split()]
        colors[effect_id] = tuple(values[:3])  # type: ignore[assignment]
    return colors


def _material_colors(root: ET.Element, ns: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    effect_colors = _effect_colors(root, ns)
    colors: dict[str, tuple[float, float, float]] = {}
    for material in root.findall(".//c:library_materials/c:material", ns):
        material_id = material.attrib.get("id")
        if material_id is None:
            continue
        instance_effect = material.find("c:instance_effect", ns)
        effect_id = instance_effect.attrib["url"].lstrip("#") if instance_effect is not None else ""
        colors[material_id] = effect_colors.get(effect_id, (0.72, 0.74, 0.72))
    return colors


def parse_collada_mesh(path: Path) -> ColladaMesh:
    from pxr import Gf

    root = ET.parse(path).getroot()
    ns = _xml_namespace(root)
    geometry = root.find(".//c:library_geometries/c:geometry", ns)
    if geometry is None:
        raise ValueError(f"Collada file has no geometry: {path}")
    mesh = geometry.find("c:mesh", ns)
    if mesh is None:
        raise ValueError(f"Collada geometry has no mesh: {path}")

    matrix = _node_matrix(root, ns)
    material_colors = _material_colors(root, ns)
    sources = {
        source.attrib["id"]: _float_source(mesh, ns, source.attrib["id"])
        for source in mesh.findall("c:source", ns)
    }
    points: list[Any] = []
    normals: list[Any] = []
    face_vertex_indices: list[int] = []
    face_vertex_counts: list[int] = []
    material_face_indices: dict[str, list[int]] = {}

    position_source_id: str | None = None
    for triangles in mesh.findall("c:triangles", ns):
        inputs = triangles.findall("c:input", ns)
        offsets = {entry.attrib["semantic"]: int(entry.attrib["offset"]) for entry in inputs}
        input_sources = {entry.attrib["semantic"]: entry.attrib["source"].lstrip("#") for entry in inputs}
        if "VERTEX" not in offsets:
            raise ValueError(f"Collada triangles without VERTEX input: {path}")
        if position_source_id is None:
            position_source_id = _vertices_position_source(mesh, ns, input_sources["VERTEX"])
            points = [
                Gf.Vec3f(*_transform_point(position, matrix))
                for position in sources[position_source_id]
            ]
        normal_source = sources.get(input_sources.get("NORMAL", ""))
        stride = max(offsets.values()) + 1
        p_node = triangles.find("c:p", ns)
        if p_node is None or p_node.text is None:
            raise ValueError(f"Collada triangles without index buffer: {path}")
        raw_indices = [int(value) for value in p_node.text.split()]
        vertex_offset = offsets["VERTEX"]
        normal_offset = offsets.get("NORMAL")
        triangle_count = int(triangles.attrib["count"])
        face_start = len(face_vertex_counts)
        for triangle in range(triangle_count):
            face_vertex_counts.append(3)
            for corner in range(3):
                base = (triangle * 3 + corner) * stride
                face_vertex_indices.append(raw_indices[base + vertex_offset])
                if normal_source is not None and normal_offset is not None:
                    normal_index = raw_indices[base + normal_offset]
                    normals.append(Gf.Vec3f(*_transform_normal(normal_source[normal_index], matrix)))
        material = triangles.attrib.get("material", "default_material")
        material_face_indices.setdefault(material, []).extend(
            range(face_start, face_start + triangle_count)
        )

    if not points:
        raise ValueError(f"Collada file produced no points: {path}")
    return ColladaMesh(
        points=points,
        normals=normals,
        face_vertex_counts=face_vertex_counts,
        face_vertex_indices=face_vertex_indices,
        material_face_indices=material_face_indices,
        material_colors=material_colors,
    )


def _make_material(stage: Any, path: str, color: tuple[float, float, float]) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.56)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _apply_pose(xformable: Any, pose: tuple[float, float, float, float, float, float]) -> None:
    from pxr import Gf

    x, y, z, roll, pitch, yaw = pose
    if (x, y, z) != (0.0, 0.0, 0.0):
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    if (roll, pitch, yaw) != (0.0, 0.0, 0.0):
        xformable.AddRotateXYZOp().Set(
            Gf.Vec3f(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
        )


def _author_mesh(
    stage: Any,
    path: str,
    collada_mesh: ColladaMesh,
    materials: dict[str, Any],
) -> None:
    from pxr import UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(collada_mesh.points)
    mesh.CreateFaceVertexCountsAttr(collada_mesh.face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(collada_mesh.face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    if collada_mesh.normals:
        mesh.CreateNormalsAttr(collada_mesh.normals)
        mesh.SetNormalsInterpolation("faceVarying")

    for material_id, face_indices in collada_mesh.material_face_indices.items():
        subset_name = _sanitize_name(material_id)
        subset = UsdGeom.Subset.Define(stage, f"{path}/{subset_name}")
        subset.CreateElementTypeAttr("face")
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(face_indices)
        material = materials.get(material_id)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)


def convert_model_to_usd(
    model_dir: Path,
    output_usd: Path,
    *,
    source_repo_url: str = "https://github.com/clydemcqueen/bluerov2_gz",
) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom

    model_dir = model_dir.resolve()
    visual_meshes = parse_sdf_visual_meshes(model_dir)
    if not visual_meshes:
        raise ValueError(f"No SDF visual meshes found in {model_dir}")

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT_PRIM)
    stage.SetDefaultPrim(root.GetPrim())
    source_commit = _git_head(model_dir)
    source_license = _package_license(model_dir)
    root.GetPrim().CreateAttribute("oceanscale:assetRole", Sdf.ValueTypeNames.String).Set(
        "external_high_fidelity_mesh"
    )
    root.GetPrim().CreateAttribute("oceanscale:vehicle", Sdf.ValueTypeNames.String).Set(
        "BlueROV2 Heavy"
    )
    root.GetPrim().CreateAttribute("oceanscale:sourceRepo", Sdf.ValueTypeNames.String).Set(
        source_repo_url
    )
    root.GetPrim().CreateAttribute("oceanscale:sourceCommit", Sdf.ValueTypeNames.String).Set(
        source_commit or "unknown"
    )
    root.GetPrim().CreateAttribute("oceanscale:sourceLicense", Sdf.ValueTypeNames.String).Set(
        source_license or "unknown"
    )
    root.GetPrim().CreateAttribute("oceanscale:sourceModel", Sdf.ValueTypeNames.String).Set(
        "models/bluerov2_heavy/model.sdf"
    )
    root.GetPrim().CreateAttribute("oceanscale:sourceAssetType", Sdf.ValueTypeNames.String).Set(
        "SDF + Collada visual mesh"
    )
    root.GetPrim().CreateAttribute("oceanscale:thrusterCount", Sdf.ValueTypeNames.Int).Set(8)

    mesh_cache: dict[Path, ColladaMesh] = {}
    materials: dict[tuple[Path, str], Any] = {}
    total_triangles = 0

    for visual in visual_meshes:
        collada_mesh = mesh_cache.get(visual.mesh_path)
        if collada_mesh is None:
            collada_mesh = parse_collada_mesh(visual.mesh_path)
            mesh_cache[visual.mesh_path] = collada_mesh
        link_path = f"{ROOT_PRIM}/{_sanitize_name(visual.link_name)}"
        link_xform = UsdGeom.Xform.Define(stage, link_path)
        _apply_pose(UsdGeom.Xformable(link_xform.GetPrim()), visual.link_pose)

        visual_path = f"{link_path}/{_sanitize_name(visual.visual_name)}"
        visual_xform = UsdGeom.Xform.Define(stage, visual_path)
        visual_xformable = UsdGeom.Xformable(visual_xform.GetPrim())
        _apply_pose(visual_xformable, visual.visual_pose)
        if visual.scale != (1.0, 1.0, 1.0):
            visual_xformable.AddScaleOp().Set(Gf.Vec3f(*visual.scale))
        visual_xform.GetPrim().CreateAttribute("oceanscale:sourceMesh", Sdf.ValueTypeNames.Asset).Set(
            str(visual.mesh_path)
        )

        for material_id, color in collada_mesh.material_colors.items():
            key = (visual.mesh_path, material_id)
            if key not in materials:
                mat_path = (
                    f"{ROOT_PRIM}/Materials/"
                    f"{_sanitize_name(visual.mesh_path.stem)}_{_sanitize_name(material_id)}"
                )
                materials[key] = _make_material(stage, mat_path, color)
        visual_materials = {
            material_id: materials[(visual.mesh_path, material_id)]
            for material_id in collada_mesh.material_colors
            if (visual.mesh_path, material_id) in materials
        }
        _author_mesh(stage, f"{visual_path}/mesh", collada_mesh, visual_materials)
        total_triangles += len(collada_mesh.face_vertex_counts)

    root.GetPrim().CreateAttribute("oceanscale:visualMeshCount", Sdf.ValueTypeNames.Int).Set(
        len(visual_meshes)
    )
    root.GetPrim().CreateAttribute("oceanscale:uniqueColladaMeshCount", Sdf.ValueTypeNames.Int).Set(
        len(mesh_cache)
    )
    root.GetPrim().CreateAttribute("oceanscale:triangleInstanceCount", Sdf.ValueTypeNames.Int).Set(
        total_triangles
    )
    stage.GetRootLayer().Save()
    return {
        "output_usd": str(output_usd),
        "root_prim": ROOT_PRIM,
        "visual_mesh_count": len(visual_meshes),
        "unique_collada_mesh_count": len(mesh_cache),
        "triangle_instance_count": total_triangles,
        "source_commit": source_commit,
        "source_license": source_license,
    }


def main() -> None:
    args = parse_args()
    summary = convert_model_to_usd(
        args.model_dir,
        args.output_usd,
        source_repo_url=args.source_repo_url,
    )
    print(summary)


if __name__ == "__main__":
    main()
