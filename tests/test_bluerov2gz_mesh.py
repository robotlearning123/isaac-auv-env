"""Tests for BlueROV2 Heavy high-fidelity meshes from clydemcqueen/bluerov2_gz."""


from oceanscale.vehicles.bluerov2 import (
    bluerov2_heavy_gz_mesh_paths,
    bluerov2_heavy_gz_sdf_path,
    sand_heightmap_mesh_paths,
)


class TestBlueROV2GZMeshes:
    def test_hull_mesh_exists(self):
        paths = bluerov2_heavy_gz_mesh_paths()
        assert paths["hull"].exists()
        assert paths["hull"].suffix == ".dae"

    def test_prop_meshes_exist(self):
        paths = bluerov2_heavy_gz_mesh_paths()
        assert paths["prop_ccw"].exists()
        assert paths["prop_cw"].exists()

    def test_hull_mesh_has_content(self):
        paths = bluerov2_heavy_gz_mesh_paths()
        size = paths["hull"].stat().st_size
        assert size > 1_000_000, f"Hull mesh too small: {size} bytes"

    def test_sdf_model_exists(self):
        sdf = bluerov2_heavy_gz_sdf_path()
        assert sdf.exists()
        content = sdf.read_text()
        assert "bluerov2_heavy" in content
        assert "thruster" in content

    def test_sdf_has_8_thrusters(self):
        sdf = bluerov2_heavy_gz_sdf_path()
        content = sdf.read_text()
        thruster_count = content.count('<link name="thruster')
        assert thruster_count == 8, f"Expected 8 thrusters, got {thruster_count}"

    def test_sdf_has_hydro_params(self):
        sdf = bluerov2_heavy_gz_sdf_path()
        content = sdf.read_text()
        assert "xUabsU" in content
        assert "Hydrodynamics" in content


class TestSandHeightmap:
    def test_heightmap_exists(self):
        paths = sand_heightmap_mesh_paths()
        assert paths["heightmap"].exists()

    def test_seabed_exists(self):
        paths = sand_heightmap_mesh_paths()
        assert paths["seabed"].exists()

    def test_texture_exists(self):
        paths = sand_heightmap_mesh_paths()
        assert paths["texture"].exists()
        assert paths["texture"].suffix == ".jpg"
