"""``apyrobo mcp keepout``: the mask must mark the same places the zones do."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
import yaml

from apyrobo.mcp_keepout import KEEPOUT, MAX_PIXELS, build_mask, nav2_config, write_mask


def zone(x_min: float, y_min: float, x_max: float, y_max: float) -> dict[str, float]:
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    magic, size, maxval, pixels = path.read_bytes().split(b"\n", 3)
    assert (magic, maxval) == (b"P5", b"255")
    width, height = (int(n) for n in size.split())
    return width, height, pixels


def black_cells(path: Path) -> set[tuple[float, float]]:
    """The lower-left corner, in metres, of each black cell, read as Nav2 does:

    the YAML's origin is the lower-left pixel and the image's first row is the top.
    """
    meta = yaml.safe_load(path.read_text())
    width, height, pixels = read_pgm(path.parent / meta["image"])
    assert len(pixels) == width * height
    res, (ox, oy, yaw) = meta["resolution"], meta["origin"]
    assert yaw == 0
    return {
        (ox + col * res, oy + (height - 1 - row) * res)
        for row in range(height)
        for col in range(width)
        if pixels[row * width + col] == KEEPOUT
    }


def test_cells_land_where_the_zones_are_with_y_up(tmp_path):
    # Two zones, so a flipped y would show. Same layout loaded in Nav2's map_server
    # on Jazzy and Humble: /keepout_filter_mask held 100 at these cells only.
    mask = build_mask([zone(-1, -1, 0, 0), zone(2, 1, 3, 1.5)], resolution=0.5)
    _, yaml_path = write_mask(mask, tmp_path)
    assert (mask.width, mask.height) == (12, 9)
    assert black_cells(yaml_path) == {
        (-1, -1), (-0.5, -1), (-1, -0.5), (-0.5, -0.5), (2, 1), (2.5, 1),
    }


def test_the_map_yaml_says_black_is_occupied(tmp_path):
    _, yaml_path = write_mask(build_mask([zone(0, 0, 1, 1)]), tmp_path)
    meta = yaml.safe_load(yaml_path.read_text())
    assert meta["image"] == "keepout_mask.pgm"
    assert meta["mode"] == "scale" and meta["negate"] == 0
    assert 0 < meta["free_thresh"] < meta["occupied_thresh"] < 1


def test_a_zone_off_the_grid_is_grown_not_shrunk(tmp_path):
    _, yaml_path = write_mask(build_mask([zone(0.1, 0.1, 0.2, 0.2)], resolution=1.0), tmp_path)
    cells = black_cells(yaml_path)
    assert len(cells) == 1
    x, y = cells.pop()
    assert x <= 0.1 and x + 1 >= 0.2 and y <= 0.1 and y + 1 >= 0.2


def test_a_zero_width_zone_still_marks_a_pixel(tmp_path):
    _, yaml_path = write_mask(build_mask([zone(1, 1, 1, 2)], resolution=0.5), tmp_path)
    assert len(black_cells(yaml_path)) == 2


def test_no_zones_and_bad_resolutions_are_refused():
    with pytest.raises(ValueError, match="no no-go zones"):
        build_mask([])
    for resolution in (0, -0.05, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="resolution"):
            build_mask([zone(0, 0, 1, 1)], resolution=resolution)


def test_a_huge_mask_is_refused_before_it_is_allocated():
    with pytest.raises(ValueError, match=str(MAX_PIXELS)):
        build_mask([zone(0, 0, 100000, 100000)])


def test_the_config_covers_both_costmaps_and_names_the_mask():
    mask = Path("/maps/keepout_mask.yaml")
    config = yaml.safe_load(nav2_config(mask))
    assert config["keepout_filter_mask_server"]["ros__parameters"]["yaml_filename"] == str(mask)
    mask_server = config["keepout_filter_mask_server"]["ros__parameters"]
    info = config["keepout_costmap_filter_info_server"]["ros__parameters"]
    assert info["type"] == 0
    assert info["mask_topic"] == mask_server["topic_name"]
    for costmap in ("global_costmap", "local_costmap"):
        params = config[costmap][costmap]["ros__parameters"]
        assert params["filters"] == ["keepout_filter"]
        assert params["keepout_filter"]["plugin"] == "nav2_costmap_2d::KeepoutFilter"
        assert params["keepout_filter"]["filter_info_topic"] == info["filter_info_topic"]


@pytest.mark.parametrize("mask", [
    PureWindowsPath(r"C:\Users\robot\maps\keepout_mask.yaml"),
    PurePosixPath('/maps/a "quoted" name/keepout_mask.yaml'),
    PurePosixPath("/maps/\u00e9t\u00e9/keepout_mask.yaml"),
])
def test_any_mask_path_survives_the_yaml(mask):
    config = yaml.safe_load(nav2_config(mask))
    assert config["keepout_filter_mask_server"]["ros__parameters"]["yaml_filename"] == str(mask)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "keepout", *args],
        capture_output=True, text=True, timeout=60,
    )


def test_cli_draws_policy_file_zones_and_flags_together(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"no_go_zones": [{"name": "dock", **zone(0, 0, 0.5, 0.5)}]}))
    out = tmp_path / "masks"
    result = run_cli("--policy-file", str(policy), "--no-go", "-2,-2,-1.5,-1.5",
                     "--resolution", "0.5", "--out", str(out))
    assert result.returncode == 0, result.stderr
    yaml_path = out / "keepout_mask.yaml"
    assert black_cells(yaml_path) == {(0, 0), (-2, -2)}
    config = yaml.safe_load(result.stdout)
    assert config["keepout_filter_mask_server"]["ros__parameters"]["yaml_filename"] == (
        str(yaml_path.resolve())
    )


def test_cli_errors_are_one_readable_line(tmp_path):
    no_zones = run_cli("--out", str(tmp_path))
    assert no_zones.returncode == 1
    assert no_zones.stderr.startswith("Error: no no-go zones")
    blocked = tmp_path / "file"
    blocked.write_text("")
    no_folder = run_cli("--no-go", "0,0,1,1", "--out", str(blocked))
    assert no_folder.returncode == 1
    assert no_folder.stderr.startswith("Error: ")
    assert "Traceback" not in no_folder.stderr
