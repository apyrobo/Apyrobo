"""``apyrobo mcp keepout``: the map YAML must carry the origin and resolution exactly.

Nav2's map server places the mask from these two numbers alone, so any digit lost
in the YAML moves every black pixel off the zone it was drawn for.
"""

from __future__ import annotations

import pytest
import yaml

from apyrobo.mcp_keepout import build_mask, mask_yaml, write_mask
from tests.test_mcp_keepout import black_cells, zone


@pytest.mark.parametrize(
    "z",
    [
        zone(1234.567, 10, 1236, 12),
        # UTM-sized coordinates: easting ~6.5e5, northing ~5.4e6.
        zone(652341.27, 5412345.68, 652345, 5412349),
        zone(-98765.4321, -12345.678, -98760, -12340),
    ],
)
def test_yaml_origin_is_the_origin_the_mask_was_drawn_from(z):
    mask = build_mask([z])
    meta = yaml.safe_load(mask_yaml(mask, "keepout_mask.pgm"))
    assert meta["origin"] == [mask.origin_x, mask.origin_y, 0.0]
    assert meta["resolution"] == mask.resolution


@pytest.mark.parametrize("resolution", [0.05, 1 / 30, 0.123456789])
def test_yaml_resolution_keeps_every_digit(resolution):
    mask = build_mask([zone(0, 0, 1, 1)], resolution=resolution)
    meta = yaml.safe_load(mask_yaml(mask, "keepout_mask.pgm"))
    assert meta["resolution"] == resolution


def test_a_zone_far_from_zero_is_covered_where_nav2_places_the_mask(tmp_path):
    z = zone(652341.27, 5412345.68, 652345, 5412349)
    res = 0.05
    _, yaml_path = write_mask(build_mask([z], resolution=res), tmp_path)
    cells = black_cells(yaml_path)
    # Read back as Nav2 would, the black cells still cover the whole zone.
    assert min(x for x, _ in cells) <= z["x_min"] + 1e-6
    assert max(x for x, _ in cells) + res >= z["x_max"] - 1e-6
    assert min(y for _, y in cells) <= z["y_min"] + 1e-6
    assert max(y for _, y in cells) + res >= z["y_max"] - 1e-6
    # And they do not drift more than a pixel past it.
    assert min(x for x, _ in cells) >= z["x_min"] - res - 1e-6
    assert min(y for _, y in cells) >= z["y_min"] - res - 1e-6


def test_small_origins_still_read_as_plain_numbers():
    mask = build_mask([zone(0, 0, 1, 1)])
    text = mask_yaml(mask, "keepout_mask.pgm")
    assert "origin: [-1.0, -1.0, 0.0]" in text
    assert "resolution: 0.05\n" in text
