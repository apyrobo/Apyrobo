"""``apyrobo mcp keepout``: turn no-go zones into a Nav2 keepout mask.

``apyrobo mcp`` refuses moves into its no-go zones, but it does not watch the
drive. A Nav2 keepout filter works inside the costmaps, so Nav2 plans and drives
around a zone. This writes that filter's mask from the same zones (``--no-go``
and ``--policy-file``), and prints the config to paste into Nav2, so one list
feeds both layers.

Pure Python, no ROS. A pixel goes black if any part of it touches a zone, so
the mask can only be a little larger than the zones, never smaller.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

#: Metres per pixel unless ``--resolution`` says otherwise.
DEFAULT_RESOLUTION = 0.05
#: Free space kept around the zones' bounding box, in metres.
MARGIN_M = 1.0
#: A bigger mask is almost always a typo in a coordinate or the resolution.
MAX_PIXELS = 25_000_000
#: Absorbs float error, so a zone edge on a pixel edge does not spill over it.
_EPS = 1e-9

FREE = 255
KEEPOUT = 0
#: Black is occupancy 1.0 and white 0.0, so these only need to sit between.
OCCUPIED_THRESH = 0.65
FREE_THRESH = 0.25


@dataclass(frozen=True)
class Mask:
    """A grayscale grid. ``rows[0]`` is the top row: the highest y in the map."""

    width: int
    height: int
    origin_x: float
    origin_y: float
    resolution: float
    rows: tuple[bytes, ...]


def build_mask(
    zones: Sequence[dict[str, Any]], resolution: float = DEFAULT_RESOLUTION
) -> Mask:
    """Draw ``zones`` (rectangles in the map frame, metres) on a grid."""
    if not zones:
        raise ValueError("no no-go zones to draw; give --no-go or a --policy-file with zones")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"resolution must be a positive number, got {resolution!r}")
    origin_x = round(min(z["x_min"] for z in zones) - MARGIN_M, 6)
    origin_y = round(min(z["y_min"] for z in zones) - MARGIN_M, 6)
    x_end = max(z["x_max"] for z in zones) + MARGIN_M
    y_end = max(z["y_max"] for z in zones) + MARGIN_M
    width = math.ceil((x_end - origin_x) / resolution - _EPS)
    height = math.ceil((y_end - origin_y) / resolution - _EPS)
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"a {width} x {height} mask is over the {MAX_PIXELS} pixel limit; "
            "check the zones, or raise --resolution"
        )
    grid = [bytearray([FREE]) * width for _ in range(height)]
    for zone in zones:
        first_col, end_col = _span(zone["x_min"], zone["x_max"], origin_x, resolution)
        first_row, end_row = _span(zone["y_min"], zone["y_max"], origin_y, resolution)
        for row in range(first_row, end_row):
            # The image's first row is the top of the map, so y counts up from the end.
            grid[height - 1 - row][first_col:end_col] = bytes([KEEPOUT]) * (end_col - first_col)
    return Mask(width, height, origin_x, origin_y, resolution, tuple(bytes(r) for r in grid))


def _span(low: float, high: float, origin: float, resolution: float) -> tuple[int, int]:
    """Pixels touching ``low..high`` on one axis, as ``first, end``; at least one."""
    first = math.floor((low - origin) / resolution + _EPS)
    end = math.ceil((high - origin) / resolution - _EPS)
    return first, max(end, first + 1)


def mask_yaml(mask: Mask, image: str) -> str:
    """The map YAML Nav2's map server reads the mask through."""
    return (
        f"image: {image}\n"
        "mode: scale\n"
        f"resolution: {mask.resolution:g}\n"
        f"origin: [{mask.origin_x:g}, {mask.origin_y:g}, 0.0]\n"
        "negate: 0\n"
        f"occupied_thresh: {OCCUPIED_THRESH}\n"
        f"free_thresh: {FREE_THRESH}\n"
    )


def write_mask(mask: Mask, out_dir: Path, name: str = "keepout_mask") -> tuple[Path, Path]:
    """Write ``<name>.pgm`` and ``<name>.yaml``, replacing any earlier ones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pgm = out_dir / f"{name}.pgm"
    yaml_path = out_dir / f"{name}.yaml"
    pgm.write_bytes(f"P5\n{mask.width} {mask.height}\n255\n".encode() + b"".join(mask.rows))
    yaml_path.write_text(mask_yaml(mask, pgm.name), encoding="utf-8")
    return pgm, yaml_path


def nav2_config(mask_path: PurePath) -> str:
    """The Nav2 parameters that load the mask and use it in both costmaps."""
    # A JSON string is a valid YAML double-quoted one: backslashes stay escaped.
    yaml_filename = json.dumps(str(mask_path), ensure_ascii=False)
    filter_lines = (
        '      filters: ["keepout_filter"]\n'
        "      keepout_filter:\n"
        '        plugin: "nav2_costmap_2d::KeepoutFilter"\n'
        "        enabled: true\n"
        '        filter_info_topic: "/costmap_filter_info"\n'
    )
    return (
        "# Servers: run these two with a lifecycle manager that lists both.\n"
        "keepout_filter_mask_server:\n"
        "  ros__parameters:\n"
        '    frame_id: "map"\n'
        '    topic_name: "/keepout_filter_mask"\n'
        f"    yaml_filename: {yaml_filename}\n"
        "keepout_costmap_filter_info_server:\n"
        "  ros__parameters:\n"
        "    type: 0\n"
        '    filter_info_topic: "/costmap_filter_info"\n'
        '    mask_topic: "/keepout_filter_mask"\n'
        "    base: 0.0\n"
        "    multiplier: 1.0\n"
        "\n"
        "# Costmaps: merge into your Nav2 params file.\n"
        "global_costmap:\n"
        "  global_costmap:\n"
        "    ros__parameters:\n"
        f"{filter_lines}"
        "local_costmap:\n"
        "  local_costmap:\n"
        "    ros__parameters:\n"
        f"{filter_lines}"
    )


def _zones(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The zones ``apyrobo mcp`` would load from the same flags, in the same order."""
    from apyrobo.mcp_server import load_policy_file, parse_zone
    from apyrobo.safety.enforcer import POLICY_REGISTRY

    policy_file = getattr(args, "ko_policy_file", None) or getattr(args, "policy_file", None)
    file_cfg = load_policy_file(policy_file) if policy_file else {}
    base_name = file_cfg.get("base") or "default"
    base = POLICY_REGISTRY.get(base_name)
    if base is None:
        raise ValueError(f"unknown policy {base_name!r}; choose from {sorted(POLICY_REGISTRY)}")
    no_go = list(getattr(args, "no_go", None) or []) + list(getattr(args, "ko_no_go", None) or [])
    return (
        list(base.collision_zones)
        + list(file_cfg.get("no_go_zones", []))
        + [parse_zone(z) for z in no_go]
    )


def main(args: argparse.Namespace) -> int:
    """Write the mask, print where it went and the config to paste."""
    out_dir = Path(args.ko_out)
    try:
        mask = build_mask(_zones(args), args.ko_resolution)
        _, yaml_path = write_mask(mask, out_dir)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {yaml_path} and its .pgm: {mask.width} x {mask.height} pixels at "
        f"{mask.resolution:g} m, in the map frame.",
        file=sys.stderr,
    )
    print(nav2_config(yaml_path.resolve()), end="")
    return 0
