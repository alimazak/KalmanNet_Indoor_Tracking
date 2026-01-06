#!/usr/bin/env python3
"""
gen_sensor_world.py

Generate a Gazebo SDF world by inserting a grid of visual-only "sensor/beacon" models.

- Outputs:
  1) .world (SDF) file with inserted <model> blocks
  2) optional CSV layout file (x,y) for your measurement generator

Insertion strategy:
  - If template contains the marker: <!-- INSERT_SENSORS_HERE -->
    sensors are inserted there (recommended).
  - Else: sensors are inserted right before the last </world> tag.

Grid:
  - n_side x n_side points, centered at (center_x, center_y)
  - b is the side length in meters
  - row-major ordering: y then x  (matches your range generator convention)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


INSERT_MARKER = "<!-- INSERT_SENSORS_HERE -->"


def make_grid(n_side: int, b: float, center_x: float = 0.0, center_y: float = 0.0) -> List[Tuple[float, float]]:
    if n_side <= 1:
        return [(center_x, center_y)]

    half = 0.5 * float(b)
    step = float(b) / float(n_side - 1)

    xs = [center_x - half + i * step for i in range(n_side)]
    ys = [center_y - half + j * step for j in range(n_side)]

    # row-major: y first, then x
    return [(x, y) for y in ys for x in xs]


def sensor_model(name: str, x: float, y: float, z: float, radius: float, length: float) -> str:
    # visual only (no collision) -> robot çarpmaz
    return f"""
  <model name="{name}">
    <static>true</static>
    <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius:.6f}</radius>
            <length>{length:.6f}</length>
          </cylinder>
        </geometry>
      </visual>
    </link>
  </model>
"""


def write_layout_csv(path: Path, points: Iterable[Tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# x,y (sensor layout)\n")
        for (x, y) in points:
            f.write(f"{x:.6f},{y:.6f}\n")


def insert_models_into_world(template_text: str, models_text: str) -> str:
    if INSERT_MARKER in template_text:
        return template_text.replace(INSERT_MARKER, models_text)

    idx = template_text.rfind("</world>")
    if idx < 0:
        raise RuntimeError("Template has no </world> tag. Is it an SDF .world file?")
    return template_text[:idx] + models_text + "\n" + template_text[idx:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template world file path (e.g., empty.world)")
    ap.add_argument("--out", required=True, help="Output world file path (e.g., paper_sensors.world)")
    ap.add_argument("--layout_out", default="", help="Optional CSV output for sensor layout (x,y).")

    ap.add_argument("--n_side", type=int, default=5)
    ap.add_argument("--b", type=float, default=8.0, help="Grid side length (m)")
    ap.add_argument("--center_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)

    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--length", type=float, default=0.5)
    ap.add_argument("--z", type=float, default=None, help="If None -> length/2 (sit on ground)")
    ap.add_argument("--name_prefix", default="paper_sensor_")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    args = ap.parse_args()

    template_path = Path(args.template).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    layout_path = Path(args.layout_out).expanduser().resolve() if args.layout_out.strip() else None

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    if out_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output exists: {out_path} (use --overwrite)")

    if layout_path is not None and layout_path.exists() and not args.overwrite:
        raise RuntimeError(f"Layout exists: {layout_path} (use --overwrite)")

    z = float(args.z) if args.z is not None else (float(args.length) / 2.0)

    txt = template_path.read_text(encoding="utf-8")

    points = make_grid(args.n_side, args.b, center_x=args.center_x, center_y=args.center_y)

    models: List[str] = []
    for i, (x, y) in enumerate(points):
        models.append(sensor_model(f"{args.name_prefix}{i:02d}", x, y, z, args.radius, args.length))

    insert = "\n  <!-- Paper sensors (visual only, no collision) -->\n" + "".join(models) + "\n"
    out_txt = insert_models_into_world(txt, insert)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_txt, encoding="utf-8")

    if layout_path is not None:
        write_layout_csv(layout_path, points)

    print(f"Generated world:  {out_path}  ({len(points)} sensors)")
    if layout_path is not None:
        print(f"Generated layout: {layout_path}")
    if INSERT_MARKER in txt:
        print("Insert mode: marker")
    else:
        print("Insert mode: </world> (fallback)")


if __name__ == "__main__":
    main()
