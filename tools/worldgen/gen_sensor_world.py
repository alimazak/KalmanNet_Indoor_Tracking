import argparse
import os


def make_grid(n_side: int, b: float):
    half = 0.5 * b
    if n_side <= 1:
        return [(0.0, 0.0)]
    step = b / float(n_side - 1)
    xs = [-half + i * step for i in range(n_side)]
    ys = [-half + j * step for j in range(n_side)]
    # row-major: same as paper_measurements.py
    return [(x, y) for y in ys for x in xs]


def sensor_model(name: str, x: float, y: float, z: float, radius: float, length: float):
    # collision yok -> robot çarpmaz; sadece görünür (visual)
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


def write_layout_csv(path: str, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# x,y (paper sensor layout)\n")
        for (x, y) in points:
            f.write(f"{x:.6f},{y:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template world file path (e.g., empty.world)")
    ap.add_argument("--out", required=True, help="Output world file path (e.g., paper_sensors.world)")
    ap.add_argument("--layout_out", default="", help="Optional CSV output for sensor layout (x,y).")

    ap.add_argument("--n_side", type=int, default=5)
    ap.add_argument("--b", type=float, default=8.0)
    ap.add_argument("--radius", type=float, default=0.03)
    ap.add_argument("--length", type=float, default=0.5)
    ap.add_argument("--z", type=float, default=None, help="If None -> length/2 (sit on ground)")
    ap.add_argument("--name_prefix", default="paper_sensor_")
    args = ap.parse_args()

    z = args.z if args.z is not None else (args.length / 2.0)

    with open(args.template, "r", encoding="utf-8") as f:
        txt = f.read()

    idx = txt.rfind("</world>")
    if idx < 0:
        raise RuntimeError("Template has no </world> tag. Is it an SDF .world file?")

    points = make_grid(args.n_side, args.b)

    models = []
    for i, (x, y) in enumerate(points):
        models.append(sensor_model(f"{args.name_prefix}{i:02d}", x, y, z, args.radius, args.length))

    insert = "\n  <!-- Paper sensors (visual only, no collision) -->\n" + "".join(models) + "\n"
    out_txt = txt[:idx] + insert + txt[idx:]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_txt)

    if args.layout_out:
        write_layout_csv(args.layout_out, points)

    print(f"Generated world:  {args.out}  ({len(points)} sensors)")
    if args.layout_out:
        print(f"Generated layout: {args.layout_out}")


if __name__ == "__main__":
    main()
