#!/usr/bin/env python3
"""
ROS2 bag -> range-only KalmanNet dataset (.pt)

Reads:
  z_topic  : /tracking/z          (Float32MultiArray expected)
  gt_topic : /tracking/gt/odom    (nav_msgs/Odometry expected)

Exports torch tensors:
  Y_* : (N, obs_dim, T)    ranges
  X_* : (N, 4, T)          [px, py, vx, vy]  (world frame)
  init_* : (N, 4)          initial state (t=0)

Robustness:
  - velocity source: diff, twist, auto (twist rotated to world, compared to diff)
  - residual filtering: drop windows where measured ranges disagree with 2D geometry
  - drop windows containing invalid samples (robot devrildi/breakdance => pipeline kırılmaz)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch


def _require_ros_imports() -> bool:
    try:
        import rosbag2_py  # noqa
        from rclpy.serialization import deserialize_message  # noqa
        from rosidl_runtime_py.utilities import get_message  # noqa
        return True
    except Exception as e:
        print("\n[ERROR] ROS Python modülleri import edilemedi.")
        print("Çözüm: Aynı terminalde sırayla:")
        print("  source /opt/ros/jazzy/setup.bash")
        print("  source ~/KalmanNet_Indoor_Tracking/install/setup.bash")
        print("  source ~/KalmanNet_Indoor_Tracking/experiments/.venv_knet/bin/activate")
        print("\nDetay hata:", repr(e))
        return False


def load_layout_csv(path: Path) -> np.ndarray:
    """Parse sensor layout file with lines like: x,y (comments starting with #)."""
    pts = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            pts.append([x, y])
    if len(pts) == 0:
        raise ValueError(f"Layout parse edilemedi: {path}")
    return np.asarray(pts, dtype=np.float64)  # (n,2)


def list_bag_dirs(bags_dir: Path) -> list[Path]:
    bags = []
    for p in sorted(bags_dir.glob("run_*")):
        if p.is_dir() and (p / "metadata.yaml").exists():
            bags.append(p)
    return bags


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # yaw from quaternion (z-rotation), standard formula
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def read_bag(bag_dir: Path, z_topic: str, gt_topic: str):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {tt.name: tt.type for tt in topics_and_types}

    if z_topic not in type_map:
        raise RuntimeError(f"{bag_dir}: '{z_topic}' topic yok.")
    if gt_topic not in type_map:
        raise RuntimeError(f"{bag_dir}: '{gt_topic}' topic yok.")

    z_msg_type = get_message(type_map[z_topic])
    gt_msg_type = get_message(type_map[gt_topic])

    z_times = []
    z_list = []

    gt_times = []
    gt_pos = []     # (x,y,z)
    gt_quat = []    # (qx,qy,qz,qw)
    gt_twist = []   # (vx,vy) as in msg.twist.twist.linear.{x,y}

    while reader.has_next():
        topic, data, t = reader.read_next()  # t in nanoseconds (int)
        if topic == z_topic:
            msg = deserialize_message(data, z_msg_type)
            z = np.asarray(msg.data, dtype=np.float32)
            z_times.append(int(t))
            z_list.append(z)
        elif topic == gt_topic:
            msg = deserialize_message(data, gt_msg_type)
            px = float(msg.pose.pose.position.x)
            py = float(msg.pose.pose.position.y)
            pz = float(msg.pose.pose.position.z)
            qx = float(msg.pose.pose.orientation.x)
            qy = float(msg.pose.pose.orientation.y)
            qz = float(msg.pose.pose.orientation.z)
            qw = float(msg.pose.pose.orientation.w)

            vx = float(msg.twist.twist.linear.x)
            vy = float(msg.twist.twist.linear.y)

            gt_times.append(int(t))
            gt_pos.append((px, py, pz))
            gt_quat.append((qx, qy, qz, qw))
            gt_twist.append((vx, vy))

    if len(z_list) < 5:
        raise RuntimeError(f"{bag_dir}: z mesajı çok az ({len(z_list)}).")
    if len(gt_pos) < 5:
        raise RuntimeError(f"{bag_dir}: gt/odom mesajı çok az ({len(gt_pos)}).")

    z_times = np.asarray(z_times, dtype=np.int64)
    Z = np.stack(z_list, axis=0)  # [Nz, obs_dim]

    gt_times = np.asarray(gt_times, dtype=np.int64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)    # [Ng, 3]
    gt_quat = np.asarray(gt_quat, dtype=np.float64)  # [Ng, 4]
    gt_twist = np.asarray(gt_twist, dtype=np.float64)  # [Ng, 2]

    # sort safety
    z_ord = np.argsort(z_times)
    z_times = z_times[z_ord]
    Z = Z[z_ord]

    g_ord = np.argsort(gt_times)
    gt_times = gt_times[g_ord]
    gt_pos = gt_pos[g_ord]
    gt_quat = gt_quat[g_ord]
    gt_twist = gt_twist[g_ord]

    return z_times, Z, gt_times, gt_pos, gt_quat, gt_twist


def align_gt_to_z(z_times, gt_times, gt_pos, gt_quat, gt_twist):
    idx = np.searchsorted(gt_times, z_times)
    idx = np.clip(idx, 1, len(gt_times) - 1)
    left = idx - 1
    right = idx
    choose_left = (z_times - gt_times[left]) <= (gt_times[right] - z_times)
    chosen = np.where(choose_left, left, right)

    pos = gt_pos[chosen]     # [Nz, 3]
    quat = gt_quat[chosen]   # [Nz, 4]
    twist = gt_twist[chosen] # [Nz, 2]
    return pos, quat, twist


def compute_world_vel_from_twist(quat: np.ndarray, twist_xy: np.ndarray) -> np.ndarray:
    # assume twist is in robot/body frame; rotate to world using yaw
    yaws = np.array([quat_to_yaw(*q) for q in quat], dtype=np.float64)  # [N]
    c = np.cos(yaws)
    s = np.sin(yaws)
    vx_b = twist_xy[:, 0]
    vy_b = twist_xy[:, 1]
    vx_w = c * vx_b - s * vy_b
    vy_w = s * vx_b + c * vy_b
    return np.stack([vx_w, vy_w], axis=1)


def compute_diff_vel(xy: np.ndarray, z_times: np.ndarray, dt_fallback: float) -> tuple[np.ndarray, float]:
    # world-frame velocity by derivative of position
    dt = (z_times[1:] - z_times[:-1]).astype(np.float64) * 1e-9
    dt_med = float(np.nanmedian(dt)) if np.all(np.isfinite(dt)) and dt.size > 0 else float(dt_fallback)
    dt_safe = dt.copy()
    bad = (~np.isfinite(dt_safe)) | (dt_safe <= 1e-9)
    dt_safe[bad] = dt_med

    v = (xy[1:] - xy[:-1]) / dt_safe[:, None]
    v = np.vstack([v[0], v])  # repeat first
    return v.astype(np.float64), dt_med


def predict_ranges_2d(xy: np.ndarray, sensors_xy: np.ndarray, eps: float, min_range: float) -> np.ndarray:
    # xy: [N,2], sensors_xy: [M,2] => pred: [N,M]
    dx = xy[:, 0:1] - sensors_xy[None, :, 0]
    dy = xy[:, 1:2] - sensors_xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy + eps)
    d = np.maximum(d, float(min_range))
    return d


def chunk_sequences(Y: np.ndarray, X: np.ndarray, valid: np.ndarray, seq_len: int, stride: int | None):
    """
    Y: [N, obs_dim]
    X: [N, 4]
    valid: [N] bool
    Returns:
      Yc: [n_seq, obs_dim, T]
      Xc: [n_seq, 4, T]
      init: [n_seq, 4]
    """
    N = Y.shape[0]
    if stride is None:
        stride = seq_len

    Y_seqs = []
    X_seqs = []
    init = []

    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        if not bool(np.all(valid[start:end])):
            continue
        y_win = Y[start:end]  # [T, obs_dim]
        x_win = X[start:end]  # [T, 4]
        Y_seqs.append(y_win.T)  # [obs_dim, T]
        X_seqs.append(x_win.T)  # [4, T]
        init.append(x_win[0])

    if len(Y_seqs) == 0:
        return None, None, None

    Yc = np.stack(Y_seqs, axis=0).astype(np.float32)
    Xc = np.stack(X_seqs, axis=0).astype(np.float32)
    init = np.stack(init, axis=0).astype(np.float32)
    return Yc, Xc, init


def split_indices(n, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags_dir", default="datasets/bags", help="datasets/bags dizini")
    ap.add_argument("--out", default="datasets/pt/range_v2.pt", help="çıktı .pt yolu")

    ap.add_argument("--layout_file", default="config/paper_sensors_5x5_b20.csv", help="Sensor layout CSV (x,y)")

    ap.add_argument("--z_topic", default="/tracking/z")
    ap.add_argument("--gt_topic", default="/tracking/gt/odom")

    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--stride", type=int, default=0, help="0 => non-overlap (stride=seq_len), else sliding stride")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--vel_source", choices=["auto", "diff", "twist"], default="auto",
                    help="vx/vy source: diff=dx/dt, twist=odom.twist rotated by yaw, auto=choose")

    # range geometry + filtering
    ap.add_argument("--min_range", type=float, default=1e-3)
    ap.add_argument("--eps_range", type=float, default=0.0, help="if 0, uses min_range^2")
    ap.add_argument("--max_residual_rms", type=float, default=2.0,
                    help="Drop windows if RMS(range_meas - range_pred) exceeds this [m].")
    ap.add_argument("--max_speed", type=float, default=6.0, help="Drop windows if speed exceeds this [m/s].")

    args = ap.parse_args()

    if not _require_ros_imports():
        raise SystemExit(2)

    bags_dir = Path(args.bags_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout_file = Path(args.layout_file)
    sensors_xy = load_layout_csv(layout_file)  # (obs_dim,2)

    eps = float(args.eps_range) if float(args.eps_range) > 0 else float(args.min_range) ** 2

    bag_dirs = list_bag_dirs(bags_dir)
    if not bag_dirs:
        raise SystemExit(f"Bag bulunamadı: {bags_dir}")

    Y_all = []
    X_all = []
    init_all = []
    dt_meds = []
    kept_total = 0
    drop_total = 0

    for b in bag_dirs:
        print(f"[INFO] Reading bag: {b}")
        z_times, Z, gt_times, gt_pos, gt_quat, gt_twist = read_bag(b, args.z_topic, args.gt_topic)

        obs_dim = Z.shape[1]
        if sensors_xy.shape[0] != obs_dim:
            raise RuntimeError(
                f"{b}: obs_dim={obs_dim} ama layout sensor sayısı={sensors_xy.shape[0]}. "
                f"Yanlış layout_file kullanıyorsun."
            )

        # align GT to z times
        pos, quat, twist = align_gt_to_z(z_times, gt_times, gt_pos, gt_quat, gt_twist)
        xy = pos[:, :2].copy()  # [N,2]
        z_meas = Z.astype(np.float64)

        # clamp ranges
        z_meas = np.maximum(z_meas, float(args.min_range))

        # velocities
        v_diff, dt_med = compute_diff_vel(xy, z_times, dt_fallback=0.1)
        v_tw_world = compute_world_vel_from_twist(quat, twist)

        if args.vel_source == "diff":
            v = v_diff
            vel_used = "diff"
        elif args.vel_source == "twist":
            v = v_tw_world
            vel_used = "twist(world)"
        else:
            # auto: pick twist if it matches diff reasonably
            med_err = float(np.nanmedian(np.linalg.norm(v_tw_world - v_diff, axis=1)))
            if np.isfinite(med_err) and med_err < 0.5:
                v = v_tw_world
                vel_used = "twist(world)"
            else:
                v = v_diff
                vel_used = "diff"
            print(f"       vel_source=auto -> {vel_used} (median |v_tw - v_diff| = {med_err:.3f} m/s)")

        dt_meds.append(dt_med)

        # build state
        X = np.concatenate([xy, v], axis=1)  # [N,4]

        # validity mask per-sample (later window-AND)
        valid = np.ones((X.shape[0],), dtype=bool)

        # speed filter
        speed = np.linalg.norm(X[:, 2:4], axis=1)
        valid &= np.isfinite(speed)
        valid &= speed <= float(args.max_speed)

        # residual filter vs 2D geometry
        pred = predict_ranges_2d(xy, sensors_xy, eps=eps, min_range=float(args.min_range))  # [N,obs_dim]
        resid = z_meas - pred
        resid_rms = np.sqrt(np.mean(resid * resid, axis=1))  # [N]
        valid &= np.isfinite(resid_rms)
        valid &= resid_rms <= float(args.max_residual_rms)

        # chunk into sequences, dropping windows containing invalid samples
        stride = args.seq_len if int(args.stride) <= 0 else int(args.stride)
        Yc, Xc, init = chunk_sequences(z_meas.astype(np.float32), X.astype(np.float32), valid, args.seq_len, stride)

        if Yc is None:
            print("       [WARN] No valid sequences from this bag after filtering.")
            continue

        kept = int(Yc.shape[0])
        # estimate how many windows existed without filtering
        n_total = 1 + (z_meas.shape[0] - args.seq_len) // stride if z_meas.shape[0] >= args.seq_len else 0
        dropped = max(0, n_total - kept)

        kept_total += kept
        drop_total += dropped

        print(f"       z_samples={len(z_meas)}  windows_total={n_total}  kept_seq={kept}  dropped_seq={dropped}  "
              f"obs_dim={Yc.shape[1]}  T={Yc.shape[2]}")

        Y_all.append(Yc)
        X_all.append(Xc)
        init_all.append(init)

    if not Y_all:
        raise SystemExit("[FATAL] Hiç sequence üretilemedi. max_residual/max_speed filtrelerini gevşet.")

    Y_all = np.concatenate(Y_all, axis=0)
    X_all = np.concatenate(X_all, axis=0)
    init_all = np.concatenate(init_all, axis=0)

    n_seq = Y_all.shape[0]
    train_idx, val_idx, test_idx = split_indices(n_seq, args.val_ratio, args.test_ratio, args.seed)

    def take(a, idx):
        return a[idx]

    dt_global = float(np.median(dt_meds)) if len(dt_meds) else 0.1

    data = {
        "Y_train": torch.tensor(take(Y_all, train_idx), dtype=torch.float32),
        "X_train": torch.tensor(take(X_all, train_idx), dtype=torch.float32),
        "init_train": torch.tensor(take(init_all, train_idx), dtype=torch.float32),

        "Y_val": torch.tensor(take(Y_all, val_idx), dtype=torch.float32),
        "X_val": torch.tensor(take(X_all, val_idx), dtype=torch.float32),
        "init_val": torch.tensor(take(init_all, val_idx), dtype=torch.float32),

        "Y_test": torch.tensor(take(Y_all, test_idx), dtype=torch.float32),
        "X_test": torch.tensor(take(X_all, test_idx), dtype=torch.float32),
        "init_test": torch.tensor(take(init_all, test_idx), dtype=torch.float32),

        "meta": {
            "obs_dim": int(Y_all.shape[1]),
            "state_dim": int(X_all.shape[1]),
            "seq_len": int(args.seq_len),
            "dt_median": float(dt_global),
            "z_topic": args.z_topic,
            "gt_topic": args.gt_topic,
            "layout_file": str(layout_file),
            "vel_source": args.vel_source,
            "filters": {
                "min_range": float(args.min_range),
                "eps_range": float(eps),
                "max_residual_rms": float(args.max_residual_rms),
                "max_speed": float(args.max_speed),
                "stride": int(stride),
            },
            "bags": [p.name for p in bag_dirs],
            "split": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "kept_total": int(kept_total),
            "dropped_total": int(drop_total),
        }
    }

    torch.save(data, str(out_path))
    print(f"\n[DONE] Saved dataset: {out_path}")
    print("      Shapes:")
    print("        Y_train:", tuple(data["Y_train"].shape))
    print("        X_train:", tuple(data["X_train"].shape))
    print("        Y_val  :", tuple(data["Y_val"].shape))
    print("        X_val  :", tuple(data["X_val"].shape))
    print("        Y_test :", tuple(data["Y_test"].shape))
    print("        X_test :", tuple(data["X_test"].shape))
    print(f"      dt_median={dt_global:.6f}s kept_seq={kept_total} dropped_seq={drop_total}")


if __name__ == "__main__":
    main()
