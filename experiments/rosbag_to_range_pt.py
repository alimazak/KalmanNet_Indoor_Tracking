#!/usr/bin/env python3
"""
ROS2 bag -> range-only KalmanNet dataset (.pt)

Goal:
  Build supervised sequences for training KalmanNet (range-only tracking).

Inputs (from bag):
  - GT odometry:  /tracking/gt/odom   (nav_msgs/msg/Odometry)
  - Optional z:   /tracking/z         (std_msgs/msg/Float32MultiArray)

If /tracking/z is missing, we can SYNTHESIZE z from GT position + sensor layout:
  z_i(t) = || [px(t),py(t)] - sensor_i || + noise

Outputs (.pt):
  Y_* : (Nseq, obs_dim, T)   range measurements
  X_* : (Nseq, 4, T)         [px, py, vx, vy]  (world frame)
  init_* : (Nseq, 4)         initial state at t=0
  meta : dict
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

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
        print("Aynı terminalde sırayla şunları yap:")
        print("  source /opt/ros/jazzy/setup.bash")
        print("  source ~/KalmanNet_Indoor_Tracking/install/setup.bash")
        print("  source ~/KalmanNet_Indoor_Tracking/experiments/.venv_knet/bin/activate")
        print("\nDetay hata:", repr(e))
        return False


def load_layout_csv(path: Path) -> np.ndarray:
    pts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
    if not pts:
        raise ValueError(f"Layout parse edilemedi: {path}")
    return np.asarray(pts, dtype=np.float64)  # (obs_dim,2)


def list_bag_dirs(bags_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(bags_dir.glob("run_*")):
        if p.is_dir() and (p / "metadata.yaml").exists():
            out.append(p)
    return out


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _nearest_indices(src_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """
    For each target_time, pick nearest index in src_times (both int64 ns).
    src_times must be sorted.
    """
    idx = np.searchsorted(src_times, target_times)
    idx = np.clip(idx, 1, len(src_times) - 1)
    left = idx - 1
    right = idx
    choose_left = (target_times - src_times[left]) <= (src_times[right] - target_times)
    return np.where(choose_left, left, right)


def _uniform_times(gt_times: np.ndarray, rate_hz: float) -> np.ndarray:
    if gt_times.size < 2:
        return gt_times.copy()
    step_ns = int(round(1e9 / float(rate_hz)))
    t0 = int(gt_times[0])
    t1 = int(gt_times[-1])
    if t1 <= t0:
        return gt_times.copy()
    return np.arange(t0, t1 + 1, step_ns, dtype=np.int64)


def compute_world_vel_from_twist(quat: np.ndarray, twist_xy: np.ndarray) -> np.ndarray:
    # twist assumed body frame; rotate to world using yaw
    yaws = np.array([quat_to_yaw(*q) for q in quat], dtype=np.float64)
    c = np.cos(yaws)
    s = np.sin(yaws)
    vx_b = twist_xy[:, 0]
    vy_b = twist_xy[:, 1]
    vx_w = c * vx_b - s * vy_b
    vy_w = s * vx_b + c * vy_b
    return np.stack([vx_w, vy_w], axis=1)


def compute_diff_vel(xy: np.ndarray, times_ns: np.ndarray, dt_fallback: float = 0.1) -> Tuple[np.ndarray, float]:
    dt = (times_ns[1:] - times_ns[:-1]).astype(np.float64) * 1e-9
    if dt.size == 0:
        return np.zeros_like(xy), float(dt_fallback)
    dt_med = float(np.nanmedian(dt[np.isfinite(dt) & (dt > 1e-9)])) if np.any(np.isfinite(dt)) else float(dt_fallback)
    dt_safe = dt.copy()
    bad = (~np.isfinite(dt_safe)) | (dt_safe <= 1e-9)
    dt_safe[bad] = dt_med
    v = (xy[1:] - xy[:-1]) / dt_safe[:, None]
    v = np.vstack([v[0], v])  # repeat first
    return v.astype(np.float64), dt_med


def predict_ranges_2d(xy: np.ndarray, sensors_xy: np.ndarray, eps: float, min_range: float) -> np.ndarray:
    dx = xy[:, 0:1] - sensors_xy[None, :, 0]
    dy = xy[:, 1:2] - sensors_xy[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy + eps)
    d = np.maximum(d, float(min_range))
    return d


def chunk_sequences(
    Y: np.ndarray,
    X: np.ndarray,
    valid: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Y: [N, obs_dim]
    X: [N, 4]
    valid: [N]
    Returns Yc:[nseq,obs_dim,T], Xc:[nseq,4,T], init:[nseq,4], windows_total
    """
    N = Y.shape[0]
    if N < seq_len:
        return None, None, None, 0

    windows_total = 1 + (N - seq_len) // stride

    Y_seqs = []
    X_seqs = []
    init = []

    for start in range(0, N - seq_len + 1, stride):
        end = start + seq_len
        if not bool(np.all(valid[start:end])):
            continue
        y_win = Y[start:end]  # [T,obs]
        x_win = X[start:end]  # [T,4]
        Y_seqs.append(y_win.T)  # [obs,T]
        X_seqs.append(x_win.T)  # [4,T]
        init.append(x_win[0])

    if not Y_seqs:
        return None, None, None, windows_total

    Yc = np.stack(Y_seqs, axis=0).astype(np.float32)
    Xc = np.stack(X_seqs, axis=0).astype(np.float32)
    init = np.stack(init, axis=0).astype(np.float32)
    return Yc, Xc, init, windows_total


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


def read_bag(
    bag_dir: Path,
    gt_topic: str,
    z_topic: Optional[str],
):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {tt.name: tt.type for tt in topics_and_types}

    if gt_topic not in type_map:
        raise RuntimeError(f"{bag_dir}: '{gt_topic}' topic yok.")

    has_z = (z_topic is not None) and (z_topic in type_map)

    gt_msg_type = get_message(type_map[gt_topic])
    z_msg_type = get_message(type_map[z_topic]) if has_z else None

    gt_times = []
    gt_pos = []
    gt_quat = []
    gt_twist = []

    z_times = []
    z_list = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == gt_topic:
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

        elif has_z and topic == z_topic:
            msg = deserialize_message(data, z_msg_type)
            z = np.asarray(msg.data, dtype=np.float64)
            z_times.append(int(t))
            z_list.append(z)

    if len(gt_times) < 5:
        raise RuntimeError(f"{bag_dir}: gt/odom mesajı çok az ({len(gt_times)}).")

    gt_times = np.asarray(gt_times, dtype=np.int64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    gt_quat = np.asarray(gt_quat, dtype=np.float64)
    gt_twist = np.asarray(gt_twist, dtype=np.float64)

    g_ord = np.argsort(gt_times)
    gt_times = gt_times[g_ord]
    gt_pos = gt_pos[g_ord]
    gt_quat = gt_quat[g_ord]
    gt_twist = gt_twist[g_ord]

    if has_z and len(z_list) >= 5:
        z_times = np.asarray(z_times, dtype=np.int64)
        Z = np.stack(z_list, axis=0).astype(np.float64)
        z_ord = np.argsort(z_times)
        z_times = z_times[z_ord]
        Z = Z[z_ord]
        return True, gt_times, gt_pos, gt_quat, gt_twist, z_times, Z

    return False, gt_times, gt_pos, gt_quat, gt_twist, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags_dir", default="datasets/bags")
    ap.add_argument("--out", default="datasets/pt/range_v3.pt")
    ap.add_argument("--layout_file", default="config/paper_sensors_5x5_b20.csv")

    ap.add_argument("--gt_topic", default="/tracking/gt/odom")
    ap.add_argument("--z_topic", default="/tracking/z", help="Set empty '' to ignore bag z completely.")

    ap.add_argument("--z_mode", choices=["auto", "bag", "synth"], default="auto",
                    help="auto: use bag z if exists else synth; bag: require z; synth: always synth from GT")

    ap.add_argument("--rate_hz", type=float, default=10.0,
                    help="Only used when synthesizing z (uniform sampling from GT).")

    ap.add_argument("--sigma", type=float, default=0.10,
                    help="Noise std for synthetic z generation [m]. Ignored if using bag z.")

    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--stride", type=int, default=0, help="0 => non-overlap, else sliding stride")

    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--vel_source", choices=["auto", "diff", "twist"], default="auto")

    ap.add_argument("--min_range", type=float, default=1e-3)
    ap.add_argument("--eps_range", type=float, default=0.0, help="0 => min_range^2")

    ap.add_argument("--max_residual_rms", type=float, default=2.0,
                    help="Drop samples where RMS(z - h(x)) exceeds this [m]. Set huge to disable.")
    ap.add_argument("--max_speed", type=float, default=20.0,
                    help="Drop samples where speed exceeds this [m/s]. Set huge to disable.")

    args = ap.parse_args()

    if not _require_ros_imports():
        raise SystemExit(2)

    bags_dir = Path(args.bags_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sensors_xy = load_layout_csv(Path(args.layout_file))
    obs_dim_layout = int(sensors_xy.shape[0])

    eps = float(args.eps_range) if float(args.eps_range) > 0 else float(args.min_range) ** 2

    z_topic = args.z_topic.strip()
    z_topic = z_topic if z_topic else None

    bag_dirs = list_bag_dirs(bags_dir)
    if not bag_dirs:
        raise SystemExit(f"Bag bulunamadı: {bags_dir}")

    Y_all = []
    X_all = []
    init_all = []
    dt_meds = []
    kept_total = 0
    dropped_total = 0

    for bi, b in enumerate(bag_dirs):
        print(f"[INFO] Reading bag: {b}")

        has_z, gt_times, gt_pos, gt_quat, gt_twist, z_times, Z = read_bag(
            b, gt_topic=args.gt_topic, z_topic=z_topic
        )

        # Decide measurement source
        use_bag_z = False
        if args.z_mode == "bag":
            if not has_z:
                raise RuntimeError(f"{b}: z_mode=bag but '{z_topic}' yok veya mesaj yok.")
            use_bag_z = True
        elif args.z_mode == "synth":
            use_bag_z = False
        else:  # auto
            use_bag_z = bool(has_z)

        rng = np.random.default_rng(int(args.seed) + 1000 * bi)

        if use_bag_z:
            # measurement timestamps = z times
            sample_times = z_times
            Y = Z
            if Y.ndim != 2:
                raise RuntimeError(f"{b}: z shape garip: {Y.shape}")
            if Y.shape[1] != obs_dim_layout:
                raise RuntimeError(
                    f"{b}: obs_dim mismatch. z has {Y.shape[1]} but layout has {obs_dim_layout} sensors."
                )
            meas_src = "bag"
        else:
            # uniform sampling from GT at rate_hz, then synth ranges
            sample_times = _uniform_times(gt_times, args.rate_hz)
            meas_src = "synth"
            # align GT first to sample times, then compute ranges
            idx = _nearest_indices(gt_times, sample_times)
            xy = gt_pos[idx, :2]
            Y = predict_ranges_2d(xy, sensors_xy, eps=eps, min_range=float(args.min_range))
            if float(args.sigma) > 0:
                Y = Y + rng.normal(0.0, float(args.sigma), size=Y.shape)
            Y = np.maximum(Y, float(args.min_range))

        # Align GT to sample times
        idx = _nearest_indices(gt_times, sample_times)
        pos = gt_pos[idx]
        quat = gt_quat[idx]
        twist = gt_twist[idx]

        xy = pos[:, :2].copy()

        # Velocities
        v_diff, dt_med = compute_diff_vel(xy, sample_times, dt_fallback=0.1)
        v_tw_world = compute_world_vel_from_twist(quat, twist)

        if args.vel_source == "diff":
            v = v_diff
            vel_used = "diff"
            med_err = float(np.nanmedian(np.linalg.norm(v_tw_world - v_diff, axis=1)))
        elif args.vel_source == "twist":
            v = v_tw_world
            vel_used = "twist(world)"
            med_err = float(np.nanmedian(np.linalg.norm(v_tw_world - v_diff, axis=1)))
        else:
            med_err = float(np.nanmedian(np.linalg.norm(v_tw_world - v_diff, axis=1)))
            # If twist matches diff reasonably, keep twist; else use diff
            if np.isfinite(med_err) and med_err < 0.5:
                v = v_tw_world
                vel_used = "twist(world)"
            else:
                v = v_diff
                vel_used = "diff"

        dt_meds.append(dt_med)

        X = np.concatenate([xy, v], axis=1)  # [N,4]
        Y = np.asarray(Y, dtype=np.float64)

        # Per-sample validity mask
        valid = np.ones((X.shape[0],), dtype=bool)

        # Finite checks
        valid &= np.isfinite(X).all(axis=1)
        valid &= np.isfinite(Y).all(axis=1)

        # Speed filter
        speed = np.linalg.norm(X[:, 2:4], axis=1)
        valid &= np.isfinite(speed)
        valid &= speed <= float(args.max_speed)

        # Residual filter vs geometry
        pred = predict_ranges_2d(xy, sensors_xy, eps=eps, min_range=float(args.min_range))
        resid = Y - pred
        resid_rms = np.sqrt(np.mean(resid * resid, axis=1))
        valid &= np.isfinite(resid_rms)
        valid &= resid_rms <= float(args.max_residual_rms)

        stride = int(args.seq_len) if int(args.stride) <= 0 else int(args.stride)
        Yc, Xc, init, windows_total = chunk_sequences(
            Y.astype(np.float32), X.astype(np.float32), valid, int(args.seq_len), stride
        )

        if Yc is None:
            print(f"       [WARN] No valid sequences from this bag after filtering. (meas={meas_src})")
            continue

        kept = int(Yc.shape[0])
        dropped = max(0, int(windows_total) - kept)

        kept_total += kept
        dropped_total += dropped

        print(f"       meas={meas_src} vel={vel_used} (median |v_tw - v_diff|={med_err:.3f} m/s)")
        print(f"       samples={len(sample_times)} windows_total={windows_total} kept_seq={kept} dropped_seq={dropped} "
              f"obs_dim={Yc.shape[1]} T={Yc.shape[2]}")

        Y_all.append(Yc)
        X_all.append(Xc)
        init_all.append(init)

    if not Y_all:
        raise SystemExit("[FATAL] Hiç sequence üretilemedi. Filtreleri gevşet veya rate_hz/seq_len ayarlarını kontrol et.")

    Y_all = np.concatenate(Y_all, axis=0)
    X_all = np.concatenate(X_all, axis=0)
    init_all = np.concatenate(init_all, axis=0)

    n_seq = int(Y_all.shape[0])
    train_idx, val_idx, test_idx = split_indices(n_seq, float(args.val_ratio), float(args.test_ratio), int(args.seed))

    dt_global = float(np.median(dt_meds)) if dt_meds else 0.1

    def take(a, idx):
        return a[idx]

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
            "layout_file": str(Path(args.layout_file)),
            "obs_dim": int(Y_all.shape[1]),
            "state_dim": int(X_all.shape[1]),
            "seq_len": int(args.seq_len),
            "dt_median": float(dt_global),
            "gt_topic": args.gt_topic,
            "z_topic": (z_topic or ""),
            "z_mode": args.z_mode,
            "rate_hz": float(args.rate_hz),
            "sigma_synth": float(args.sigma),
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
            "dropped_total": int(dropped_total),
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
    print(f"      dt_median={dt_global:.6f}s kept_seq={kept_total} dropped_seq={dropped_total}")


if __name__ == "__main__":
    main()
