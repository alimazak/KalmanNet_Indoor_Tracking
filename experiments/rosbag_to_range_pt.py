#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import torch

def _require_ros_imports():
    try:
        import rosbag2_py  # noqa
        from rclpy.serialization import deserialize_message  # noqa
        from rosidl_runtime_py.utilities import get_message  # noqa
        return True
    except Exception as e:
        print("\n[ERROR] ROS Python modülleri import edilemedi.")
        print("Çözüm: Aynı terminalde önce şunu yap:")
        print("  source /opt/ros/jazzy/setup.bash")
        print("  source ~/KalmanNet_Indoor_Tracking/install/setup.bash   # opsiyonel ama güvenli")
        print("  source ~/KalmanNet_Indoor_Tracking/experiments/.venv_knet/bin/activate")
        print("\nDetay hata:", repr(e))
        return False

def list_bag_dirs(bags_dir: Path):
    bags = []
    for p in sorted(bags_dir.glob("run_*")):
        if p.is_dir() and (p / "metadata.yaml").exists():
            bags.append(p)
    return bags

def read_bag(bag_dir: Path, z_topic: str, gt_topic: str):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {tt.name: tt.type for tt in topics_and_types}

    if z_topic not in type_map:
        raise RuntimeError(f"{bag_dir}: '{z_topic}' topic yok. Bag içeriğini kontrol et.")
    if gt_topic not in type_map:
        raise RuntimeError(f"{bag_dir}: '{gt_topic}' topic yok. Bag içeriğini kontrol et.")

    z_msg_type = get_message(type_map[z_topic])
    gt_msg_type = get_message(type_map[gt_topic])

    z_times = []
    z_list = []

    gt_times = []
    gt_xy = []

    while reader.has_next():
        topic, data, t = reader.read_next()  # t: int nanoseconds
        if topic == z_topic:
            msg = deserialize_message(data, z_msg_type)
            z = np.asarray(msg.data, dtype=np.float32)
            z_times.append(int(t))
            z_list.append(z)
        elif topic == gt_topic:
            msg = deserialize_message(data, gt_msg_type)
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            gt_times.append(int(t))
            gt_xy.append((x, y))

    if len(z_list) < 5:
        raise RuntimeError(f"{bag_dir}: z mesajı çok az ({len(z_list)}). Kayıt kısa olabilir.")
    if len(gt_xy) < 5:
        raise RuntimeError(f"{bag_dir}: gt/odom mesajı çok az ({len(gt_xy)}).")

    z_times = np.asarray(z_times, dtype=np.int64)
    Z = np.stack(z_list, axis=0)  # [N, n]
    gt_times = np.asarray(gt_times, dtype=np.int64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)  # [M, 2]

    # guarantee sorted (reader genelde sıralı ama yine de)
    z_ord = np.argsort(z_times)
    z_times = z_times[z_ord]
    Z = Z[z_ord]

    gt_ord = np.argsort(gt_times)
    gt_times = gt_times[gt_ord]
    gt_xy = gt_xy[gt_ord]

    return z_times, Z, gt_times, gt_xy

def align_and_build_state(z_times, Z, gt_times, gt_xy):
    # z timestamp’lerine en yakın gt sample’ı al
    idx = np.searchsorted(gt_times, z_times)
    idx = np.clip(idx, 1, len(gt_times) - 1)
    left = idx - 1
    right = idx
    choose_left = (z_times - gt_times[left]) <= (gt_times[right] - z_times)
    chosen = np.where(choose_left, left, right)

    xy = gt_xy[chosen]  # [N,2] world xy

    # vx, vy: pose farkından türev
    dt = (z_times[1:] - z_times[:-1]).astype(np.float64) * 1e-9
    dt[dt <= 0] = np.nan  # güvenlik
    vxy = (xy[1:] - xy[:-1]) / dt[:, None]
    vxy = np.vstack([vxy[0], vxy])  # ilk sample için kopya

    X = np.concatenate([xy, vxy], axis=1).astype(np.float32)  # [N,4]
    return X

def chunk_sequences(Y, X, seq_len: int):
    # non-overlapping chunk
    N = Y.shape[0]
    n_seq = N // seq_len
    if n_seq <= 0:
        raise RuntimeError(f"Sequence üretilemedi: N={N}, seq_len={seq_len}")

    N_used = n_seq * seq_len
    Yc = Y[:N_used].reshape(n_seq, seq_len, Y.shape[1]).transpose(0, 2, 1)  # [n_seq, n, T]
    Xc = X[:N_used].reshape(n_seq, seq_len, X.shape[1]).transpose(0, 2, 1)  # [n_seq, 4, T]
    init = Xc[:, :, 0].copy()  # [n_seq, 4]
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
    ap.add_argument("--out", default="datasets/pt/range_v1.pt", help="çıktı .pt yolu")
    ap.add_argument("--z_topic", default="/tracking/z")
    ap.add_argument("--gt_topic", default="/tracking/gt/odom")
    ap.add_argument("--seq_len", type=int, default=100)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not _require_ros_imports():
        raise SystemExit(2)

    bags_dir = Path(args.bags_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bag_dirs = list_bag_dirs(bags_dir)
    if not bag_dirs:
        raise SystemExit(f"Bag bulunamadı: {bags_dir}")

    Y_all = []
    X_all = []
    init_all = []
    obs_dim = None

    for b in bag_dirs:
        print(f"[INFO] Reading bag: {b}")
        z_times, Z, gt_times, gt_xy = read_bag(b, args.z_topic, args.gt_topic)

        if obs_dim is None:
            obs_dim = Z.shape[1]
        elif Z.shape[1] != obs_dim:
            raise RuntimeError(f"Sensor sayısı tutarsız: {b} obs_dim={Z.shape[1]} beklenen={obs_dim}")

        X = align_and_build_state(z_times, Z, gt_times, gt_xy)
        Yc, Xc, init = chunk_sequences(Z.astype(np.float32), X, args.seq_len)

        print(f"       z_samples={len(Z)}  -> seq={Yc.shape[0]}  obs_dim={Yc.shape[1]}  T={Yc.shape[2]}")
        Y_all.append(Yc)
        X_all.append(Xc)
        init_all.append(init)

    Y_all = np.concatenate(Y_all, axis=0)
    X_all = np.concatenate(X_all, axis=0)
    init_all = np.concatenate(init_all, axis=0)

    n_seq = Y_all.shape[0]
    train_idx, val_idx, test_idx = split_indices(n_seq, args.val_ratio, args.test_ratio, args.seed)

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
            "obs_dim": int(Y_all.shape[1]),
            "state_dim": int(X_all.shape[1]),
            "seq_len": int(args.seq_len),
            "z_topic": args.z_topic,
            "gt_topic": args.gt_topic,
            "bags": [p.name for p in bag_dirs],
            "split": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
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

if __name__ == "__main__":
    main()
