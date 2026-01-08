#!/usr/bin/env python3
"""
Train KalmanNet (KalmanNet_TSP) on range-only tracking dataset exported as .pt

Expected dataset keys (your file already has these):
  - Y_train, X_train, init_train
  - Y_val,   X_val,   init_val
  - Y_test,  X_test,  init_test
Shapes:
  Y_* : (N, obs_dim, T)
  X_* : (N, state_dim, T)   state_dim should be 4: [px, py, vx, vy]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def add_kalman_tsp_to_path(repo_root: Path) -> Path:
    knet_root = repo_root / "third_party" / "KalmanNet_TSP"
    if not knet_root.is_dir():
        raise FileNotFoundError(f"KalmanNet_TSP not found at: {knet_root}")
    sys.path.insert(0, str(knet_root))
    return knet_root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_layout_csv(path: Path) -> torch.Tensor:
    """Parse sensor layout file with lines like: x,y (comments starting with #)."""
    pts = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
            else:
                parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            pts.append([x, y])
    if len(pts) == 0:
        raise ValueError(
            f"Could not parse any (x,y) rows from {path}. "
            "Expected lines like: 0.0,0.2"
        )
    return torch.tensor(pts, dtype=torch.float32)


def ensure_init_shape(init: torch.Tensor | None, m: int) -> torch.Tensor | None:
    """Return init as (N, m, 1) float32."""
    if init is None:
        return None
    t = init
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    t = t.float()

    if t.ndim == 1:
        if t.numel() != m:
            raise ValueError(f"init vector expected length {m}, got {t.numel()}")
        t = t.view(1, m, 1)
    elif t.ndim == 2:
        if t.shape[1] != m:
            raise ValueError(f"init expected (N,{m}), got {tuple(t.shape)}")
        t = t.unsqueeze(-1)  # (N,m,1)
    elif t.ndim == 3:
        if t.shape[1] != m:
            raise ValueError(f"init expected (N,{m},1), got {tuple(t.shape)}")
        if t.shape[2] != 1:
            t = t[:, :, :1]
    else:
        raise ValueError(f"init must be 1D/2D/3D, got ndim={t.ndim}")
    return t


def build_cv_F(dt: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, dt, 0.0],
         [0.0, 1.0, 0.0, dt],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def build_cv_Q(dt: float, tau: float, device=None, dtype=torch.float32) -> torch.Tensor:
    # Matches repo ekf_range_core ConstantVelocityModel Q (CWNA discretization)
    d = float(dt)
    return float(tau) * torch.tensor(
        [[d**3 / 3.0, 0.0,        d**2 / 2.0, 0.0],
         [0.0,        d**3 / 3.0, 0.0,        d**2 / 2.0],
         [d**2 / 2.0, 0.0,        d,          0.0],
         [0.0,        d**2 / 2.0, 0.0,        d]],
        dtype=dtype,
        device=device,
    )


def build_R(sigma: float, n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    return (float(sigma) ** 2) * torch.eye(n, dtype=dtype, device=device)


def make_f_h(dt: float, sensors_xy: torch.Tensor):
    """
    Returns torch-based f/h functions compatible with KalmanNet_TSP Extended_sysmdl:
      - x: (B,4,1)
      - f(x) -> (B,4,1)
      - h(x) -> (B,n,1)
    Also supports jacobian=True returning (y, J) for EKF compatibility.
    """
    sensors_xy = sensors_xy.float()
    sx = sensors_xy[:, 0].clone()
    sy = sensors_xy[:, 1].clone()
    n = int(sensors_xy.shape[0])

    def f(x: torch.Tensor, jacobian: bool = False):
        B = x.shape[0]
        F = build_cv_F(dt, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        x_next = torch.bmm(F, x)
        if jacobian:
            return x_next, F
        return x_next

    def h(x: torch.Tensor, jacobian: bool = False):
        px = x[:, 0, 0].unsqueeze(1)  # (B,1)
        py = x[:, 1, 0].unsqueeze(1)

        dx = px - sx.to(x.device).unsqueeze(0)  # (B,n)
        dy = py - sy.to(x.device).unsqueeze(0)

        d = torch.sqrt(dx * dx + dy * dy + 1e-9)  # (B,n)
        y = d.unsqueeze(2)  # (B,n,1)

        if jacobian:
            H = torch.zeros((x.shape[0], n, 4), dtype=x.dtype, device=x.device)
            inv_d = 1.0 / d.clamp_min(1e-9)
            H[:, :, 0] = dx * inv_d
            H[:, :, 1] = dy * inv_d
            return y, H

        return y

    return f, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset .pt (e.g. datasets/pt/range_v1.pt)")
    parser.add_argument("--layout_file", default="config/paper_sensors_5x5_b20.csv", help="Sensor layout CSV (x,y)")
    parser.add_argument("--out_dir", default="models/knet_range_v1", help="Where to save training outputs")
    parser.add_argument("--model_name", default="", help="Optional run name. If empty, auto timestamp is used.")
    parser.add_argument("--seed", type=int, default=0)

    # Match launch defaults (paper_tracking.launch.py) by default
    parser.add_argument("--dt", type=float, default=0.1, help="delta (seconds) per step")
    parser.add_argument("--tau", type=float, default=1.0, help="process noise scale")
    parser.add_argument("--sigma", type=float, default=0.10, help="range measurement std")

    parser.add_argument("--init_pos_std", type=float, default=5.0)
    parser.add_argument("--init_vel_std", type=float, default=2.0)

    # Training hyperparams (keep small; you have 45 sequences)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--n_batch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)

    # KalmanNet size multipliers (same style as KalmanNet_TSP examples)
    parser.add_argument("--in_mult_KNet", type=int, default=40)
    parser.add_argument("--out_mult_KNet", type=int, default=5)

    parser.add_argument("--use_cuda", action="store_true", help="Try GPU (if available)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    add_kalman_tsp_to_path(repo_root)

    # Import after sys.path injection
    from Simulations.Extended_sysmdl import SystemModel
    from Pipelines.Pipeline_EKF import Pipeline_EKF
    from KNet.KalmanNet_nn import KalmanNetNN

    if args.use_cuda and not torch.cuda.is_available():
        print("[WARN] --use_cuda given but CUDA not available. Falling back to CPU.")
        args.use_cuda = False

    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    D = torch.load(str(data_path), map_location="cpu")

    Y_train = D["Y_train"].float()
    X_train = D["X_train"].float()
    Y_val = D["Y_val"].float()
    X_val = D["X_val"].float()
    Y_test = D["Y_test"].float()
    X_test = D["X_test"].float()

    N_train, obs_dim, T = Y_train.shape
    state_dim = X_train.shape[1]

    if state_dim != 4:
        raise ValueError(f"Expected state_dim=4 ([px,py,vx,vy]), got {state_dim}")

    layout_file = Path(args.layout_file)
    if not layout_file.is_file():
        raise FileNotFoundError(f"layout_file not found: {layout_file}")

    sensors_xy = load_layout_csv(layout_file)
    if sensors_xy.shape[0] != obs_dim:
        raise ValueError(
            f"layout_file sensor count ({sensors_xy.shape[0]}) != obs_dim ({obs_dim}). "
            "You must use the SAME layout file as in the sim/launch."
        )

    # Build SystemModel (f/h + Q/R)
    Q = build_cv_Q(args.dt, args.tau)
    R = build_R(args.sigma, obs_dim)

    prior_Sigma = torch.diag(
        torch.tensor(
            [args.init_pos_std**2, args.init_pos_std**2, args.init_vel_std**2, args.init_vel_std**2],
            dtype=torch.float32,
        )
    )

    f, h = make_f_h(args.dt, sensors_xy)

    sys_model = SystemModel(
        f, Q, h, R,
        T, T,  # T, T_test
        state_dim, obs_dim,
        prior_Q=Q, prior_Sigma=prior_Sigma, prior_S=R
    )

    # InitSequence is required by the upstream examples
    m1x_0 = X_train[:, :, 0].mean(dim=0).view(state_dim, 1)
    m2x_0 = prior_Sigma.clone()
    sys_model.InitSequence(m1x_0, m2x_0)

    # Prepare init tensors (randomInit=True in Pipeline_EKF)
    init_train = ensure_init_shape(D.get("init_train", None), state_dim)
    init_val = ensure_init_shape(D.get("init_val", None), state_dim)
    init_test = ensure_init_shape(D.get("init_test", None), state_dim)

    # Fix batch size if user picked too large
    n_batch = min(int(args.n_batch), int(N_train))
    if n_batch < 1:
        raise ValueError("n_batch must be >= 1")

    # Training args object that Pipeline_EKF + KalmanNetNN expect
    train_args = argparse.Namespace(
        use_cuda=bool(args.use_cuda),
        n_steps=int(args.n_steps),
        n_batch=int(n_batch),
        lr=float(args.lr),
        wd=float(args.wd),
        # we start simple: pure state MSE
        CompositionLoss=False,
        alpha=0.0,
        randomLength=False,
        # KNet sizes
        in_mult_KNet=int(args.in_mult_KNet),
        out_mult_KNet=int(args.out_mult_KNet),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_results = str(out_dir) + "/"

    run_name = args.model_name.strip()
    if not run_name:
        run_name = f"knet_range_{data_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"

    print("[INFO] Dataset:", data_path)
    print(f"[INFO] Shapes: Y_train={tuple(Y_train.shape)} X_train={tuple(X_train.shape)}")
    print(f"[INFO] Layout: {layout_file} (N={sensors_xy.shape[0]})")
    print(f"[INFO] Model params: dt={args.dt} tau={args.tau} sigma={args.sigma}")
    print(f"[INFO] Train: steps={train_args.n_steps} batch={train_args.n_batch} lr={train_args.lr} wd={train_args.wd}")
    print("[INFO] Out:", out_dir)
    print("[INFO] Run name:", run_name)

    # Pipeline (same pattern as KalmanNet_TSP examples)
    pipe = Pipeline_EKF(time.strftime("%m.%d.%y_%H:%M:%S"), str(out_dir), run_name)
    pipe.setssModel(sys_model)

    model = KalmanNetNN()
    model.NNBuild(sys_model, train_args)

    pipe.setModel(model)
    pipe.setTrainingParams(train_args)

    # Train
    pipe.NNTrain(
        sys_model,
        Y_val, X_val,
        Y_train, X_train,
        path_results,
        randomInit=True,
        cv_init=init_val,
        train_init=init_train,
    )

    # Test (loads best-model.pt internally)
    test_out = pipe.NNTest(
        sys_model,
        Y_test, X_test,
        path_results,
        randomInit=True,
        test_init=init_test,
    )

    # Unpack + save quick summary
    mse_test_arr, mse_test_avg, mse_test_avg_db, x_hat_test, t_elapsed = test_out
    summary = {
        "data": str(data_path),
        "layout_file": str(layout_file),
        "dt": args.dt,
        "tau": args.tau,
        "sigma": args.sigma,
        "init_pos_std": args.init_pos_std,
        "init_vel_std": args.init_vel_std,
        "train": {
            "n_steps": train_args.n_steps,
            "n_batch": train_args.n_batch,
            "lr": train_args.lr,
            "wd": train_args.wd,
            "in_mult_KNet": train_args.in_mult_KNet,
            "out_mult_KNet": train_args.out_mult_KNet,
        },
        "test": {
            "mse_avg": float(mse_test_avg),
            "mse_avg_db": float(mse_test_avg_db),
            "elapsed_sec": float(t_elapsed),
        },
        "artifacts": {
            "best_model": str(out_dir / "best-model.pt"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        {"x_hat_test": x_hat_test.cpu(), "x_true_test": X_test.cpu(), "y_test": Y_test.cpu()},
        str(out_dir / "test_predictions.pt"),
    )

    print("\n[DONE] Training+Test complete.")
    print("       best model :", out_dir / "best-model.pt")
    print("       summary    :", out_dir / "summary.json")
    print("       preds      :", out_dir / "test_predictions.pt")


if __name__ == "__main__":
    main()
