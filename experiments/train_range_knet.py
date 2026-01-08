#!/usr/bin/env python3
"""
Train KalmanNet (KalmanNet_TSP) on range-only tracking dataset exported as .pt

Dataset keys:
  Y_* : (N, obs_dim, T)
  X_* : (N, 4, T)   [px, py, vx, vy] in world frame
  init_* : (N, 4) or (N,4,1)

Main fixes vs NaN + save issues:
  - f/h are module-level class methods (pickle-safe)
  - range eps default = min_range^2 (prevents insane gradients near 0)
  - optional data filtering (residual + speed) so breakdance/outliers don't kill training
  - global optimizer.step patch: non-finite grad skip + grad clipping
  - best-model.pt corruption handling
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
import faulthandler, signal
faulthandler.register(signal.SIGUSR1)


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
        raise ValueError(f"Could not parse any (x,y) rows from {path}.")
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
        t = t.unsqueeze(-1)
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


def build_cv_Q_base(dt: float, device=None, dtype=torch.float32) -> torch.Tensor:
    d = float(dt)
    return torch.tensor(
        [[d**3 / 3.0, 0.0,        d**2 / 2.0, 0.0],
         [0.0,        d**3 / 3.0, 0.0,        d**2 / 2.0],
         [d**2 / 2.0, 0.0,        d,          0.0],
         [0.0,        d**2 / 2.0, 0.0,        d]],
        dtype=dtype,
        device=device,
    )


def build_cv_Q(dt: float, tau: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return float(tau) * build_cv_Q_base(dt, device=device, dtype=dtype)


def build_R(sigma: float, n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    return (float(sigma) ** 2) * torch.eye(n, dtype=dtype, device=device)


class RangeCV2DModel:
    """
    Pickle-safe f/h for 2D range-only tracking with constant-velocity dynamics.

    f/h signatures match KalmanNet_TSP Extended_sysmdl:
      x: (B,4,1)
      f(x, jacobian=False) -> x_next OR (x_next, F)
      h(x, jacobian=False) -> y      OR (y, H)
    """

    def __init__(self, dt: float, sensors_xy: torch.Tensor, min_range: float = 1e-3, eps: float | None = None):
        self.dt = float(dt)
        self.sensors_xy = sensors_xy.float().clone()
        self.sx = self.sensors_xy[:, 0].clone()
        self.sy = self.sensors_xy[:, 1].clone()
        self.n = int(self.sensors_xy.shape[0])
        self.min_range = float(min_range)
        if eps is None or eps <= 0.0:
            self.eps = float(self.min_range ** 2)
        else:
            self.eps = float(max(eps, self.min_range ** 2))

    def f(self, x: torch.Tensor, jacobian: bool = False):
        B = x.shape[0]
        F = build_cv_F(self.dt, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        x_next = torch.bmm(F, x)
        if jacobian:
            return x_next, F
        return x_next

    def h(self, x: torch.Tensor, jacobian: bool = False):
        px = x[:, 0, 0].unsqueeze(1)  # (B,1)
        py = x[:, 1, 0].unsqueeze(1)

        dx = px - self.sx.to(x.device).unsqueeze(0)  # (B,n)
        dy = py - self.sy.to(x.device).unsqueeze(0)

        d = torch.sqrt(dx * dx + dy * dy + self.eps)  # (B,n)
        d = d.clamp_min(self.min_range)
        y = d.unsqueeze(2)  # (B,n,1)

        if jacobian:
            H = torch.zeros((x.shape[0], self.n, 4), dtype=x.dtype, device=x.device)
            inv_d = 1.0 / d.clamp_min(self.min_range)  # IMPORTANT: clamp to min_range (not eps)
            H[:, :, 0] = dx * inv_d
            H[:, :, 1] = dy * inv_d
            return y, H

        return y


def estimate_sigma_from_data(Y: torch.Tensor, X: torch.Tensor, sensors_xy: torch.Tensor, min_range: float, eps: float) -> float:
    """
    Estimate measurement noise sigma from residuals: y - h(x_gt).
    Uses RMS over all sequences, sensors, time.
    """
    with torch.no_grad():
        px = X[:, 0, :]  # (N,T)
        py = X[:, 1, :]
        sx = sensors_xy[:, 0].view(1, -1, 1)  # (1,n,1)
        sy = sensors_xy[:, 1].view(1, -1, 1)

        dx = px.unsqueeze(1) - sx  # (N,n,T)
        dy = py.unsqueeze(1) - sy
        pred = torch.sqrt(dx * dx + dy * dy + eps).clamp_min(min_range)  # (N,n,T)
        resid = (Y - pred).float()
        sigma = torch.sqrt(torch.mean(resid * resid)).item()
        return float(max(sigma, 1e-6))


def estimate_tau_from_data(X: torch.Tensor, dt: float) -> float:
    """
    Rough estimate of tau in Q = tau * Q_base using velocity residuals:
      r_k = x_{k+1} - F x_k
    """
    with torch.no_grad():
        F = build_cv_F(dt, device=X.device, dtype=X.dtype)  # (4,4)
        # X: (N,4,T)
        xk = X[:, :, :-1]  # (N,4,T-1)
        xk1 = X[:, :, 1:]  # (N,4,T-1)
        # apply F: (N,4,T-1) where each (4) multiplies (4)
        Fxk = torch.einsum("ij,njt->nit", F, xk)  # (N,4,T-1)
        r = (xk1 - Fxk).float()  # residual
        # use velocity dims only (2,3)
        rv = r[:, 2:4, :].reshape(-1, 2)  # (N*(T-1),2)
        cov = torch.mean(rv * rv, dim=0)  # approx diag variances for vx,vy residual
        q_base = build_cv_Q_base(dt, device=X.device, dtype=X.dtype)
        denom = torch.tensor([q_base[2, 2], q_base[3, 3]], dtype=torch.float32, device=X.device).clamp_min(1e-12)
        tau = torch.mean(cov / denom).item()
        # keep it sane
        tau = float(np.clip(tau, 1e-4, 100.0))
        return tau


def filter_sequences(Y: torch.Tensor, X: torch.Tensor, sensors_xy: torch.Tensor,
                     min_range: float, eps: float, max_residual_rms: float, max_speed: float) -> torch.Tensor:
    """
    Returns boolean mask (N,) of "good" sequences.
    """
    N, n, T = Y.shape
    with torch.no_grad():
        finite = torch.isfinite(Y).all(dim=(1, 2)) & torch.isfinite(X).all(dim=(1, 2))

        # speed check
        vx = X[:, 2, :]
        vy = X[:, 3, :]
        speed = torch.sqrt(vx * vx + vy * vy)
        finite &= torch.isfinite(speed).all(dim=1)
        finite &= (speed.max(dim=1).values <= max_speed)

        # residual check vs geometry
        px = X[:, 0, :]
        py = X[:, 1, :]
        sx = sensors_xy[:, 0].view(1, -1, 1)
        sy = sensors_xy[:, 1].view(1, -1, 1)
        dx = px.unsqueeze(1) - sx
        dy = py.unsqueeze(1) - sy
        pred = torch.sqrt(dx * dx + dy * dy + eps).clamp_min(min_range)
        resid = (Y - pred).float()
        resid_rms_t = torch.sqrt(torch.mean(resid * resid, dim=1))  # (N,T)
        resid_rms_med = resid_rms_t.median(dim=1).values  # (N,)
        finite &= (resid_rms_med <= max_residual_rms)

        return finite


def patch_optimizer_step(grad_clip: float, skip_nonfinite_grad: bool = True):
    """
    Patch torch.optim.Optimizer.step so Pipeline_EKF 내부 optimizer.step() çağrıları da güvenli olsun.
    """
    if grad_clip <= 0 and not skip_nonfinite_grad:
        return None

    orig_step = torch.optim.Optimizer.step

    def step_with_safety(self, closure=None):
        params = []
        for group in self.param_groups:
            for p in group.get("params", []):
                if p is not None and p.grad is not None:
                    params.append(p)

        if skip_nonfinite_grad:
            for p in params:
                g = p.grad
                if g is not None and (not torch.isfinite(g).all()):
                    # skip this step to avoid poisoning weights
                    try:
                        self.zero_grad(set_to_none=True)
                    except TypeError:
                        self.zero_grad()
                    return None

        if grad_clip > 0 and len(params) > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))

        return orig_step(self, closure=closure)

    torch.optim.Optimizer.step = step_with_safety
    return orig_step


def is_loadable_torch(path: Path, device: str) -> bool:
    if not path.exists():
        return False
    try:
        _ = torch.load(str(path), map_location=device)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--layout_file", default="config/paper_sensors_5x5_b20.csv")
    parser.add_argument("--out_dir", default="models/knet_range_v1")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.10)

    parser.add_argument("--estimate_tau", action="store_true", help="Estimate tau from GT states (overrides --tau)")
    parser.add_argument("--estimate_sigma", action="store_true", help="Estimate sigma from (Y - h(X_gt)) (overrides --sigma)")

    parser.add_argument("--min_range", type=float, default=1e-3)
    parser.add_argument("--eps_range", type=float, default=0.0, help="0 => min_range^2")

    parser.add_argument("--init_pos_std", type=float, default=5.0)
    parser.add_argument("--init_vel_std", type=float, default=2.0)

    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--n_batch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-6)

    # safer defaults than your previous 40/5
    parser.add_argument("--in_mult_KNet", type=int, default=10)
    parser.add_argument("--out_mult_KNet", type=int, default=1)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")

    # optional filtering (highly recommended)
    parser.add_argument("--filter_bad_seq", action="store_true", help="Drop sequences with large residual/speed.")
    parser.add_argument("--max_residual_rms", type=float, default=2.0)
    parser.add_argument("--max_speed", type=float, default=6.0)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    add_kalman_tsp_to_path(repo_root)

    from Simulations.Extended_sysmdl import SystemModel
    from Pipelines.Pipeline_EKF import Pipeline_EKF
    from KNet.KalmanNet_nn import KalmanNetNN

    if args.use_cuda and not torch.cuda.is_available():
        print("[WARN] --use_cuda given but CUDA not available. Falling back to CPU.")
        args.use_cuda = False

    device = "cuda" if args.use_cuda else "cpu"

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("[INFO] torch.autograd anomaly detection ENABLED")

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
    sensors_xy = load_layout_csv(layout_file)
    if sensors_xy.shape[0] != obs_dim:
        raise ValueError(
            f"layout sensor count ({sensors_xy.shape[0]}) != obs_dim ({obs_dim}). "
            "Same layout file must be used as sim."
        )

    min_range = float(args.min_range)
    eps = float(args.eps_range) if float(args.eps_range) > 0 else float(min_range ** 2)
    eps = float(max(eps, min_range ** 2))

    # optional: filter sequences to avoid breakdance / out-of-plane mismatches killing training
    if args.filter_bad_seq:
        m_tr = filter_sequences(Y_train, X_train, sensors_xy, min_range, eps, args.max_residual_rms, args.max_speed)
        m_va = filter_sequences(Y_val, X_val, sensors_xy, min_range, eps, args.max_residual_rms, args.max_speed)
        m_te = filter_sequences(Y_test, X_test, sensors_xy, min_range, eps, args.max_residual_rms, args.max_speed)

        def apply_mask(Y, X, init, mask, name):
            if mask.sum().item() < 1:
                print(f"[WARN] {name}: filtering removed everything. Keeping original split.")
                return Y, X, init
            return Y[mask], X[mask], (init[mask] if init is not None else None)

        init_train_raw = D.get("init_train", None)
        init_val_raw = D.get("init_val", None)
        init_test_raw = D.get("init_test", None)

        init_train = init_train_raw.float() if init_train_raw is not None else None
        init_val = init_val_raw.float() if init_val_raw is not None else None
        init_test = init_test_raw.float() if init_test_raw is not None else None

        Y_train, X_train, init_train = apply_mask(Y_train, X_train, init_train, m_tr, "train")
        Y_val, X_val, init_val = apply_mask(Y_val, X_val, init_val, m_va, "val")
        Y_test, X_test, init_test = apply_mask(Y_test, X_test, init_test, m_te, "test")

        print(f"[INFO] After filtering: train={Y_train.shape[0]} val={Y_val.shape[0]} test={Y_test.shape[0]}")
    else:
        init_train = D.get("init_train", None)
        init_val = D.get("init_val", None)
        init_test = D.get("init_test", None)

    # init tensors -> (N,4,1)
    init_train = ensure_init_shape(init_train, state_dim)
    init_val = ensure_init_shape(init_val, state_dim)
    init_test = ensure_init_shape(init_test, state_dim)

    # estimate noise if requested
    tau = float(args.tau)
    sigma = float(args.sigma)

    if args.estimate_tau:
        tau = estimate_tau_from_data(X_train, float(args.dt))
    if args.estimate_sigma:
        sigma = estimate_sigma_from_data(Y_train, X_train, sensors_xy, min_range=min_range, eps=eps)

    # build SystemModel
    Q = build_cv_Q(args.dt, tau)
    R = build_R(sigma, obs_dim)

    prior_Sigma = torch.diag(
        torch.tensor(
            [args.init_pos_std**2, args.init_pos_std**2, args.init_vel_std**2, args.init_vel_std**2],
            dtype=torch.float32,
        )
    )

    range_model = RangeCV2DModel(dt=args.dt, sensors_xy=sensors_xy, min_range=min_range, eps=eps)

    sys_model = SystemModel(
        range_model.f, Q,
        range_model.h, R,
        T, T,
        state_dim, obs_dim,
        prior_Q=Q, prior_Sigma=prior_Sigma, prior_S=R
    )

    # InitSequence
    m1x_0 = X_train[:, :, 0].mean(dim=0).view(state_dim, 1)
    m2x_0 = prior_Sigma.clone()
    sys_model.InitSequence(m1x_0, m2x_0)

    n_batch = min(int(args.n_batch), int(Y_train.shape[0]))
    if n_batch < 1:
        raise ValueError("n_batch must be >= 1")

    train_args = argparse.Namespace(
        use_cuda=bool(args.use_cuda),
        n_steps=int(args.n_steps),
        n_batch=int(n_batch),
        lr=float(args.lr),
        wd=float(args.wd),
        CompositionLoss=False,
        alpha=0.0,
        randomLength=False,
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
    print(f"[INFO] Model params: dt={args.dt} tau={tau} sigma={sigma}  (min_range={min_range}, eps={eps})")
    print(f"[INFO] Train: steps={train_args.n_steps} batch={train_args.n_batch} lr={train_args.lr} wd={train_args.wd} "
          f"in_mult={train_args.in_mult_KNet} out_mult={train_args.out_mult_KNet}")
    print("[INFO] Out:", out_dir)
    print("[INFO] Run name:", run_name)

    # patch optimizer.step globally (affects Pipeline_EKF internal optimizer)
    patch_optimizer_step(grad_clip=float(args.grad_clip), skip_nonfinite_grad=True)
    print(f"[INFO] Patched optimizer.step with grad_clip={args.grad_clip} (skip non-finite grads)")

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

    # ensure best-model.pt exists and is loadable
    best_path = out_dir / "best-model.pt"
    if not is_loadable_torch(best_path, device="cpu"):
        print("[WARN] best-model.pt missing/corrupt (validation NaN olabilir). Saving current model as best-model.pt")
        try:
            # now pickle-safe because f/h are class methods (not local closures)
            torch.save(pipe.model, str(best_path))
        except Exception as e:
            print("[ERROR] Could not torch.save(model) ->", repr(e))
            print("[HINT] Skipping NNTest. You still have training logs; fix NaNs first.")
            args.no_test = True

    # Test
    test_out = None
    if not args.no_test:
        try:
            test_out = pipe.NNTest(
                sys_model,
                Y_test, X_test,
                path_results,
                randomInit=True,
                test_init=init_test,
            )
        except Exception as e:
            print("[ERROR] NNTest failed:", repr(e))
            print("[HINT] Re-run with --no_test until training stabilizes.")
            test_out = None

    summary = {
        "data": str(data_path),
        "layout_file": str(layout_file),
        "dt": float(args.dt),
        "tau": float(tau),
        "sigma": float(sigma),
        "min_range": float(min_range),
        "eps_range": float(eps),
        "train": {
            "n_steps": int(train_args.n_steps),
            "n_batch": int(train_args.n_batch),
            "lr": float(train_args.lr),
            "wd": float(train_args.wd),
            "in_mult_KNet": int(train_args.in_mult_KNet),
            "out_mult_KNet": int(train_args.out_mult_KNet),
            "grad_clip": float(args.grad_clip),
            "filter_bad_seq": bool(args.filter_bad_seq),
            "max_residual_rms": float(args.max_residual_rms),
            "max_speed": float(args.max_speed),
        },
        "artifacts": {
            "best_model": str(best_path),
            "summary_json": str(out_dir / "summary.json"),
        },
    }

    if test_out is not None:
        mse_test_arr, mse_test_avg, mse_test_avg_db, x_hat_test, t_elapsed = test_out
        summary["test"] = {
            "mse_avg": float(mse_test_avg),
            "mse_avg_db": float(mse_test_avg_db),
            "elapsed_sec": float(t_elapsed),
        }
        torch.save(
            {"x_hat_test": x_hat_test.cpu(), "x_true_test": X_test.cpu(), "y_test": Y_test.cpu()},
            str(out_dir / "test_predictions.pt"),
        )
        summary["artifacts"]["preds"] = str(out_dir / "test_predictions.pt")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[DONE] Training complete.")
    print("       best model :", best_path)
    print("       summary    :", out_dir / "summary.json")
    if test_out is not None:
        print("       preds      :", out_dir / "test_predictions.pt")


if __name__ == "__main__":
    main()
