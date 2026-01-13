#!/usr/bin/env python3
from __future__ import annotations

"""
knet_tracker_from_range.py

ROS2 node: KalmanNetNN tracker for 2D range-only measurements.

✅ Supports checkpoints saved as:
  1) full torch.nn.Module (torch.save(model, ...))
  2) training checkpoint dict with 'state_dict' (your best-model.pt is this)

Your checkpoint:
  {'state_dict', 'step', 'cv_mse', 'cv_db'}
so we rebuild the KalmanNetNN architecture from state_dict layer shapes and load weights.

Sub:
  - z_topic  (std_msgs/Float32MultiArray) : N ranges
  - gt_topic (nav_msgs/Odometry)          : only if init_from_gt=true

Pub:
  - est_topic (nav_msgs/Odometry)         : [x,y,(vx,vy)]

Notes:
  - Online (batch_size = 1)
  - If the terminal "looks stuck": that's normal — rclpy.spin() waits for messages.
    This script prints wall-time status logs periodically to prove it’s alive.
"""

import importlib.util
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.exceptions import ParameterAlreadyDeclaredException

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


# ======================================================================================
# Layout helpers
# ======================================================================================

def load_layout_csv(path: str) -> np.ndarray:
    """CSV lines: 'x,y' or 'x y'. '#' comments allowed. Returns (N,2) float32."""
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) < 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not pts:
        raise RuntimeError(f"layout parse edilemedi/boş: {path}")
    return np.asarray(pts, dtype=np.float32)


def predict_ranges_xy(
    xy: torch.Tensor,
    sensors_xy: torch.Tensor,
    *,
    eps: float,
    min_range: float,
) -> torch.Tensor:
    """xy:(2,) sensors_xy:(N,2) -> (N,) predicted ranges"""
    dx = xy[0] - sensors_xy[:, 0]
    dy = xy[1] - sensors_xy[:, 1]
    d = torch.sqrt(dx * dx + dy * dy + float(eps))
    return torch.clamp(d, min=float(min_range))


def ls_init_xy_from_ranges(
    sensors_xy: np.ndarray,
    z: np.ndarray,
    *,
    min_range: float = 1e-3,
) -> np.ndarray:
    """
    Fast linear LS multilateration init in 2D.

    sensors_xy: (N,2)
    z: (N,)
    returns: (2,) float32
    """
    sensors_xy = np.asarray(sensors_xy, dtype=float).reshape((-1, 2))
    z = np.asarray(z, dtype=float).reshape((-1,))
    N = sensors_xy.shape[0]
    if z.shape[0] != N:
        raise ValueError(f"z length {z.shape[0]} != N {N}")
    if N < 3:
        raise ValueError(f"Need at least 3 sensors, got N={N}")

    z = np.maximum(z, float(min_range))

    # Reference = smallest range
    i0 = int(np.argmin(z))
    x0, y0 = sensors_xy[i0]
    z0 = float(z[i0])

    A = []
    b = []
    for i in range(N):
        if i == i0:
            continue
        xi, yi = sensors_xy[i]
        zi = float(z[i])
        A.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b.append((xi * xi - x0 * x0) + (yi * yi - y0 * y0) - (zi * zi - z0 * z0))

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("Degenerate sensor geometry (rank < 2)")

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol.astype(np.float32)


# ======================================================================================
# KalmanNetNN import (robust)
# ======================================================================================

def _try_import_kalmannet_from_sys_path():
    """Try: from KalmanNet_nn import KalmanNetNN (if repo root is on sys.path)."""
    try:
        from KalmanNet_nn import KalmanNetNN  # type: ignore
        return KalmanNetNN
    except Exception:
        return None


def _find_repo_root(start: Path) -> Path:
    """
    Find a sensible 'repo root' by walking upwards until we see 'scripts/'.
    If not found, fallback to script parent.
    """
    for p in [start] + list(start.parents):
        if (p / "scripts").is_dir():
            return p
    return start


def _find_kalmannet_nn_file(explicit: str = "") -> Optional[Path]:
    """
    Locate KalmanNet_nn.py.

    Search order:
      1) explicit path (if provided)
      2) repo root candidates upwards from this file
      3) limited recursive search under repo root
    """
    if explicit.strip():
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None

    here = Path(__file__).resolve()
    repo_root = _find_repo_root(here.parent)

    # Fast common locations
    candidates = [
        repo_root / "KalmanNet_nn.py",
        repo_root / "src" / "KalmanNet_nn.py",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Search a bit more (bounded)
    try:
        hits = list(repo_root.rglob("KalmanNet_nn.py"))
        for h in hits:
            if h.is_file():
                return h
    except Exception:
        pass

    return None


def _import_kalmannet_nn(explicit_file: str = ""):
    """
    Return KalmanNetNN class.

    If file exists -> dynamic import by file path.
    Else try normal import.
    Else fallback to embedded implementation.
    """
    # 1) Normal import if possible
    cls = _try_import_kalmannet_from_sys_path()
    if cls is not None and not explicit_file.strip():
        return cls

    # 2) Dynamic import by file
    path = _find_kalmannet_nn_file(explicit_file)
    if path is not None:
        spec = importlib.util.spec_from_file_location("KalmanNet_nn", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"KalmanNet_nn import spec oluşturulamadı: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if not hasattr(module, "KalmanNetNN"):
            raise ImportError(f"{path} içinde KalmanNetNN class bulunamadı")
        return module.KalmanNetNN

    # 3) Embedded fallback (last resort)
    return _EmbeddedKalmanNetNN


# ======================================================================================
# Embedded KalmanNetNN (fallback) — identical to repo's KalmanNet_nn.py
# ======================================================================================

import torch.nn as nn
import torch.nn.functional as func


class _EmbeddedKalmanNetNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def NNBuild(self, SysModel, args):
        self.device = torch.device("cuda" if args.use_cuda else "cpu")
        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        self.seq_len_input = 1
        self.batch_size = args.n_batch

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU()).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        ).to(self.device)

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU()).to(self.device)

        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU()).to(self.device)

        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU()).to(self.device)

        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU()).to(self.device)

        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU()).to(self.device)

    def InitSystemDynamics(self, f, h, m, n):
        self.f = f
        self.m = m
        self.h = h
        self.n = n

    def InitSequence(self, M1_0, T):
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    def step_prior(self):
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = self.h(self.m1x_prior)

    def step_KGain_est(self, y):
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(self.m1x_prior_previous, 2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def KNet_step(self, y):
        self.step_prior()
        self.step_KGain_est(y)
        dy = y - self.m1y
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        return self.m1x_posterior

    def KGain_step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        out_FC5 = self.FC5(fw_update_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_evol_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        self.h_Sigma = out_FC4
        return out_FC2

    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)

    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)


# ======================================================================================
# Checkpoint helpers
# ======================================================================================

def _torch_load_any(path: str):
    """torch.load wrapper that works across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_module_prefix(sd: dict) -> dict:
    """If saved under DataParallel, keys start with 'module.'"""
    out = {}
    for k, v in sd.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out


def _looks_like_kalmannet_state_dict(sd: dict) -> bool:
    keys = set(sd.keys())
    must = {
        "GRU_Q.weight_ih_l0",
        "GRU_Sigma.weight_ih_l0",
        "GRU_S.weight_ih_l0",
        "FC2.0.weight",
        "FC5.0.weight",
        "FC7.0.weight",
    }
    return len(must - keys) == 0


def _infer_kalmannet_hparams(sd: dict) -> Tuple[int, int, int, int]:
    """
    Infer (m, n, in_mult, out_mult) from KalmanNetNN state_dict layer shapes.
    """
    w5 = sd["FC5.0.weight"]  # (m*in_mult, m)
    w7 = sd["FC7.0.weight"]  # (2*n*in_mult, 2*n)
    w2 = sd["FC2.0.weight"]  # ((m^2+n^2)*out_mult, (m^2+n^2))

    m = int(w5.shape[1])
    in_mult = int(w5.shape[0] // max(m, 1))

    n2 = int(w7.shape[1])
    if n2 % 2 != 0:
        raise RuntimeError(f"FC7.0.weight in_features expected 2*n, got {n2}")
    n = int(n2 // 2)

    din = int(w2.shape[1])
    if din <= 0:
        raise RuntimeError("FC2.0.weight in_features invalid")

    if din != (m * m + n * n):
        raise RuntimeError(f"state_dict inconsistent: FC2 in_features={din} != m^2+n^2={m*m+n*n}")

    out_mult = int(w2.shape[0] // din)
    return m, n, in_mult, out_mult


@dataclass
class _SysModel:
    f: object
    h: object
    m: int
    n: int
    prior_Q: torch.Tensor
    prior_Sigma: torch.Tensor
    prior_S: torch.Tensor


def _build_kalmannet_from_state_dict(
    sd: dict,
    sensors_xy: torch.Tensor,
    *,
    dt: float,
    device: torch.device,
    kalmannet_nn_file: str = "",
):
    """
    Rebuild KalmanNetNN (architecture) and return model (weights not loaded yet).
    """
    KalmanNetNN = _import_kalmannet_nn(kalmannet_nn_file)

    m, n, in_mult, out_mult = _infer_kalmannet_hparams(sd)

    if int(sensors_xy.shape[0]) != n:
        raise RuntimeError(f"Model expects n={n} sensors but layout has N={int(sensors_xy.shape[0])}.")

    dt = float(dt)

    # f(x): constant velocity on first 4 dims, identity on the rest
    def f(x: torch.Tensor) -> torch.Tensor:
        # x: (B,m,1)
        B = int(x.shape[0])
        F = torch.eye(m, device=x.device, dtype=x.dtype)
        if m >= 4:
            F[0, 2] = dt
            F[1, 3] = dt
        F = F.unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(F, x)

    # h(x): ranges to sensors using first 2 dims as (x,y)
    eps = 1e-6

    def h(x: torch.Tensor) -> torch.Tensor:
        # x: (B,m,1) -> (B,n,1)
        B = int(x.shape[0])
        px = x[:, 0, 0].view(B, 1)
        py = x[:, 1, 0].view(B, 1)
        dx = px - sensors_xy[:, 0].view(1, n)
        dy = py - sensors_xy[:, 1].view(1, n)
        d = torch.sqrt(dx * dx + dy * dy + eps)
        return d.view(B, n, 1)

    # Priors (match Extended_sysmdl.py defaults: Q=I, Sigma=0, S=I)
    prior_Q = torch.eye(m, device=device)
    prior_Sigma = torch.zeros((m, m), device=device)
    prior_S = torch.eye(n, device=device)

    sysm = _SysModel(f=f, h=h, m=m, n=n, prior_Q=prior_Q, prior_Sigma=prior_Sigma, prior_S=prior_S)

    args = SimpleNamespace(
        use_cuda=(device.type == "cuda"),
        n_batch=1,          # online
        in_mult_KNet=in_mult,
        out_mult_KNet=out_mult,
    )

    model = KalmanNetNN()
    model.NNBuild(sysm, args)
    return model


# ======================================================================================
# ROS2 Node
# ======================================================================================

class KNetTracker(Node):
    def __init__(self):
        super().__init__("knet_tracker")

        # ---------------- Params (declare) ----------------
        # use_sim_time is often already declared by ROS2 — don't crash if it is.
        try:
            self.declare_parameter("use_sim_time", False)
        except ParameterAlreadyDeclaredException:
            pass

        self.declare_parameter("model_path", "")
        self.declare_parameter("layout_file", "")

        # Topics (relative defaults -> namespace friendly)
        self.declare_parameter("z_topic", "z")
        self.declare_parameter("est_topic", "estimated")

        # Init
        self.declare_parameter("init_from_gt", True)
        self.declare_parameter("gt_topic", "gt/odom")

        # Frames
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("child_frame", "base_link")

        # Runtime / model
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("delta", 0.1)        # dt
        self.declare_parameter("max_seq_len", 10000)

        # Range helpers
        self.declare_parameter("min_range", 1e-3)
        self.declare_parameter("eps_range", 1e-6)

        # Gating
        self.declare_parameter("gate_resid_rms", 0.30)

        # Optional explicit KalmanNet_nn.py location
        self.declare_parameter("kalmannet_nn_file", "")

        # ---------------- Read params ----------------
        self.model_path = str(self.get_parameter("model_path").value)
        self.layout_file = str(self.get_parameter("layout_file").value)
        self.z_topic = str(self.get_parameter("z_topic").value)
        self.est_topic = str(self.get_parameter("est_topic").value)

        self.init_from_gt = bool(self.get_parameter("init_from_gt").value)
        self.gt_topic = str(self.get_parameter("gt_topic").value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)

        use_cuda_req = bool(self.get_parameter("use_cuda").value)
        self.dt = float(self.get_parameter("delta").value)
        self.max_seq_len = int(self.get_parameter("max_seq_len").value)

        self.min_range = float(self.get_parameter("min_range").value)
        self.eps_range = float(self.get_parameter("eps_range").value)
        self.gate_resid_rms = float(self.get_parameter("gate_resid_rms").value)

        self.kalmannet_nn_file = str(self.get_parameter("kalmannet_nn_file").value).strip()

        # ---------------- Sanity ----------------
        if not self.model_path:
            raise RuntimeError("model_path param boş. Örn: -p model_path:=models/.../best-model.pt")
        if not self.layout_file:
            raise RuntimeError("layout_file param boş. Örn: -p layout_file:=config/paper_sensors_5x5_b20.csv")
        if self.dt <= 0.0:
            raise RuntimeError("delta(dt) > 0 olmalı")
        if self.min_range <= 0.0:
            raise RuntimeError("min_range > 0 olmalı")

        mp = Path(self.model_path)
        lf = Path(self.layout_file)
        if not mp.is_file():
            raise RuntimeError(f"model_path bulunamadı: {mp}")
        if not lf.is_file():
            raise RuntimeError(f"layout_file bulunamadı: {lf}")

        # ---------------- Device ----------------
        self.device = torch.device("cuda") if (use_cuda_req and torch.cuda.is_available()) else torch.device("cpu")
        self.get_logger().info(f"device={self.device}")

        # ---------------- Load layout ----------------
        sensors = load_layout_csv(self.layout_file)
        self.sensors_xy_np = sensors
        self.sensors_xy = torch.tensor(sensors, dtype=torch.float32, device=self.device)
        self.obs_dim = int(self.sensors_xy.shape[0])
        self.get_logger().info(f"layout loaded: N_sensors={self.obs_dim}")

        # ---------------- Load model ----------------
        self.model = self._load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.state_dim = int(getattr(self.model, "m", 4))
        self.get_logger().info(f"model ready: state_dim(m)={self.state_dim}, obs_dim(n)={self.obs_dim}")

        # ---------------- State ----------------
        self._initialized = False
        self._step = 0
        self._last_gt: Optional[Odometry] = None
        self._prev_xy: Optional[Tuple[float, float]] = None

        # Debug / liveness
        self._last_z_wall: Optional[float] = None
        self._last_gt_wall: Optional[float] = None
        self._last_wait_log_wall: float = 0.0
        self._stop_evt = threading.Event()
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

        # Throttle for "waiting for GT" messages from _on_z
        self._last_gt_wait_log_wall: float = 0.0

        # ---------------- ROS I/O ----------------
        self.pub = self.create_publisher(Odometry, self.est_topic, 10)
        self.create_subscription(Float32MultiArray, self.z_topic, self._on_z, qos_profile_sensor_data)

        if self.init_from_gt:
            self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
            self.get_logger().info(f"init_from_gt=true -> subscribing gt_topic={self.gt_topic}")

        self.get_logger().info(f"sub z_topic={self.z_topic} pub est_topic={self.est_topic}")

    # ---------------- Liveness thread ----------------
    def _status_loop(self):
        """
        Wall-time liveness logs (works even if sim-time /clock is not running).
        """
        while not self._stop_evt.is_set():
            time.sleep(2.0)
            now = time.monotonic()

            # Throttle logs
            if now - self._last_wait_log_wall < 5.0:
                continue

            # No Z received yet
            if self._last_z_wall is None:
                self._last_wait_log_wall = now
                msg = f"Waiting for first Z on topic='{self.get_namespace().rstrip('/')}/{self.z_topic}' (or '{self.z_topic}' relative)."
                if self.init_from_gt and self._last_gt is None:
                    msg += f" Also waiting for GT on '{self.gt_topic}'."
                try:
                    self.get_logger().warn(msg)
                except Exception:
                    pass
                continue

            # Z received but init_from_gt blocks initialization
            if self.init_from_gt and (self._last_gt is None):
                self._last_wait_log_wall = now
                try:
                    self.get_logger().warn(f"Z is coming, but still waiting for GT on '{self.gt_topic}' to initialize.")
                except Exception:
                    pass
                continue

            # Normal running but maybe stale Z
            if self._last_z_wall is not None and (now - self._last_z_wall) > 5.0:
                self._last_wait_log_wall = now
                try:
                    self.get_logger().warn(f"No Z received for {(now - self._last_z_wall):.1f}s on '{self.z_topic}'.")
                except Exception:
                    pass

    # ---------------- Model loading ----------------
    def _load_model(self, path: str) -> torch.nn.Module:
        obj = _torch_load_any(path)

        # 1) Full model
        if isinstance(obj, torch.nn.Module):
            self.get_logger().info("Loaded full torch.nn.Module from checkpoint.")
            return obj

        # 2) Training checkpoint dict with state_dict
        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
            if not isinstance(sd, dict):
                raise RuntimeError("checkpoint['state_dict'] dict değil.")

            sd = _strip_module_prefix(sd)

            if not _looks_like_kalmannet_state_dict(sd):
                ex = list(sd.keys())[:40]
                raise RuntimeError(
                    "Checkpoint state_dict KalmanNetNN'e benzemiyor.\n"
                    f"İlk anahtarlar: {ex}\n"
                    "Eğer bu model KalmanNetNN değilse, doğru mimariyi burada yeniden kurmak gerekir."
                )

            model = _build_kalmannet_from_state_dict(
                sd,
                self.sensors_xy,
                dt=self.dt,
                device=self.device,
                kalmannet_nn_file=self.kalmannet_nn_file,
            )

            try:
                model.load_state_dict(sd, strict=True)
            except Exception as e:
                raise RuntimeError(
                    "load_state_dict failed (checkpoint ile mimari uyuşmuyor).\n"
                    "Genelde sebep: eğitimde farklı KalmanNet_nn.py versiyonu / farklı m-n kullanılmış.\n"
                    f"Hata: {e}"
                ) from e

            self.get_logger().info("Rebuilt KalmanNetNN and loaded state_dict.")
            return model

        # 3) Raw state_dict directly (rare)
        if isinstance(obj, dict) and _looks_like_kalmannet_state_dict(_strip_module_prefix(obj)):
            sd = _strip_module_prefix(obj)
            model = _build_kalmannet_from_state_dict(
                sd,
                self.sensors_xy,
                dt=self.dt,
                device=self.device,
                kalmannet_nn_file=self.kalmannet_nn_file,
            )
            model.load_state_dict(sd, strict=True)
            self.get_logger().info("Loaded raw state_dict and rebuilt KalmanNetNN.")
            return model

        raise RuntimeError(
            f"Tanınmayan checkpoint formatı: {type(obj)} "
            f"(keys={list(obj.keys()) if isinstance(obj, dict) else 'n/a'})"
        )

    # ---------------- Callbacks / logic ----------------
    def _on_gt(self, msg: Odometry):
        self._last_gt = msg
        self._last_gt_wall = time.monotonic()

    def _reset_filter(self, x0: torch.Tensor) -> None:
        # KalmanNetNN specific init
        if hasattr(self.model, "batch_size"):
            self.model.batch_size = 1  # type: ignore[attr-defined]
        if hasattr(self.model, "init_hidden_KNet"):
            self.model.init_hidden_KNet()  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Model init_hidden_KNet() metoduna sahip değil.")

        x0_b = x0.view(1, -1, 1).to(self.device)

        if hasattr(self.model, "InitSequence"):
            self.model.InitSequence(x0_b, int(self.max_seq_len))  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Model InitSequence() metoduna sahip değil.")

        self._initialized = True
        self._step = 0
        self._prev_xy = None
        self.get_logger().info(f"[RESET] x0={x0.detach().cpu().numpy().tolist()}")

    def _build_init(self, z_np: np.ndarray) -> torch.Tensor:
        """
        Returns x0 tensor of shape (m,)
        """
        m = int(self.state_dim)

        if self.init_from_gt:
            if self._last_gt is None:
                raise RuntimeError("init_from_gt=true ama henüz gt mesajı gelmedi")
            gx = float(self._last_gt.pose.pose.position.x)
            gy = float(self._last_gt.pose.pose.position.y)

            if m >= 4:
                gvx = float(self._last_gt.twist.twist.linear.x)
                gvy = float(self._last_gt.twist.twist.linear.y)
                base = [gx, gy, gvx, gvy]
            else:
                base = [gx, gy]

            # pad to m
            base = base + [0.0] * max(0, m - len(base))
            return torch.tensor(base, dtype=torch.float32, device=self.device)

        # No GT: LS init from ranges
        xy = ls_init_xy_from_ranges(self.sensors_xy_np, z_np, min_range=self.min_range)
        if m >= 4:
            base = [float(xy[0]), float(xy[1]), 0.0, 0.0]
        else:
            base = [float(xy[0]), float(xy[1])]
        base = base + [0.0] * max(0, m - len(base))
        return torch.tensor(base, dtype=torch.float32, device=self.device)

    def _on_z(self, msg: Float32MultiArray):
        self._last_z_wall = time.monotonic()

        data = np.asarray(msg.data, dtype=np.float32).reshape((-1,))
        if data.size != self.obs_dim:
            self.get_logger().warn(f"z dim mismatch: got={data.size} expected={self.obs_dim}")
            return
        if not np.all(np.isfinite(data)):
            self.get_logger().warn("z contains NaN/inf, skipping")
            return

        # clamp to min_range (avoid negative ranges)
        if self.min_range > 0.0:
            data = np.maximum(data, self.min_range)

        z = torch.tensor(data, dtype=torch.float32, device=self.device)

        # init
        if not self._initialized:
            try:
                x0 = self._build_init(data)
            except Exception as e:
                # Throttle this (otherwise it spams every Z)
                now = time.monotonic()
                if now - self._last_gt_wait_log_wall > 2.0:
                    self._last_gt_wait_log_wall = now
                    self.get_logger().warn(f"init bekleniyor: {e}")
                return
            self._reset_filter(x0)

        # forward step
        y = z.view(1, self.obs_dim, 1)  # (B,n,1)
        with torch.inference_mode():
            xhat_b = self.model(y)          # expected: (1,m,1)
            xhat = xhat_b.squeeze(-1).squeeze(0)  # (m,)

        # gating: measurement consistency
        if self.gate_resid_rms > 0.0 and xhat.numel() >= 2:
            pred = predict_ranges_xy(xhat[0:2], self.sensors_xy, eps=self.eps_range, min_range=self.min_range)
            resid_rms = torch.sqrt(torch.mean((pred - z) ** 2)).item()
            if math.isfinite(resid_rms) and resid_rms > self.gate_resid_rms:
                self.get_logger().warn(f"[GATE] resid_rms={resid_rms:.3f} > {self.gate_resid_rms:.3f} -> reset")
                try:
                    x0 = self._build_init(data)
                    self._reset_filter(x0)
                except Exception as e:
                    self.get_logger().warn(f"reset init failed: {e}")
                return

        # publish odom
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame

        od.pose.pose.position.x = float(xhat[0].item()) if xhat.numel() > 0 else 0.0
        od.pose.pose.position.y = float(xhat[1].item()) if xhat.numel() > 1 else 0.0
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.w = 1.0

        vx = vy = 0.0
        if self.state_dim >= 4 and xhat.numel() >= 4:
            vx = float(xhat[2].item())
            vy = float(xhat[3].item())
        elif self.state_dim >= 2 and self.dt > 1e-6 and xhat.numel() >= 2:
            # finite difference velocity
            x = float(xhat[0].item())
            yv = float(xhat[1].item())
            if self._prev_xy is not None:
                x0, y0 = self._prev_xy
                vx = (x - x0) / self.dt
                vy = (yv - y0) / self.dt
            self._prev_xy = (x, yv)

        od.twist.twist.linear.x = float(vx)
        od.twist.twist.linear.y = float(vy)

        self.pub.publish(od)

        # safety reset
        self._step += 1
        if self._step >= self.max_seq_len - 1:
            self.get_logger().warn("[RESET] max_seq_len reached -> resetting with last state")
            self._reset_filter(xhat.detach())

    def stop(self):
        """Stop background thread safely."""
        self._stop_evt.set()
        try:
            if self._status_thread.is_alive():
                self._status_thread.join(timeout=1.0)
        except Exception:
            pass


def main():
    rclpy.init()
    node: Optional[KNetTracker] = None
    try:
        node = KNetTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.stop()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
        # Avoid "rcl_shutdown already called" on Jazzy when SIGINT already shutdowns the context
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
