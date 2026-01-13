#!/usr/bin/env python3
from __future__ import annotations

"""knet_tracker_from_range.py

ROS2 node: KalmanNet (KalmanNetNN) tracker for 2D range-only measurements.

Supports checkpoints saved either as:
  - full torch.nn.Module (torch.save(model, ...))
  - dict with a 'state_dict' key (common training checkpoint)

Your checkpoint (best-model.pt) is a dict with keys: state_dict, step, cv_mse, cv_db,
so this node rebuilds KalmanNetNN architecture from the state_dict layer shapes
and loads weights.

Sub:
  - z_topic (std_msgs/Float32MultiArray) : N ranges
  - (optional) gt_topic (nav_msgs/Odometry) if init_from_gt=true

Pub:
  - est_topic (nav_msgs/Odometry) : [x,y,(vx,vy)]

Notes:
  - ONLINE tracker: batch_size=1
  - Resets KalmanNet hidden state on init and when gating triggers.
"""

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


# ------------------------ Layout helpers ------------------------


def load_layout_csv(path: str) -> np.ndarray:
    """CSV: each line 'x,y' or 'x y' (comments with '#'). Returns (N,2) float32."""
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
            pts.append((float(parts[0]), float(parts[1])))
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
    """xy:(2,) sensors_xy:(N,2) -> (N,)"""
    dx = xy[0] - sensors_xy[:, 0]
    dy = xy[1] - sensors_xy[:, 1]
    d = torch.sqrt(dx * dx + dy * dy + eps)
    return torch.clamp(d, min=float(min_range))


def ls_init_xy_from_ranges(
    sensors_xy: np.ndarray,
    z: np.ndarray,
    *,
    min_range: float = 1e-3,
) -> np.ndarray:
    """Fast linear LS multilateration init in 2D. Returns (2,) float32."""
    sensors_xy = np.asarray(sensors_xy, dtype=float).reshape((-1, 2))
    z = np.asarray(z, dtype=float).reshape((-1,))
    N = sensors_xy.shape[0]
    if z.shape[0] != N:
        raise ValueError(f"z length {z.shape[0]} != N {N}")
    if N < 3:
        raise ValueError(f"Need at least 3 sensors, got N={N}")

    z = np.maximum(z, float(min_range))

    # reference = smallest range
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


# ------------------------ KalmanNet import (robust) ------------------------


def _find_kalmannet_nn_file() -> Optional[Path]:
    """Try to locate KalmanNet_nn.py by walking upwards from this script."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        # 1) <root>/KalmanNet_nn.py
        cand = p / "KalmanNet_nn.py"
        if cand.is_file():
            return cand

        # 2) <root>/src/<...>/KalmanNet_nn.py
        src = p / "src"
        if src.is_dir():
            cand2 = src / "KalmanNet_nn.py"
            if cand2.is_file():
                return cand2
            for cand3 in src.glob("*/KalmanNet_nn.py"):
                if cand3.is_file():
                    return cand3
    return None


def _import_kalmannet_nn(kalmannet_nn_file: Optional[str] = None):
    """Return KalmanNetNN class loaded from KalmanNet_nn.py (by file path)."""
    path: Optional[Path] = None

    if kalmannet_nn_file and str(kalmannet_nn_file).strip():
        p = Path(str(kalmannet_nn_file)).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"kalmannet_nn_file bulunamadı: {p}")
        path = p
    else:
        path = _find_kalmannet_nn_file()

    if path is None:
        raise ImportError(
            "KalmanNet_nn.py bulunamadı.\n"
            "Çözüm: KalmanNet_nn.py'nin olduğu yeri doğrula veya\n"
            "-p kalmannet_nn_file:=/abs/path/to/KalmanNet_nn.py parametresi ver."
        )

    spec = importlib.util.spec_from_file_location("KalmanNet_nn", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"KalmanNet_nn import spec oluşturulamadı: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "KalmanNetNN"):
        raise ImportError(f"{path} içinde KalmanNetNN class bulunamadı")

    return module.KalmanNetNN


# ------------------------ Checkpoint helpers ------------------------


def _torch_load_any(path: str):
    """torch.load wrapper that works across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_module_prefix(sd: dict) -> dict:
    """If model saved under DataParallel, keys start with 'module.'"""
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
    """Infer (m, n, in_mult, out_mult) from KalmanNetNN state_dict layer shapes."""
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
    out_mult = int(w2.shape[0] // din)

    if din != (m * m + n * n):
        raise RuntimeError(
            f"state_dict inconsistent: FC2.0.weight in_features={din} != m^2+n^2={m*m+n*n}"
        )

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
    """Rebuild KalmanNetNN (architecture) and return model (weights not loaded yet)."""
    KalmanNetNN = _import_kalmannet_nn(kalmannet_nn_file)  # dynamic import

    m, n, in_mult, out_mult = _infer_kalmannet_hparams(sd)

    if sensors_xy.shape[0] != n:
        raise RuntimeError(
            f"Model expects n={n} sensors but layout has N={int(sensors_xy.shape[0])}. "
            "layout_file yanlış olabilir."
        )

    dt = float(dt)

    def f(x: torch.Tensor) -> torch.Tensor:
        # x: (B,m,1)
        B = int(x.shape[0])
        F = torch.eye(m, device=x.device, dtype=x.dtype)
        if m >= 4:
            F[0, 2] = dt
            F[1, 3] = dt
        F = F.unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(F, x)

    eps = 1e-6

    def h(x: torch.Tensor) -> torch.Tensor:
        # x: (B,m,1) -> (B,n,1)
        if m < 2:
            raise RuntimeError("State dimension m<2, cannot compute ranges")
        B = int(x.shape[0])
        px = x[:, 0, 0].view(B, 1)
        py = x[:, 1, 0].view(B, 1)
        dx = px - sensors_xy[:, 0].view(1, n)
        dy = py - sensors_xy[:, 1].view(1, n)
        d = torch.sqrt(dx * dx + dy * dy + eps)
        return d.view(B, n, 1)

    prior_Q = torch.eye(m, device=device)
    prior_Sigma = torch.eye(m, device=device)
    prior_S = torch.eye(n, device=device)

    sysm = _SysModel(f=f, h=h, m=m, n=n, prior_Q=prior_Q, prior_Sigma=prior_Sigma, prior_S=prior_S)

    args = SimpleNamespace(
        use_cuda=(device.type == "cuda"),
        n_batch=1,
        in_mult_KNet=in_mult,
        out_mult_KNet=out_mult,
    )

    model = KalmanNetNN()
    model.NNBuild(sysm, args)
    return model


# ------------------------ ROS2 Node ------------------------


class KNetTracker(Node):
    def __init__(self):
        super().__init__("knet_tracker")

        # --- Declare params (so CLI -p works) ---
        self.declare_parameter("use_sim_time", False)

        self.declare_parameter("model_path", "")
        self.declare_parameter("layout_file", "")

        self.declare_parameter("z_topic", "z")
        self.declare_parameter("est_topic", "knet/estimated")

        self.declare_parameter("init_from_gt", True)
        self.declare_parameter("gt_topic", "gt/odom")

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("child_frame", "base_link")

        self.declare_parameter("use_cuda", False)
        self.declare_parameter("delta", 0.1)  # dt

        self.declare_parameter("max_seq_len", 10000)

        self.declare_parameter("min_range", 1e-3)
        self.declare_parameter("eps_range", 1e-6)

        self.declare_parameter("gate_resid_rms", 0.30)

        # optional explicit KalmanNet_nn.py location
        self.declare_parameter("kalmannet_nn_file", "")

        # --- Read params ---
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

        # --- Device ---
        self.device = torch.device("cuda") if (use_cuda_req and torch.cuda.is_available()) else torch.device("cpu")
        self.get_logger().info(f"device={self.device}")

        if not self.model_path:
            raise RuntimeError("model_path param boş. Örn: -p model_path:=models/.../best-model.pt")
        if not self.layout_file:
            raise RuntimeError("layout_file param boş. Örn: -p layout_file:=config/paper_sensors_5x5_b20.csv")
        if self.dt <= 0.0:
            raise RuntimeError("delta(dt) > 0 olmalı")

        mp = Path(self.model_path)
        lf = Path(self.layout_file)
        if not mp.is_file():
            raise RuntimeError(f"model_path bulunamadı: {mp}")
        if not lf.is_file():
            raise RuntimeError(f"layout_file bulunamadı: {lf}")

        # --- Load layout ---
        sensors = load_layout_csv(self.layout_file)
        self.sensors_xy_np = sensors
        self.sensors_xy = torch.tensor(sensors, dtype=torch.float32, device=self.device)
        self.obs_dim = int(self.sensors_xy.shape[0])
        self.get_logger().info(f"layout loaded: N_sensors={self.obs_dim}")

        # --- Load model ---
        self.model = self._load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.state_dim = int(getattr(self.model, "m", 4))
        self.get_logger().info(f"model ready: state_dim(m)={self.state_dim}, obs_dim(n)={self.obs_dim}")

        # --- Internal state ---
        self._initialized = False
        self._step = 0
        self._last_gt: Optional[Odometry] = None
        self._prev_xy: Optional[Tuple[float, float]] = None

        # --- ROS I/O ---
        self.pub = self.create_publisher(Odometry, self.est_topic, 10)
        self.create_subscription(Float32MultiArray, self.z_topic, self._on_z, qos_profile_sensor_data)

        if self.init_from_gt:
            self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
            self.get_logger().info(f"init_from_gt=true -> subscribing gt_topic={self.gt_topic}")

        self.get_logger().info(f"sub z_topic={self.z_topic} pub est_topic={self.est_topic}")

    def _load_model(self, path: str) -> torch.nn.Module:
        obj = _torch_load_any(path)

        if isinstance(obj, torch.nn.Module):
            self.get_logger().info("Loaded full torch.nn.Module from checkpoint.")
            return obj

        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
            if not isinstance(sd, dict):
                raise RuntimeError("checkpoint['state_dict'] dict değil")
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
                    "Bu genelde şu demek: eğitimde farklı m/n veya farklı KalmanNetNN versiyonu kullanılmış.\n"
                    f"Hata: {e}"
                ) from e

            self.get_logger().info("Rebuilt KalmanNetNN and loaded state_dict.")
            return model

        raise RuntimeError(
            f"Tanınmayan checkpoint formatı: {type(obj)} "
            f"(keys={list(obj.keys()) if isinstance(obj, dict) else 'n/a'})"
        )

    def _on_gt(self, msg: Odometry):
        self._last_gt = msg

    def _reset_filter(self, x0: torch.Tensor) -> None:
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
        m = self.state_dim

        if self.init_from_gt:
            if self._last_gt is None:
                raise RuntimeError("init_from_gt=true ama henüz gt mesajı gelmedi")
            gx = float(self._last_gt.pose.pose.position.x)
            gy = float(self._last_gt.pose.pose.position.y)
            if m >= 4:
                gvx = float(self._last_gt.twist.twist.linear.x)
                gvy = float(self._last_gt.twist.twist.linear.y)
                return torch.tensor(
                    [gx, gy, gvx, gvy] + [0.0] * max(0, m - 4),
                    dtype=torch.float32,
                    device=self.device,
                )
            return torch.tensor([gx, gy] + [0.0] * max(0, m - 2), dtype=torch.float32, device=self.device)

        xy = ls_init_xy_from_ranges(self.sensors_xy_np, z_np, min_range=self.min_range)
        if m >= 4:
            base = [float(xy[0]), float(xy[1]), 0.0, 0.0] + [0.0] * max(0, m - 4)
            return torch.tensor(base, dtype=torch.float32, device=self.device)
        base = [float(xy[0]), float(xy[1])] + [0.0] * max(0, m - 2)
        return torch.tensor(base, dtype=torch.float32, device=self.device)

    def _on_z(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32).reshape((-1,))
        if data.size != self.obs_dim:
            self.get_logger().warn(f"z dim mismatch: got={data.size} expected={self.obs_dim}")
            return
        if not np.all(np.isfinite(data)):
            self.get_logger().warn("z contains NaN/inf, skipping")
            return

        if self.min_range > 0.0:
            data = np.maximum(data, self.min_range)

        z = torch.tensor(data, dtype=torch.float32, device=self.device)

        if not self._initialized:
            try:
                x0 = self._build_init(data)
            except Exception as e:
                self.get_logger().warn(f"init bekleniyor: {repr(e)}")
                return
            self._reset_filter(x0)

        y = z.view(1, self.obs_dim, 1)
        with torch.inference_mode():
            xhat_b = self.model(y)
            xhat = xhat_b.squeeze(-1).squeeze(0)

        if self.gate_resid_rms > 0.0 and xhat.numel() >= 2:
            pred = predict_ranges_xy(xhat[0:2], self.sensors_xy, eps=self.eps_range, min_range=self.min_range)
            resid_rms = torch.sqrt(torch.mean((pred - z) ** 2)).item()
            if math.isfinite(resid_rms) and resid_rms > self.gate_resid_rms:
                self.get_logger().warn(
                    f"[GATE] resid_rms={resid_rms:.3f} > {self.gate_resid_rms:.3f} -> reset"
                )
                try:
                    x0 = self._build_init(data)
                    self._reset_filter(x0)
                except Exception as e:
                    self.get_logger().warn(f"reset init failed: {repr(e)}")
                return

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

        self._step += 1
        if self._step >= self.max_seq_len - 1:
            self.get_logger().warn("[RESET] max_seq_len reached -> resetting with last state")
            self._reset_filter(xhat.detach())


def main():
    rclpy.init()
    node = KNetTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
