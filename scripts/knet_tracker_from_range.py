#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


# ---------------- Path helper (so we can import KalmanNet_nn.py from repo root) ----------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from KalmanNet_nn import KalmanNetNN  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "KalmanNetNN import edilemedi. Repo root içinde 'KalmanNet_nn.py' olmalı. "
        "Kodu repo root'tan çalıştırdığından emin ol (python3 scripts/...). "
        f"Orijinal hata: {e}"
    )


def load_layout_csv(path: str) -> np.ndarray:
    """
    CSV: her satır 'x,y' veya 'x y' olabilir. '#' ile başlayanlar comment.
    returns: (N,2) float32
    """
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
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
    if not pts:
        raise RuntimeError(f"layout parse edilemedi: {path}")
    return np.asarray(pts, dtype=np.float32)


def predict_ranges_xy(
    xy: torch.Tensor,
    sensors_xy: torch.Tensor,
    *,
    eps: float,
    min_range: float,
) -> torch.Tensor:
    """
    xy: (2,)
    sensors_xy: (N,2)
    returns: (N,) predicted ranges
    """
    dx = xy[0] - sensors_xy[:, 0]
    dy = xy[1] - sensors_xy[:, 1]
    d = torch.sqrt(dx * dx + dy * dy + eps)
    return torch.clamp(d, min=min_range)


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
    returns: (2,) [x,y]
    """
    sensors_xy = np.asarray(sensors_xy, dtype=float).reshape((-1, 2))
    z = np.asarray(z, dtype=float).reshape((-1,))
    N = sensors_xy.shape[0]
    if z.shape[0] != N:
        raise ValueError(f"z length {z.shape[0]} != N {N}")
    if N < 3:
        raise ValueError(f"Need at least 3 sensors, got N={N}")
    z = np.maximum(z, float(min_range))

    # choose reference sensor = smallest range
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


def _strip_module_prefix(state_dict: dict) -> dict:
    # DataParallel ile kaydedildiyse "module." prefix'i olur
    out = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
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


def _infer_kalmannet_hparams(sd: dict) -> tuple[int, int, int, int]:
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
        raise RuntimeError(f"FC7.0.weight in_features beklenen 2*n olmalı, geldi: {n2}")
    n = int(n2 // 2)

    d_in = int(w2.shape[1])  # should be m^2 + n^2
    if d_in <= 0:
        raise RuntimeError("FC2.0.weight in_features invalid")
    out_mult = int(w2.shape[0] // d_in)

    return m, n, in_mult, out_mult


def _build_kalmannet_from_state_dict(
    sd: dict,
    sensors_xy: torch.Tensor,
    *,
    dt: float,
    device: torch.device,
) -> KalmanNetNN:
    """
    Build KalmanNetNN with the SAME architecture implied by state_dict shapes,
    then caller can load_state_dict.
    """
    m, n, in_mult, out_mult = _infer_kalmannet_hparams(sd)

    # Quick sanity: layout sensor count must match n
    if int(sensors_xy.shape[0]) != n:
        raise RuntimeError(
            f"Model n={n} ama layout N_sensors={int(sensors_xy.shape[0])}. "
            "Yanlış layout_file seçmiş olabilirsin."
        )

    # --- f(x): constant velocity for first 4 states, otherwise identity ---
    dt = float(dt)

    def f(x: torch.Tensor) -> torch.Tensor:
        # x: (B,m,1)
        if m >= 4:
            B = x.shape[0]
            F = torch.eye(m, device=x.device, dtype=x.dtype)
            F0 = torch.tensor(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=x.device,
                dtype=x.dtype,
            )
            F[:4, :4] = F0
            F = F.view(1, m, m).expand(B, -1, -1)
            return torch.bmm(F, x)
        return x

    # --- h(x): ranges to sensors using first 2 states as (x,y) ---
    eps = 1e-6

    def h(x: torch.Tensor) -> torch.Tensor:
        # x: (B,m,1) -> (B,n,1)
        B = x.shape[0]
        px = x[:, 0, 0].view(B, 1)
        py = x[:, 1, 0].view(B, 1)
        dx = px - sensors_xy[:, 0].view(1, n)
        dy = py - sensors_xy[:, 1].view(1, n)
        d = torch.sqrt(dx * dx + dy * dy + eps)
        return d.view(B, n, 1)

    # Minimal SysModel container
    class _Sys:
        pass

    sysm = _Sys()
    sysm.f = f
    sysm.h = h
    sysm.m = m
    sysm.n = n
    sysm.prior_Q = torch.eye(m, device=device)
    sysm.prior_Sigma = torch.zeros(m, m, device=device)
    sysm.prior_S = torch.eye(n, device=device)

    args = SimpleNamespace(
        use_cuda=(device.type == "cuda"),
        n_batch=1,  # online inference: batch=1
        in_mult_KNet=in_mult,
        out_mult_KNet=out_mult,
    )

    model = KalmanNetNN()
    model.NNBuild(sysm, args)
    return model


class KNetTracker(Node):
    def __init__(self):
        super().__init__("knet_tracker")

        # ---------------- Params ----------------
        # core
        self.declare_parameter("model_path", "")
        self.declare_parameter("layout_file", "")
        self.declare_parameter("z_topic", "z")
        self.declare_parameter("est_topic", "knet/estimated")

        # init
        self.declare_parameter("init_from_gt", True)
        self.declare_parameter("gt_topic", "gt/odom")

        # frames
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("child_frame", "base_link")

        # runtime
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("delta", 0.1)  # dt (s) for f(x) and optional vx/vy estimate
        self.declare_parameter("max_seq_len", 10000)

        # range model params (gating + init)
        self.declare_parameter("min_range", 1e-3)
        self.declare_parameter("eps_range", 1e-6)

        # gating
        self.declare_parameter("gate_resid_rms", 0.30)

        # (optional) declare sim time to be friendly with CLI overrides
        self.declare_parameter("use_sim_time", False)

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

        self.device = torch.device("cuda") if (use_cuda_req and torch.cuda.is_available()) else torch.device("cpu")
        self.get_logger().info(f"device={self.device}")

        if not self.model_path:
            raise RuntimeError("model_path param boş. Örn: -p model_path:=models/.../best-model.pt")
        if not self.layout_file:
            raise RuntimeError("layout_file param boş. Örn: -p layout_file:=config/paper_sensors_5x5_b20.csv")

        if not Path(self.model_path).is_file():
            raise RuntimeError(f"model_path bulunamadı: {self.model_path}")
        if not Path(self.layout_file).is_file():
            raise RuntimeError(f"layout_file bulunamadı: {self.layout_file}")

        # ---------------- Load layout ----------------
        sensors = load_layout_csv(self.layout_file)  # (N,2)
        self.sensors_xy_np = sensors
        self.sensors_xy = torch.tensor(sensors, dtype=torch.float32, device=self.device)
        self.obs_dim = int(self.sensors_xy.shape[0])
        self.get_logger().info(f"layout loaded: N_sensors={self.obs_dim}")

        # ---------------- Load model ----------------
        self.model: torch.nn.Module = self._load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Infer state dim (m) for publishing / init vector
        if not hasattr(self.model, "m"):
            # KalmanNetNN sets self.m in InitSystemDynamics
            self.state_dim = 4
        else:
            self.state_dim = int(getattr(self.model, "m"))
        self.get_logger().info(f"model loaded. state_dim(m)={self.state_dim}, obs_dim(n)={self.obs_dim}")

        # internal state
        self._initialized = False
        self._step = 0
        self._last_gt: Optional[Odometry] = None

        # for m==2 velocity estimation
        self._prev_xy: Optional[tuple[float, float]] = None

        # pubs/subs
        self.pub = self.create_publisher(Odometry, self.est_topic, 10)
        self.create_subscription(Float32MultiArray, self.z_topic, self._on_z, qos_profile_sensor_data)

        if self.init_from_gt:
            self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
            self.get_logger().info(f"init_from_gt=true -> subscribing gt_topic={self.gt_topic}")

        self.get_logger().info(f"sub z_topic={self.z_topic} pub est_topic={self.est_topic}")

    # ---------------- Model loading ----------------
    def _torch_load_any(self, path: str):
        # PyTorch >=2.0: weights_only arg may exist; use weights_only=False to allow full objects.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_model(self, path: str) -> torch.nn.Module:
        obj = self._torch_load_any(path)

        # 1) Direct full model
        if isinstance(obj, torch.nn.Module):
            self.get_logger().info("Loaded full torch.nn.Module from checkpoint.")
            return obj

        # 2) Checkpoint dict with state_dict
        if isinstance(obj, dict) and ("state_dict" in obj):
            sd = obj["state_dict"]
            if not isinstance(sd, dict):
                raise RuntimeError("checkpoint['state_dict'] dict değil.")

            sd = _strip_module_prefix(sd)

            # If this looks like KalmanNetNN, rebuild and load
            if _looks_like_kalmannet_state_dict(sd):
                self.get_logger().info("Checkpoint is a state_dict (KalmanNetNN-like). Rebuilding model...")
                model = _build_kalmannet_from_state_dict(
                    sd,
                    self.sensors_xy,
                    dt=self.dt,
                    device=self.device,
                )
                model.load_state_dict(sd, strict=True)
                return model

            # Unknown architecture
            example_keys = list(sd.keys())[:30]
            raise RuntimeError(
                "Checkpoint sadece state_dict içeriyor ama mimariyi tanıyamadım.\n"
                "state_dict ilk anahtarlar: "
                f"{example_keys}\n"
                "Bu durumda: ya full model kaydet (torch.save(model,...)), "
                "ya da burada mimariyi yeniden kuracak kodu eklemelisin."
            )

        # 3) Raw state_dict directly
        if isinstance(obj, dict) and _looks_like_kalmannet_state_dict(_strip_module_prefix(obj)):
            sd = _strip_module_prefix(obj)
            self.get_logger().info("Checkpoint raw state_dict gibi görünüyor. Rebuilding KalmanNetNN...")
            model = _build_kalmannet_from_state_dict(sd, self.sensors_xy, dt=self.dt, device=self.device)
            model.load_state_dict(sd, strict=True)
            return model

        raise RuntimeError(
            f"Tanınmayan checkpoint formatı: {type(obj)} "
            f"(keys={list(obj.keys()) if isinstance(obj, dict) else 'n/a'})"
        )

    # ---------------- Callbacks / logic ----------------
    def _on_gt(self, msg: Odometry):
        self._last_gt = msg

    def _reset_filter(self, x0: torch.Tensor):
        """
        x0: (m,) initial state vector
        """
        # KalmanNetNN specific init
        if hasattr(self.model, "batch_size"):
            self.model.batch_size = 1  # type: ignore
        if hasattr(self.model, "init_hidden_KNet"):
            self.model.init_hidden_KNet()  # type: ignore

        x0_b = x0.view(1, -1, 1).to(self.device)

        if hasattr(self.model, "InitSequence"):
            self.model.InitSequence(x0_b, int(self.max_seq_len))  # type: ignore
        else:
            raise RuntimeError("Model InitSequence() methoduna sahip değil. Bu script KalmanNetNN için yazıldı.")

        self._initialized = True
        self._step = 0
        self._prev_xy = None
        self.get_logger().info(f"[RESET] x0={x0.detach().cpu().numpy().tolist()}")

    def _build_init(self, z_np: np.ndarray) -> torch.Tensor:
        """
        Returns x0 tensor of shape (m,)
        """
        m = self.state_dim

        if self.init_from_gt:
            if self._last_gt is None:
                raise RuntimeError("init_from_gt=true ama henüz gt mesajı gelmedi.")
            gx = float(self._last_gt.pose.pose.position.x)
            gy = float(self._last_gt.pose.pose.position.y)

            if m >= 4:
                gvx = float(self._last_gt.twist.twist.linear.x)
                gvy = float(self._last_gt.twist.twist.linear.y)
                return torch.tensor([gx, gy, gvx, gvy], dtype=torch.float32, device=self.device)

            # m==2
            return torch.tensor([gx, gy], dtype=torch.float32, device=self.device)

        # No GT: initialize from ranges (LS) + v=0 if needed
        xy = ls_init_xy_from_ranges(self.sensors_xy_np, z_np, min_range=self.min_range)
        if m >= 4:
            return torch.tensor([float(xy[0]), float(xy[1]), 0.0, 0.0], dtype=torch.float32, device=self.device)
        return torch.tensor([float(xy[0]), float(xy[1])], dtype=torch.float32, device=self.device)

    def _on_z(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32).reshape((-1,))
        if data.size != self.obs_dim:
            self.get_logger().warn(f"z dim mismatch: got={data.size} expected={self.obs_dim}")
            return

        # clamp ranges (avoid negative/zero)
        if self.min_range > 0.0:
            data = np.maximum(data, self.min_range)

        z = torch.tensor(data, dtype=torch.float32, device=self.device)  # (N,)

        # init
        if not self._initialized:
            try:
                x0 = self._build_init(data)
            except Exception as e:
                self.get_logger().warn(f"init bekleniyor: {repr(e)}")
                return
            self._reset_filter(x0)

        # forward step (KalmanNet expects (B,n,1))
        y = z.view(1, self.obs_dim, 1)
        with torch.inference_mode():
            xhat_b = self.model(y)  # expected: (1,m,1)
            xhat = xhat_b.squeeze(-1).squeeze(0)  # (m,)

        # gating: measurement consistency (optional)
        if self.gate_resid_rms > 0.0:
            pred = predict_ranges_xy(xhat[0:2], self.sensors_xy, eps=self.eps_range, min_range=self.min_range)
            resid_rms = torch.sqrt(torch.mean((pred - z) ** 2)).item()
            if math.isfinite(resid_rms) and resid_rms > self.gate_resid_rms:
                self.get_logger().warn(f"[GATE] resid_rms={resid_rms:.3f} > {self.gate_resid_rms:.3f} -> reset")
                try:
                    x0 = self._build_init(data)
                    self._reset_filter(x0)
                except Exception as e:
                    self.get_logger().warn(f"reset init failed: {repr(e)}")
                return

        # publish odom
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame

        # pose
        od.pose.pose.position.x = float(xhat[0].item())
        od.pose.pose.position.y = float(xhat[1].item())
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.w = 1.0

        # twist
        vx = vy = 0.0
        if self.state_dim >= 4 and xhat.numel() >= 4:
            vx = float(xhat[2].item())
            vy = float(xhat[3].item())
        elif self.state_dim == 2 and self.dt > 1e-6:
            # estimate v by finite difference on positions
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

        # step / safety reset
        self._step += 1
        if self._step >= self.max_seq_len - 1:
            self.get_logger().warn("[RESET] max_seq_len reached -> resetting filter with last state")
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
