#!/usr/bin/env python3
from __future__ import annotations

"""
knet_tracker_from_range.py (ROS2 / rclpy)

KalmanNetNN online tracker for 2D range-only measurements.

Sub:
  - z_topic  (std_msgs/Float32MultiArray) : N ranges
  - gt_topic (nav_msgs/Odometry)          : only if init_from_gt=true

Pub:
  - est_topic (nav_msgs/Odometry)         : [x,y,(vx,vy)]

Key points:
  - Loads checkpoint either as:
      (1) full torch.nn.Module
      (2) dict with 'state_dict'
      (3) raw state_dict
  - Imports KalmanNetNN from KalmanNet_TSP by default:
      <repo>/third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py
    If you pass kalmannet_nn_file and it is wrong, we DO NOT crash:
    we warn and fallback to auto-discovery.

  - Uses dt (delta), tau and sigma params to match training assumptions as much as possible.
"""

import importlib.util
import math
import sys
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


# ----------------------------- small utilities -----------------------------

def torch_load_any(path: str):
    """torch.load wrapper across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_module_prefix(sd: dict) -> dict:
    """Only strip DataParallel 'module.' prefix."""
    out = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def load_layout_csv(path: str) -> np.ndarray:
    """
    CSV lines: 'x,y' or 'x y'. Allows comments with '#'.
    Returns (N,2) float32.
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
        raise RuntimeError(f"layout empty/unreadable: {path}")
    return np.asarray(pts, dtype=np.float32)


def ls_init_xy_from_ranges(sensors_xy: np.ndarray, z: np.ndarray, *, min_range: float) -> np.ndarray:
    """
    Linear LS multilateration init in 2D.
    sensors_xy: (N,2)
    z: (N,)
    returns (2,)
    """
    sensors_xy = np.asarray(sensors_xy, dtype=float).reshape((-1, 2))
    z = np.asarray(z, dtype=float).reshape((-1,))
    N = sensors_xy.shape[0]
    if z.shape[0] != N:
        raise ValueError(f"z length {z.shape[0]} != N {N}")
    if N < 3:
        raise ValueError("Need >=3 sensors")

    z = np.maximum(z, float(min_range))
    i0 = int(np.argmin(z))
    x0, y0 = sensors_xy[i0]
    z0 = float(z[i0])

    A, b = [], []
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
        raise ValueError("Degenerate sensor geometry (rank<2)")

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol.astype(np.float32)


def infer_hparams_from_state_dict(sd: dict) -> Tuple[int, int, int, int]:
    """
    Infer (m, n, in_mult, out_mult) from KalmanNetNN state_dict.
    Assumes KalmanNet_TSP/KNet/KalmanNet_nn.py architecture (#2 in paper).
    """
    # FC5: (m*in_mult, m)
    w5 = sd["FC5.0.weight"]
    # FC7: (2*n*in_mult, 2*n)
    w7 = sd["FC7.0.weight"]
    # FC2: ((m^2+n^2)*out_mult, (m^2+n^2))
    w2 = sd["FC2.0.weight"]

    m = int(w5.shape[1])
    in_mult = int(w5.shape[0] // max(m, 1))

    n2 = int(w7.shape[1])
    if n2 % 2 != 0:
        raise RuntimeError(f"FC7 in_features expected 2*n, got {n2}")
    n = int(n2 // 2)

    din = int(w2.shape[1])
    if din != (m * m + n * n):
        raise RuntimeError(f"FC2 in_features={din} != m^2+n^2={m*m+n*n}")

    out_mult = int(w2.shape[0] // din)
    return m, n, in_mult, out_mult


def find_repo_root(start: Path) -> Path:
    """
    Walk upwards until we find third_party/KalmanNet_TSP.
    """
    for p in [start] + list(start.parents):
        if (p / "third_party" / "KalmanNet_TSP").is_dir():
            return p
    return start


def resolve_kalmannet_nn_path(explicit: str) -> Path:
    """
    Resolve KalmanNet_nn.py path.
    Priority:
      1) explicit param (file or directory) if valid
      2) <repo>/third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py
      3) <repo>/KalmanNet_nn.py  (if user copied it)
    If explicit is provided but wrong, we WARN and fallback (do NOT crash).
    """
    here = Path(__file__).resolve()
    repo = find_repo_root(here.parent)

    candidates = []

    exp = explicit.strip()
    if exp:
        p = Path(exp).expanduser().resolve()
        if p.is_file():
            candidates.append(p)
        elif p.is_dir():
            # allow passing ".../KNet" or ".../KalmanNet_TSP" etc.
            candidates.append(p / "KalmanNet_nn.py")
            candidates.append(p / "KNet" / "KalmanNet_nn.py")
        else:
            # keep going, fallback
            pass

    # prefer the third_party version (most likely the one used for training)
    candidates.append(repo / "third_party" / "KalmanNet_TSP" / "KNet" / "KalmanNet_nn.py")
    candidates.append(repo / "KalmanNet_nn.py")

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "KalmanNet_nn.py not found. Tried:\n" + "\n".join([str(c) for c in candidates])
    )


def import_kalmannet_nn(explicit: str, logger: Optional[Node] = None):
    """
    Dynamic import KalmanNetNN class from KalmanNet_nn.py file.
    """
    try:
        path = resolve_kalmannet_nn_path(explicit)
    except Exception as e:
        raise ImportError(str(e)) from e

    if logger is not None:
        logger.get_logger().info(f"Using KalmanNet_nn.py: {path}")

    spec = importlib.util.spec_from_file_location("knet_kalmannet_nn", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "KalmanNetNN"):
        raise ImportError(f"{path} does not define KalmanNetNN")
    return mod.KalmanNetNN


def build_and_load_kalmannet(
    sd: dict,
    sensors_xy: torch.Tensor,
    *,
    dt: float,
    tau: float,
    sigma: float,
    min_range: float,
    eps_range: float,
    device: torch.device,
    kalmannet_nn_file: str,
    logger: Optional[Node],
) -> torch.nn.Module:
    """
    Build KalmanNetNN architecture and load weights.
    """
    KalmanNetNN = import_kalmannet_nn(kalmannet_nn_file, logger=logger)

    m, n, in_mult, out_mult = infer_hparams_from_state_dict(sd)

    if int(sensors_xy.shape[0]) != n:
        raise RuntimeError(f"layout N={int(sensors_xy.shape[0])} but model expects n={n}")

    dt = float(dt)
    tau = float(tau)
    sigma = float(sigma)

    # velocity decay (if tau<=0 -> no decay)
    a = 1.0 if tau <= 0.0 else float(math.exp(-dt / tau))

    def f(x: torch.Tensor) -> torch.Tensor:
        """
        Constant-velocity-ish model with optional velocity decay:
          x = [px, py, vx, vy, ...]
        """
        B = int(x.shape[0])
        F = torch.eye(m, device=x.device, dtype=x.dtype)
        if m >= 4:
            F[0, 2] = dt
            F[1, 3] = dt
            F[2, 2] = a
            F[3, 3] = a
        F = F.unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(F, x)

    def h(x: torch.Tensor) -> torch.Tensor:
        """
        Range to sensors using first 2 dims as (x,y).
        """
        B = int(x.shape[0])
        px = x[:, 0, 0].view(B, 1)
        py = x[:, 1, 0].view(B, 1)
        dx = px - sensors_xy[:, 0].view(1, n)
        dy = py - sensors_xy[:, 1].view(1, n)
        d = torch.sqrt(dx * dx + dy * dy + float(eps_range))
        d = torch.clamp(d, min=float(min_range))
        return d.view(B, n, 1)

    # Priors: only used to init GRU hidden states on reset.
    # These should be close to what you used during training.
    prior_Q = torch.eye(m, device=device)
    prior_Sigma = torch.eye(m, device=device)
    prior_S = torch.eye(n, device=device) * float(sigma * sigma)

    sysm = SimpleNamespace(
        f=f,
        h=h,
        m=m,
        n=n,
        prior_Q=prior_Q,
        prior_Sigma=prior_Sigma,
        prior_S=prior_S,
    )

    args = SimpleNamespace(
        use_cuda=(device.type == "cuda"),
        n_batch=1,
        in_mult_KNet=int(in_mult),
        out_mult_KNet=int(out_mult),
    )

    model = KalmanNetNN()
    model.NNBuild(sysm, args)

    # strict=True is correct; if this fails, you imported the wrong KalmanNet_nn.py
    model.load_state_dict(sd, strict=True)

    if logger is not None:
        logger.get_logger().info(
            f"KalmanNetNN built: m={m} n={n} in_mult={in_mult} out_mult={out_mult}"
        )

    return model


# ----------------------------- ROS2 node -----------------------------

class KNetTracker(Node):
    def __init__(self):
        super().__init__("knet")  # default name; you can still override with -r __node:=knet

        # --- declare params ---
        try:
            self.declare_parameter("use_sim_time", True)
        except ParameterAlreadyDeclaredException:
            pass

        self.declare_parameter("model_path", "")
        self.declare_parameter("layout_file", "")

        self.declare_parameter("z_topic", "z")
        self.declare_parameter("est_topic", "knet/estimated")

        self.declare_parameter("gt_topic", "gt/odom")
        self.declare_parameter("init_from_gt", True)

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("child_frame", "base_link")

        # training-aligned dynamics params
        self.declare_parameter("delta", 0.1)   # dt
        self.declare_parameter("tau", 1.0)     # vel decay time constant (<=0 disables)
        self.declare_parameter("sigma", 0.10)  # measurement std used for prior_S init

        self.declare_parameter("min_range", 1e-3)
        self.declare_parameter("eps_range", 1e-6)

        self.declare_parameter("use_cuda", False)
        self.declare_parameter("max_seq_len", 1000000)

        # optional gating (0 disables)
        self.declare_parameter("gate_resid_rms", 0.0)

        # optional explicit kalmannet_nn path (file OR directory)
        self.declare_parameter("kalmannet_nn_file", "")

        # --- read params ---
        self.model_path = str(self.get_parameter("model_path").value)
        self.layout_file = str(self.get_parameter("layout_file").value)

        self.z_topic = str(self.get_parameter("z_topic").value)
        self.est_topic = str(self.get_parameter("est_topic").value)

        self.gt_topic = str(self.get_parameter("gt_topic").value)
        self.init_from_gt = bool(self.get_parameter("init_from_gt").value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)

        self.dt = float(self.get_parameter("delta").value)
        self.tau = float(self.get_parameter("tau").value)
        self.sigma = float(self.get_parameter("sigma").value)

        self.min_range = float(self.get_parameter("min_range").value)
        self.eps_range = float(self.get_parameter("eps_range").value)

        self.use_cuda = bool(self.get_parameter("use_cuda").value)
        self.max_seq_len = int(self.get_parameter("max_seq_len").value)

        self.gate_resid_rms = float(self.get_parameter("gate_resid_rms").value)

        self.kalmannet_nn_file = str(self.get_parameter("kalmannet_nn_file").value).strip()

        # --- sanity checks ---
        mp = Path(self.model_path)
        lf = Path(self.layout_file)
        if not mp.is_file():
            raise RuntimeError(f"model_path not found: {mp}")
        if not lf.is_file():
            raise RuntimeError(f"layout_file not found: {lf}")
        if self.dt <= 0.0:
            raise RuntimeError("delta(dt) must be > 0")
        if self.sigma <= 0.0:
            raise RuntimeError("sigma must be > 0")
        if self.min_range <= 0.0:
            raise RuntimeError("min_range must be > 0")

        # --- device ---
        self.device = torch.device("cuda") if (self.use_cuda and torch.cuda.is_available()) else torch.device("cpu")
        self.get_logger().info(f"device={self.device}")

        # --- layout ---
        sensors = load_layout_csv(self.layout_file)
        self.sensors_xy_np = sensors
        self.sensors_xy = torch.tensor(sensors, dtype=torch.float32, device=self.device)
        self.n = int(self.sensors_xy.shape[0])
        self.get_logger().info(f"layout loaded: N={self.n}")

        # --- model ---
        self.model = self._load_model(self.model_path).to(self.device)
        self.model.eval()

        self.m = int(getattr(self.model, "m", 4))
        self.get_logger().info(f"model ready: m={self.m} n={self.n}")

        # --- runtime state ---
        self._initialized = False
        self._step = 0

        # for init_from_gt velocity estimate
        self._gt_prev: Optional[Tuple[float, float, float]] = None  # (t, x, y)
        self._gt_last: Optional[Tuple[float, float, float]] = None

        # --- ROS I/O ---
        self.pub = self.create_publisher(Odometry, self.est_topic, 10)
        self.create_subscription(Float32MultiArray, self.z_topic, self._on_z, qos_profile_sensor_data)

        if self.init_from_gt:
            self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
            self.get_logger().info(f"init_from_gt=true -> sub {self.gt_topic}")

        self.get_logger().info(f"sub {self.z_topic} -> pub {self.est_topic}")

    def _load_model(self, path: str) -> torch.nn.Module:
        obj = torch_load_any(path)

        # (1) full model saved
        if isinstance(obj, torch.nn.Module):
            self.get_logger().warn("Checkpoint is a full torch.nn.Module; using directly.")
            return obj

        # (2) checkpoint dict
        if isinstance(obj, dict) and "state_dict" in obj:
            sd_raw = obj["state_dict"]
            if not isinstance(sd_raw, dict):
                raise RuntimeError("checkpoint['state_dict'] is not a dict")
            sd = strip_module_prefix(sd_raw)
        # (3) raw state_dict
        elif isinstance(obj, dict) and "FC5.0.weight" in obj:
            sd = strip_module_prefix(obj)
        else:
            raise RuntimeError(f"Unknown checkpoint format: {type(obj)}")

        return build_and_load_kalmannet(
            sd,
            self.sensors_xy,
            dt=self.dt,
            tau=self.tau,
            sigma=self.sigma,
            min_range=self.min_range,
            eps_range=self.eps_range,
            device=self.device,
            kalmannet_nn_file=self.kalmannet_nn_file,
            logger=self,
        )

    def _reset_filter(self, x0: torch.Tensor) -> None:
        # KalmanNetNN requires these
        if hasattr(self.model, "batch_size"):
            self.model.batch_size = 1  # type: ignore[attr-defined]

        if not hasattr(self.model, "init_hidden_KNet") or not hasattr(self.model, "InitSequence"):
            raise RuntimeError("Loaded model is not KalmanNetNN (missing init_hidden_KNet / InitSequence)")

        self.model.init_hidden_KNet()  # type: ignore[attr-defined]
        self.model.InitSequence(x0.view(1, -1, 1).to(self.device), int(self.max_seq_len))  # type: ignore[attr-defined]

        self._initialized = True
        self._step = 0
        self.get_logger().info(f"[RESET] x0={x0.detach().cpu().numpy().tolist()}")

    def _build_init(self, z_np: np.ndarray) -> torch.Tensor:
        if self.init_from_gt:
            if self._gt_last is None:
                raise RuntimeError("waiting for GT")
            t2, x2, y2 = self._gt_last

            vx = vy = 0.0
            if self._gt_prev is not None:
                t1, x1, y1 = self._gt_prev
                dt = t2 - t1
                if dt > 1e-6 and math.isfinite(dt):
                    vx = (x2 - x1) / dt
                    vy = (y2 - y1) / dt

            base = [float(x2), float(y2)]
            if self.m >= 4:
                base += [float(vx), float(vy)]
        else:
            xy = ls_init_xy_from_ranges(self.sensors_xy_np, z_np, min_range=self.min_range)
            base = [float(xy[0]), float(xy[1])]
            if self.m >= 4:
                base += [0.0, 0.0]

        base += [0.0] * max(0, self.m - len(base))
        return torch.tensor(base, dtype=torch.float32, device=self.device)

    def _on_gt(self, msg: Odometry) -> None:
        # Prefer message stamp (sim time), fallback to node clock if needed
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if not math.isfinite(t) or t <= 0.0:
            t = float(self.get_clock().now().nanoseconds) * 1e-9

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        self._gt_prev = self._gt_last
        self._gt_last = (t, x, y)

    def _on_z(self, msg: Float32MultiArray) -> None:
        z_np = np.asarray(msg.data, dtype=np.float32).reshape((-1,))
        if z_np.size != self.n:
            self.get_logger().warn(f"z dim mismatch: got {z_np.size}, expected {self.n}")
            return
        if not np.all(np.isfinite(z_np)):
            self.get_logger().warn("z has NaN/inf; skip")
            return

        z_np = np.maximum(z_np, float(self.min_range))
        z = torch.tensor(z_np, dtype=torch.float32, device=self.device)

        # init
        if not self._initialized:
            try:
                x0 = self._build_init(z_np)
            except Exception as e:
                # don't spam too hard
                self.get_logger().warn(str(e))
                return
            self._reset_filter(x0)

        # forward
        y = z.view(1, self.n, 1)
        with torch.inference_mode():
            xhat = self.model(y).squeeze(-1).squeeze(0)  # (m,)

        if not torch.isfinite(xhat).all():
            self.get_logger().warn("[RESET] x_hat NaN/inf -> reset")
            try:
                self._reset_filter(self._build_init(z_np))
            except Exception as e:
                self.get_logger().warn(f"reset failed: {e}")
            return

        # optional gating by measurement consistency
        if self.gate_resid_rms > 0.0 and xhat.numel() >= 2:
            dx = xhat[0] - self.sensors_xy[:, 0]
            dy = xhat[1] - self.sensors_xy[:, 1]
            pred = torch.sqrt(dx * dx + dy * dy + float(self.eps_range)).clamp(min=float(self.min_range))
            resid_rms = torch.sqrt(torch.mean((pred - z) ** 2)).item()
            if math.isfinite(resid_rms) and resid_rms > self.gate_resid_rms:
                self.get_logger().warn(
                    f"[GATE] resid_rms={resid_rms:.3f} > {self.gate_resid_rms:.3f} -> reset"
                )
                try:
                    self._reset_filter(self._build_init(z_np))
                except Exception as e:
                    self.get_logger().warn(f"reset failed: {e}")
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

        if self.m >= 4 and xhat.numel() >= 4:
            od.twist.twist.linear.x = float(xhat[2].item())
            od.twist.twist.linear.y = float(xhat[3].item())

        self.pub.publish(od)

        self._step += 1
        if self._step >= self.max_seq_len - 1:
            self.get_logger().warn("[RESET] max_seq_len reached -> reset with last state")
            try:
                self._reset_filter(xhat.detach())
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
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
