#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry


def load_layout_csv(path: str) -> np.ndarray:
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
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
    return np.asarray(pts, dtype=np.float32)  # (N,2)


def predict_ranges_xy(xy: torch.Tensor, sensors_xy: torch.Tensor, eps: float, min_range: float) -> torch.Tensor:
    # xy: (2,) ; sensors_xy: (N,2)
    dx = xy[0] - sensors_xy[:, 0]
    dy = xy[1] - sensors_xy[:, 1]
    d = torch.sqrt(dx * dx + dy * dy + eps)
    return torch.clamp(d, min=min_range)


def init_pos_from_ranges_gd(
    z: torch.Tensor,
    sensors_xy: torch.Tensor,
    eps: float,
    min_range: float,
    iters: int = 200,
    lr: float = 0.05,
) -> torch.Tensor:
    """
    Basit init: range multilateration için gradient descent.
    z: (N,) ranges
    sensors_xy: (N,2)
    returns: (2,) xy
    """
    # birkaç başlangıç dene (merkez ve 4 köşe benzeri)
    starts = [
        torch.zeros(2, device=z.device),
        torch.tensor([+1.0, +1.0], device=z.device),
        torch.tensor([-1.0, +1.0], device=z.device),
        torch.tensor([+1.0, -1.0], device=z.device),
        torch.tensor([-1.0, -1.0], device=z.device),
    ]

    best_xy = None
    best_loss = float("inf")

    for x0 in starts:
        xy = x0.clone().detach().requires_grad_(True)
        opt = torch.optim.SGD([xy], lr=lr, momentum=0.9)

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            pred = predict_ranges_xy(xy, sensors_xy, eps=eps, min_range=min_range)
            loss = torch.mean((pred - z) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([xy], 10.0)
            opt.step()

        with torch.no_grad():
            pred = predict_ranges_xy(xy, sensors_xy, eps=eps, min_range=min_range)
            loss = torch.mean((pred - z) ** 2).item()
            if loss < best_loss:
                best_loss = loss
                best_xy = xy.detach().clone()

    return best_xy


class KNetTracker(Node):
    def __init__(self):
        super().__init__("knet_tracker")

        # ---- params ----
        self.declare_parameter("model_path", "")
        self.declare_parameter("layout_file", "")
        self.declare_parameter("z_topic", "z")
        self.declare_parameter("est_topic", "knet/estimated")
        self.declare_parameter("gt_topic", "gt/odom")
        self.declare_parameter("init_from_gt", True)

        self.declare_parameter("world_frame", "world")
        self.declare_parameter("child_frame", "base_link")

        self.declare_parameter("use_cuda", False)

        # range model params (only for gating + init)
        self.declare_parameter("min_range", 1e-3)
        self.declare_parameter("eps_range", 1e-6)

        # gating (çok kritik: outlier’da reset atar)
        self.declare_parameter("gate_resid_rms", 0.30)  # ~3*sigma gibi düşün (sigma=0.10)
        self.declare_parameter("max_seq_len", 10000)    # online initSequence için

        # ---- read params ----
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        layout_file = self.get_parameter("layout_file").get_parameter_value().string_value
        self.z_topic = self.get_parameter("z_topic").get_parameter_value().string_value
        self.est_topic = self.get_parameter("est_topic").get_parameter_value().string_value
        self.gt_topic = self.get_parameter("gt_topic").get_parameter_value().string_value
        self.init_from_gt = self.get_parameter("init_from_gt").get_parameter_value().bool_value

        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.child_frame = self.get_parameter("child_frame").get_parameter_value().string_value

        self.min_range = float(self.get_parameter("min_range").value)
        self.eps = float(self.get_parameter("eps_range").value)

        self.gate_resid_rms = float(self.get_parameter("gate_resid_rms").value)
        self.max_seq_len = int(self.get_parameter("max_seq_len").value)

        use_cuda = bool(self.get_parameter("use_cuda").value)
        self.device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        self.get_logger().info(f"device={self.device}")

        if not model_path:
            raise RuntimeError("model_path param boş. Örn: -p model_path:=models/.../best-model.pt")
        if not layout_file:
            raise RuntimeError("layout_file param boş. Örn: -p layout_file:=config/paper_sensors_5x5_b20.csv")

        self.model_path = model_path
        self.layout_file = layout_file

        # ---- load layout ----
        sensors = load_layout_csv(layout_file)
        self.sensors_xy = torch.tensor(sensors, dtype=torch.float32, device=self.device)  # (N,2)
        self.obs_dim = self.sensors_xy.shape[0]
        self.get_logger().info(f"layout loaded: N_sensors={self.obs_dim}")

        # ---- load model ----
        self.model = self._load_model(self.model_path)
        self.model.eval()
        self.model.to(self.device)

        # internal state
        self._initialized = False
        self._step = 0
        self._last_gt: Optional[Odometry] = None

        # pubs/subs
        self.pub = self.create_publisher(Odometry, self.est_topic, 10)
        self.sub_z = self.create_subscription(Float32MultiArray, self.z_topic, self._on_z, 10)

        if self.init_from_gt:
            self.sub_gt = self.create_subscription(Odometry, self.gt_topic, self._on_gt, 10)
            self.get_logger().info(f"init_from_gt=true -> subscribing gt_topic={self.gt_topic}")

        self.get_logger().info(f"sub z_topic={self.z_topic} pub est_topic={self.est_topic}")

    def _load_model(self, path: str):
        # PyTorch 2.6: weights_only default True -> bizim checkpoint full obj olduğu için weights_only=False lazım
        try:
            m = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            m = torch.load(path, map_location=self.device)
        return m

    def _on_gt(self, msg: Odometry):
        self._last_gt = msg

    def _reset_filter(self, x0: torch.Tensor):
        """
        x0: (4,) [px,py,vx,vy]
        """
        self.model.batch_size = 1
        self.model.init_hidden_KNet()
        # InitSequence expects (B,m,1)
        x0_b = x0.view(1, 4, 1).to(self.device)
        # T burada "maksimum adım" gibi kullanılacak (online)
        self.model.InitSequence(x0_b, self.max_seq_len)

        self._initialized = True
        self._step = 0
        self.get_logger().info(f"[RESET] x0={x0.detach().cpu().numpy().tolist()}")

    def _build_init(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N,) float32
        if self.init_from_gt:
            if self._last_gt is None:
                raise RuntimeError("init_from_gt=true ama henüz gt mesajı gelmedi.")
            gx = float(self._last_gt.pose.pose.position.x)
            gy = float(self._last_gt.pose.pose.position.y)
            gvx = float(self._last_gt.twist.twist.linear.x)
            gvy = float(self._last_gt.twist.twist.linear.y)
            return torch.tensor([gx, gy, gvx, gvy], dtype=torch.float32, device=self.device)

        # GT yoksa: z'den pozisyon init + v=0
        xy = init_pos_from_ranges_gd(z, self.sensors_xy, eps=self.eps, min_range=self.min_range)
        return torch.tensor([xy[0], xy[1], 0.0, 0.0], dtype=torch.float32, device=self.device)

    def _on_z(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != self.obs_dim:
            self.get_logger().warn(f"z dim mismatch: got={data.size} expected={self.obs_dim}")
            return

        z = torch.tensor(data, dtype=torch.float32, device=self.device)  # (N,)

        # init
        if not self._initialized:
            try:
                x0 = self._build_init(z)
            except Exception as e:
                # GT bekliyorsak daha gelmemiş olabilir
                self.get_logger().warn(f"init bekleniyor: {repr(e)}")
                return
            self._reset_filter(x0)

        # forward step
        y = z.view(1, self.obs_dim, 1)  # (B,N,1)
        with torch.inference_mode():
            xhat = self.model(y).squeeze(-1).squeeze(0)  # (4,)

        # gating: ölçüm tutarlılığı kontrolü (çok işe yarar)
        if self.gate_resid_rms > 0.0:
            pred = predict_ranges_xy(xhat[0:2], self.sensors_xy, eps=self.eps, min_range=self.min_range)
            resid_rms = torch.sqrt(torch.mean((pred - z) ** 2)).item()
            if math.isfinite(resid_rms) and resid_rms > self.gate_resid_rms:
                self.get_logger().warn(f"[GATE] resid_rms={resid_rms:.3f} > {self.gate_resid_rms:.3f} -> reset")
                # reset’i measurement’tan tekrar başlat
                try:
                    x0 = self._build_init(z)
                    self._reset_filter(x0)
                except Exception as e:
                    self.get_logger().warn(f"reset init failed: {repr(e)}")
                return

        # publish odom
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame
        od.pose.pose.position.x = float(xhat[0].item())
        od.pose.pose.position.y = float(xhat[1].item())
        od.pose.pose.position.z = 0.0
        od.twist.twist.linear.x = float(xhat[2].item())
        od.twist.twist.linear.y = float(xhat[3].item())

        self.pub.publish(od)

        self._step += 1
        if self._step >= self.max_seq_len - 1:
            # sequence length dolarsa kendini resetle (online safety)
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
