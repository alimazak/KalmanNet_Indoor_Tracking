#!/usr/bin/env python3
"""
EKF tracker from range-only measurements (2D multilateration).

State: [x, y, vx, vy]^T

Measurement model:
  z_i = sqrt((x - sx_i)^2 + (y - sy_i)^2) + v_i

This node:
  - Subscribes:  /range/z  (Float32MultiArray)
                /ground_truth/odom (Odometry)   [for metrics + GT path, optional GT init if enabled]
  - Publishes:   /tracking/odom (Odometry)
                /tracking/path (Path)
                /ground_truth/path (Path)
                /tracking/error (Float32)
                /tracking/rmse (Float32)
                /tracking/rmse_window (Float32)
  - Service:     /tracking/reset (std_srvs/Empty)
  - Optional: pushes a Gazebo proxy model pose via /world/<gz_world>/set_pose
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from typing import List, Sequence, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import Empty

from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


# ----------------------------- Utilities -----------------------------
def load_layout_csv(path: str) -> List[Tuple[float, float]]:
    """CSV: each line 'x,y' or 'x y' (comments with '#'). Returns list[(x,y)]."""
    pts: List[Tuple[float, float]] = []
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
    return pts


def ls_init_xy_from_ranges(
    sensors_xy: Sequence[Tuple[float, float]],
    z: np.ndarray,
) -> Tuple[float, float]:
    """
    Linear LS multilateration (2D).
    Choose the smallest range as reference for stability.

    Sensors: [(xi, yi)] length N
    z: ranges (N,)
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    N = len(sensors_xy)
    if z.shape[0] != N:
        raise ValueError(f"z length {z.shape[0]} != N {N}")
    if N < 3:
        raise ValueError(f"Need at least 3 sensors for 2D LS init, got N={N}")

    eps = 1e-3
    z = np.maximum(np.abs(z), eps)

    i0 = int(np.argmin(z))
    x0, y0 = sensors_xy[i0]
    z0 = float(z[i0])

    A = []
    b = []
    for i, (xi, yi) in enumerate(sensors_xy):
        if i == i0:
            continue
        zi = float(z[i])
        A.append([2.0 * (xi - x0), 2.0 * (yi - y0)])
        b.append((xi * xi - x0 * x0) + (yi * yi - y0 * y0) - (zi * zi - z0 * z0))

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("Degenerate sensor geometry (rank < 2).")

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_est, y_est = float(sol[0]), float(sol[1])

    if not (math.isfinite(x_est) and math.isfinite(y_est)):
        raise ValueError("LS returned non-finite solution")

    return x_est, y_est


# ----------------------------- Config -----------------------------
@dataclass(frozen=True)
class EKFConfig:
    # Topics
    z_topic: str
    gt_topic: str
    odom_out: str
    path_out: str
    gt_path_out: str
    err_out: str
    rmse_out: str
    rmse_window_out: str

    # Frames
    world_frame: str
    child_frame: str

    # Sensor layout
    layout_file: str

    # Noise / models
    sigma: float
    delta: float
    tau: float

    # Init
    init_from_gt: bool
    init_pos_std: float
    init_vel_std: float

    # Buffers
    max_path_len: int
    rmse_window_N: int

    # Gazebo proxy
    gz_world: str
    gz_entity: str
    gz_z: float


# ----------------------------- Core models -----------------------------
class ConstantVelocityModel:
    """x,y,vx,vy constant-velocity discrete model with delta and tau."""

    def __init__(self, delta: float, tau: float):
        d = float(delta)
        t = float(tau)

        self.F = np.array(
            [
                [1.0, 0.0, d, 0.0],
                [0.0, 1.0, 0.0, d],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Continuous white-noise acceleration discretization (as in many tracking texts)
        self.Q = t * np.array(
            [
                [d**3 / 3.0, 0.0, d**2 / 2.0, 0.0],
                [0.0, d**3 / 3.0, 0.0, d**2 / 2.0],
                [d**2 / 2.0, 0.0, d, 0.0],
                [0.0, d**2 / 2.0, 0.0, d],
            ],
            dtype=float,
        )


class RangeMeasurementModel:
    """Range measurement model h(x), H(x) for sensors at fixed positions."""

    def __init__(self, sensors_xy: Sequence[Tuple[float, float]]):
        sensors = np.asarray(sensors_xy, dtype=float)
        if sensors.ndim != 2 or sensors.shape[1] != 2:
            raise ValueError("sensors_xy must be Nx2")
        self.sensors_xy = sensors_xy
        self.sensors = sensors
        self.N = int(sensors.shape[0])
        self._sx = sensors[:, 0]
        self._sy = sensors[:, 1]

    def predict(self, x: float, y: float) -> np.ndarray:
        dx = x - self._sx
        dy = y - self._sy
        return np.sqrt(dx * dx + dy * dy)

    def jacobian(self, x: float, y: float) -> np.ndarray:
        dx = x - self._sx
        dy = y - self._sy
        d = np.sqrt(dx * dx + dy * dy)
        d = np.maximum(d, 1e-6)

        H = np.zeros((self.N, 4), dtype=float)
        H[:, 0] = dx / d
        H[:, 1] = dy / d
        # vx, vy columns are 0
        return H


# ----------------------------- EKF engine -----------------------------
class RangeEKF:
    """
    Pure EKF engine (no ROS):
      - holds x,P
      - step(z, skip_predict)
    """

    def __init__(
        self,
        process: ConstantVelocityModel,
        meas: RangeMeasurementModel,
        R: np.ndarray,
        P0: np.ndarray,
    ):
        self.proc = process
        self.meas = meas
        self.R = np.asarray(R, dtype=float)
        self.P0 = np.asarray(P0, dtype=float)

        self.x = np.zeros((4, 1), dtype=float)
        self.P = self.P0.copy()
        self.initialized = False

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P = self.P0.copy()
        self.initialized = False

    def initialize(self, x0: Sequence[float]) -> None:
        x0 = np.asarray(x0, dtype=float).reshape((4, 1))
        self.x = x0
        self.P = self.P0.copy()
        self.initialized = True

    def step(self, z: np.ndarray, skip_predict: bool) -> None:
        z = np.asarray(z, dtype=float).reshape((-1,))
        if z.shape[0] != self.meas.N:
            raise ValueError(f"z length {z.shape[0]} != N {self.meas.N}")

        # ---- Predict ----
        if skip_predict:
            x_pred = self.x.copy()
            P_pred = self.P.copy()
        else:
            x_pred = self.proc.F @ self.x
            P_pred = self.proc.F @ self.P @ self.proc.F.T + self.proc.Q

        # ---- Update ----
        px = float(x_pred[0, 0])
        py = float(x_pred[1, 0])

        z_pred = self.meas.predict(px, py)
        H = self.meas.jacobian(px, py)

        innov = (z - z_pred).reshape((self.meas.N, 1))  # (N,1)
        S = H @ P_pred @ H.T + self.R                   # (N,N)
        PHt = P_pred @ H.T                              # (4,N)

        # K = PHt * inv(S)  -> solve(S * K^T = PHt^T)
        K = np.linalg.solve(S, PHt.T).T                 # (4,N)

        self.x = x_pred + K @ innov

        # Joseph form: numerically safer and keeps P symmetric PSD (as much as possible)
        I = np.eye(4, dtype=float)
        IKH = I - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry


# ----------------------------- Small ROS helpers -----------------------------
class PathBuffer:
    """Keeps a bounded Path message."""

    def __init__(self, frame_id: str, max_len: int):
        self.max_len = max(1, int(max_len))
        self.msg = Path()
        self.msg.header.frame_id = frame_id

    def reset(self) -> None:
        self.msg.poses.clear()

    def push(self, ps: PoseStamped) -> None:
        self.msg.header = ps.header
        self.msg.poses.append(ps)
        if len(self.msg.poses) > self.max_len:
            del self.msg.poses[:-self.max_len]


class RMSETracker:
    """Tracks instantaneous error, running RMSE and windowed RMSE efficiently."""

    def __init__(self, window_N: int):
        self.window = deque(maxlen=max(1, int(window_N)))
        self.err2_sum = 0.0
        self.count = 0
        self.win_sum = 0.0

    def reset(self) -> None:
        self.window.clear()
        self.err2_sum = 0.0
        self.count = 0
        self.win_sum = 0.0

    def update(self, ex: float, ey: float) -> Tuple[float, float, float]:
        e = math.hypot(ex, ey)
        e2 = e * e

        self.err2_sum += e2
        self.count += 1
        rmse = math.sqrt(self.err2_sum / max(1, self.count))

        if len(self.window) == self.window.maxlen:
            self.win_sum -= self.window[0]
        self.window.append(e2)
        self.win_sum += e2
        rmse_w = math.sqrt(self.win_sum / max(1, len(self.window)))

        return e, rmse, rmse_w


class GazeboProxyPusher:
    """Asynchronously pushes a model pose to Gazebo via SetEntityPose service."""

    def __init__(self, node: Node, gz_world: str, entity_name: str, z: float):
        self._node = node
        self._entity_name = entity_name
        self._z = float(z)
        self._pending = False
        self._cli = node.create_client(SetEntityPose, f"/world/{gz_world}/set_pose")

    def push(self, x: float, y: float) -> None:
        if self._pending:
            return
        if not self._cli.service_is_ready():
            return

        req = SetEntityPose.Request()
        req.entity.name = self._entity_name
        if hasattr(req.entity, "type"):
            req.entity.type = Entity.MODEL

        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self._z)
        req.pose.orientation.w = 1.0

        self._pending = True
        fut = self._cli.call_async(req)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        self._pending = False
        try:
            fut.result()
        except Exception as e:
            self._node.get_logger().warn(f"set_pose failed: {e}")


# ----------------------------- ROS Node -----------------------------
class EKFRangeNode(Node):
    def __init__(self):
        super().__init__("ekf_tracker_from_range")

        cfg = self._declare_and_read_config()

        # --- Layout / models ---
        sensors_xy = load_layout_csv(cfg.layout_file)
        if not sensors_xy:
            raise RuntimeError(f"layout_file okunamadı/boş: {cfg.layout_file}")

        self._meas = RangeMeasurementModel(sensors_xy)
        self._proc = ConstantVelocityModel(cfg.delta, cfg.tau)

        R = (cfg.sigma ** 2) * np.eye(self._meas.N, dtype=float)
        P0 = np.diag(
            [
                cfg.init_pos_std**2,
                cfg.init_pos_std**2,
                cfg.init_vel_std**2,
                cfg.init_vel_std**2,
            ]
        ).astype(float)

        self._ekf = RangeEKF(self._proc, self._meas, R=R, P0=P0)
        self._cfg = cfg

        # --- GT cache (metrics only unless init_from_gt enabled) ---
        self._gt_ready = False
        self._gt_x = self._gt_y = 0.0
        self._gt_vx = self._gt_vy = 0.0

        # --- Buffers / metrics ---
        self._path = PathBuffer(frame_id=cfg.world_frame, max_len=cfg.max_path_len)
        self._gt_path = PathBuffer(frame_id=cfg.world_frame, max_len=cfg.max_path_len)
        self._rmse = RMSETracker(window_N=cfg.rmse_window_N)

        # --- Gazebo proxy ---
        self._gz = GazeboProxyPusher(self, cfg.gz_world, cfg.gz_entity, cfg.gz_z)

        # --- Publishers ---
        self._pub_odom = self.create_publisher(Odometry, cfg.odom_out, 10)
        self._pub_path = self.create_publisher(Path, cfg.path_out, 10)
        self._pub_gt_path = self.create_publisher(Path, cfg.gt_path_out, 10)
        self._pub_err = self.create_publisher(Float32, cfg.err_out, 10)
        self._pub_rmse = self.create_publisher(Float32, cfg.rmse_out, 10)
        self._pub_rmse_w = self.create_publisher(Float32, cfg.rmse_window_out, 10)

        # --- Subscribers ---
        self.create_subscription(Odometry, cfg.gt_topic, self._on_gt, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, cfg.z_topic, self._on_z, qos_profile_sensor_data)

        # --- Reset service ---
        self.create_service(Empty, "/tracking/reset", self._on_reset)

        self.get_logger().info(
            f"EKF(range) ready. N_sensors={self._meas.N}, init_from_gt={cfg.init_from_gt}. Waiting for {cfg.z_topic}..."
        )

    def _declare_and_read_config(self) -> EKFConfig:
        # Topics
        z_topic = self.declare_parameter("z_topic", "/range/z").value
        gt_topic = self.declare_parameter("gt_topic", "/ground_truth/odom").value
        odom_out = self.declare_parameter("odom_out", "/tracking/odom").value
        path_out = self.declare_parameter("path_out", "/tracking/path").value
        gt_path_out = self.declare_parameter("gt_path_out", "/ground_truth/path").value
        err_out = self.declare_parameter("err_out", "/tracking/error").value
        rmse_out = self.declare_parameter("rmse_out", "/tracking/rmse").value
        rmse_window_out = self.declare_parameter("rmse_window_out", "/tracking/rmse_window").value

        # Frames
        world_frame = self.declare_parameter("world_frame", "world").value
        child_frame = self.declare_parameter("child_frame", "ekf_base").value

        # Layout
        layout_file = self.declare_parameter("layout_file", "").value
        if not layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")

        # Noise / models
        sigma = float(self.declare_parameter("sigma", 0.10).value)
        delta = float(self.declare_parameter("delta", 0.1).value)
        tau = float(self.declare_parameter("tau", 1.0).value)

        # Init
        init_from_gt = bool(self.declare_parameter("init_from_gt", False).value)  # default FALSE
        init_pos_std = float(self.declare_parameter("init_pos_std", 5.0).value)
        init_vel_std = float(self.declare_parameter("init_vel_std", 2.0).value)

        # Buffers
        max_path_len = int(self.declare_parameter("max_path_len", 2000).value)
        rmse_window_N = int(self.declare_parameter("rmse_window_N", 200).value)

        # Gazebo proxy
        gz_world = self.declare_parameter("gz_world", "empty_world").value
        gz_entity = self.declare_parameter("gz_entity", "ekf_proxy").value
        gz_z = float(self.declare_parameter("gz_z", 0.01).value)

        return EKFConfig(
            z_topic=str(z_topic),
            gt_topic=str(gt_topic),
            odom_out=str(odom_out),
            path_out=str(path_out),
            gt_path_out=str(gt_path_out),
            err_out=str(err_out),
            rmse_out=str(rmse_out),
            rmse_window_out=str(rmse_window_out),
            world_frame=str(world_frame),
            child_frame=str(child_frame),
            layout_file=str(layout_file),
            sigma=float(sigma),
            delta=float(delta),
            tau=float(tau),
            init_from_gt=bool(init_from_gt),
            init_pos_std=float(init_pos_std),
            init_vel_std=float(init_vel_std),
            max_path_len=int(max_path_len),
            rmse_window_N=int(rmse_window_N),
            gz_world=str(gz_world),
            gz_entity=str(gz_entity),
            gz_z=float(gz_z),
        )

    # -------------------- Callbacks --------------------
    def _on_reset(self, req, resp):
        self._ekf.reset()
        self._rmse.reset()
        self._path.reset()
        self._gt_path.reset()
        self.get_logger().warn("TRACKING RESET: EKF state/cov + paths + rmse cleared.")
        return resp

    def _on_gt(self, msg: Odometry):
        # cache (metrics + optional init)
        self._gt_x = msg.pose.pose.position.x
        self._gt_y = msg.pose.pose.position.y
        self._gt_vx = msg.twist.twist.linear.x
        self._gt_vy = msg.twist.twist.linear.y
        self._gt_ready = True

        # GT path publish
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self._cfg.world_frame
        ps.pose = msg.pose.pose

        self._gt_path.push(ps)
        self._pub_gt_path.publish(self._gt_path.msg)

    def _initialize_filter(self, z: np.ndarray) -> bool:
        """
        Initializes EKF and returns skip_predict=True for the first update.
        """
        cfg = self._cfg

        if cfg.init_from_gt and self._gt_ready:
            # Debug/sanity-check mode only
            self._ekf.initialize([self._gt_x, self._gt_y, self._gt_vx, self._gt_vy])
            self.get_logger().warn("EKF initialized from ground truth (debug/sanity-check).")
            return True

        # Default: LS init from ranges
        try:
            x0, y0 = ls_init_xy_from_ranges(self._meas.sensors_xy, z)
            self._ekf.initialize([x0, y0, 0.0, 0.0])
            self.get_logger().warn(f"EKF initialized from ranges (LS): x={x0:.2f}, y={y0:.2f}")
            return True
        except Exception as e:
            # Last-resort fallback: zeros
            self._ekf.initialize([0.0, 0.0, 0.0, 0.0])
            self.get_logger().error(f"LS init failed, fallback zeros: {e}")
            return True

    def _on_z(self, msg: Float32MultiArray):
        z = np.asarray(msg.data, dtype=float).reshape((-1,))
        if z.shape[0] != self._meas.N:
            self.get_logger().error(
                f"/range/z length={z.shape[0]} but sensors={self._meas.N}. layout mismatch!"
            )
            return

        # INIT (once)
        skip_predict = False
        if not self._ekf.initialized:
            skip_predict = self._initialize_filter(z)

        # EKF step
        try:
            self._ekf.step(z, skip_predict=skip_predict)
        except Exception as e:
            self.get_logger().error(f"EKF step failed: {e}")
            return

        # Publish odom + path
        now = self.get_clock().now().to_msg()
        x = float(self._ekf.x[0, 0])
        y = float(self._ekf.x[1, 0])
        vx = float(self._ekf.x[2, 0])
        vy = float(self._ekf.x[3, 0])

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = self._cfg.world_frame
        od.child_frame_id = self._cfg.child_frame
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.w = 1.0
        od.twist.twist.linear.x = vx
        od.twist.twist.linear.y = vy
        self._pub_odom.publish(od)

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = self._cfg.world_frame
        ps.pose = od.pose.pose

        self._path.push(ps)
        self._pub_path.publish(self._path.msg)

        # Gazebo proxy (visual trail)
        self._gz.push(x, y)

        # Metrics (only if GT exists)
        if self._gt_ready:
            ex = x - self._gt_x
            ey = y - self._gt_y
            e, rmse, rmse_w = self._rmse.update(ex, ey)
            self._pub_err.publish(Float32(data=float(e)))
            self._pub_rmse.publish(Float32(data=float(rmse)))
            self._pub_rmse_w.publish(Float32(data=float(rmse_w)))


def main():
    rclpy.init()
    rclpy.spin(EKFRangeNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
