#!/usr/bin/env python3
import math
from collections import deque

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


def load_layout_csv(path: str):
    """CSV: each line 'x,y' or 'x y' (comments with '#'). Returns list[(x,y)]."""
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
    return pts


class EKFRange(Node):
    """
    EKF for range measurements:
      z_i = d_i + v
      d_i = sqrt((x-x_i)^2+(y-y_i)^2)
    State: [x, y, vx, vy]^T
    """

    def __init__(self):
        super().__init__("ekf_tracker_from_range")

        # ----- Params: topics -----
        self.z_topic = self.declare_parameter("z_topic", "/range/z").value
        self.gt_topic = self.declare_parameter("gt_topic", "/ground_truth/odom").value

        self.odom_out = self.declare_parameter("odom_out", "/tracking/odom").value
        self.path_out = self.declare_parameter("path_out", "/tracking/path").value
        self.gt_path_out = self.declare_parameter("gt_path_out", "/ground_truth/path").value
        self.err_out = self.declare_parameter("err_out", "/tracking/error").value
        self.rmse_out = self.declare_parameter("rmse_out", "/tracking/rmse").value
        self.rmse_win_out = self.declare_parameter("rmse_window_out", "/tracking/rmse_window").value

        # ----- Params: frames -----
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.child_frame = self.declare_parameter("child_frame", "ekf_base").value

        # ----- Params: sensor layout -----
        self.layout_file = self.declare_parameter("layout_file", "").value
        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")
        self.sensors = load_layout_csv(self.layout_file)
        if not self.sensors:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")
        self.N = len(self.sensors)
        self.get_logger().info(f"Loaded {self.N} sensors from {self.layout_file}")

        # ----- Params: measurement noise -----
        self.sigma = float(self.declare_parameter("sigma", 0.10).value)  # meters
        self.R = (self.sigma ** 2) * np.eye(self.N, dtype=float)

        # ----- Params: process model (paper F,Q) -----
        self.delta = float(self.declare_parameter("delta", 0.1).value)  # seconds
        self.tau = float(self.declare_parameter("tau", 1.0).value)

        # IMPORTANT: default FALSE (GT init kapalı)
        self.init_from_gt = bool(self.declare_parameter("init_from_gt", False).value)

        # Initial covariance prior
        self.init_pos_std = float(self.declare_parameter("init_pos_std", 5.0).value)
        self.init_vel_std = float(self.declare_parameter("init_vel_std", 2.0).value)

        self.max_path_len = int(self.declare_parameter("max_path_len", 2000).value)

        # windowed rmse
        self.rmse_window_N = int(self.declare_parameter("rmse_window_N", 200).value)
        self._e2_win = deque(maxlen=max(1, self.rmse_window_N))

        # Build F,Q
        self._build_FQ(self.delta, self.tau)

        # EKF state/cov
        self.x = np.zeros((4, 1), dtype=float)
        self.P0 = np.diag([
            self.init_pos_std ** 2,
            self.init_pos_std ** 2,
            self.init_vel_std ** 2,
            self.init_vel_std ** 2,
        ]).astype(float)
        self.P = self.P0.copy()
        self.initialized = False

        # GT cache (for error metrics only)
        self.gt_ready = False
        self.gt_x = self.gt_y = 0.0
        self.gt_vx = self.gt_vy = 0.0

        # RMSE accumulators
        self.err2_sum = 0.0
        self.err_count = 0

        # Publishers
        self.pub_odom = self.create_publisher(Odometry, self.odom_out, 10)
        self.pub_path = self.create_publisher(Path, self.path_out, 10)
        self.pub_gt_path = self.create_publisher(Path, self.gt_path_out, 10)
        self.pub_err = self.create_publisher(Float32, self.err_out, 10)
        self.pub_rmse = self.create_publisher(Float32, self.rmse_out, 10)
        self.pub_rmse_win = self.create_publisher(Float32, self.rmse_win_out, 10)

        # Paths
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.world_frame
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = self.world_frame

        # Subscribers
        self.sub_gt = self.create_subscription(Odometry, self.gt_topic, self.on_gt, qos_profile_sensor_data)
        self.sub_z = self.create_subscription(Float32MultiArray, self.z_topic, self.on_z, qos_profile_sensor_data)

        # Reset service
        self.srv_reset = self.create_service(Empty, "/tracking/reset", self.on_reset)

        # --- Gazebo proxy (EKF trail için) ---
        self.gz_world = self.declare_parameter("gz_world", "empty_world").value
        self.gz_entity = self.declare_parameter("gz_entity", "ekf_proxy").value
        self.gz_z = float(self.declare_parameter("gz_z", 0.01).value)

        self._gz_cli = self.create_client(SetEntityPose, f"/world/{self.gz_world}/set_pose")
        self._gz_pending = False

        self.get_logger().info("EKF(range) ready. Waiting for /range/z...")

    # -------------------- Models --------------------
    def _build_FQ(self, delta: float, tau: float):
        self.F = np.array([
            [1.0, 0.0, delta, 0.0],
            [0.0, 1.0, 0.0, delta],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        d = float(delta)
        self.Q = float(tau) * np.array([
            [d ** 3 / 3.0, 0.0,         d ** 2 / 2.0, 0.0],
            [0.0,         d ** 3 / 3.0, 0.0,         d ** 2 / 2.0],
            [d ** 2 / 2.0, 0.0,         d,           0.0],
            [0.0,         d ** 2 / 2.0, 0.0,         d],
        ], dtype=float)

    def h(self, x_pos: float, y_pos: float) -> np.ndarray:
        """measurement function: ranges"""
        zhat = np.zeros((self.N,), dtype=float)
        for i, (sx, sy) in enumerate(self.sensors):
            dx = x_pos - sx
            dy = y_pos - sy
            zhat[i] = math.sqrt(dx * dx + dy * dy)
        return zhat

    def H_jacobian(self, x_pos: float, y_pos: float) -> np.ndarray:
        """Jacobian H (N x 4): [(x-xi)/di, (y-yi)/di, 0, 0]"""
        H = np.zeros((self.N, 4), dtype=float)
        eps = 1e-6
        for i, (sx, sy) in enumerate(self.sensors):
            dx = x_pos - sx
            dy = y_pos - sy
            d = math.sqrt(dx * dx + dy * dy)
            d = d if d > eps else eps
            H[i, 0] = dx / d
            H[i, 1] = dy / d
        return H

    # -------------------- Init from ranges (LS) --------------------
    def init_xy_from_ranges_ls(self, z: np.ndarray) -> tuple[float, float]:
        """
        Linear LS multilateration in 2D.

        We pick the smallest measured range as reference for stability.
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape[0] != self.N:
            raise ValueError(f"z length {z.shape[0]} != N {self.N}")
        if self.N < 3:
            raise ValueError(f"Need at least 3 sensors for 2D LS init, got N={self.N}")

        eps = 1e-3
        z = np.maximum(np.abs(z), eps)

        i0 = int(np.argmin(z))
        x0, y0 = self.sensors[i0]
        z0 = float(z[i0])

        A = []
        b = []
        for i, (xi, yi) in enumerate(self.sensors):
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

    # -------------------- ROS callbacks --------------------
    def on_reset(self, req, resp):
        self.x[:] = 0.0
        self.P = self.P0.copy()
        self.initialized = False

        self.err2_sum = 0.0
        self.err_count = 0
        self._e2_win.clear()

        self.path_msg.poses.clear()
        self.gt_path_msg.poses.clear()

        self.get_logger().warn("TRACKING RESET: state/cov/path/rmse cleared.")
        return resp

    def on_gt(self, msg: Odometry):
        # cache for metrics (NOT for init unless init_from_gt=true)
        self.gt_x = msg.pose.pose.position.x
        self.gt_y = msg.pose.pose.position.y
        self.gt_vx = msg.twist.twist.linear.x
        self.gt_vy = msg.twist.twist.linear.y
        self.gt_ready = True

        # publish GT path
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self.world_frame
        ps.pose = msg.pose.pose

        self.gt_path_msg.header = ps.header
        self.gt_path_msg.poses.append(ps)
        if len(self.gt_path_msg.poses) > self.max_path_len:
            self.gt_path_msg.poses = self.gt_path_msg.poses[-self.max_path_len:]
        self.pub_gt_path.publish(self.gt_path_msg)

    def on_z(self, msg: Float32MultiArray):
        z = np.asarray(msg.data, dtype=float).reshape(-1)
        if z.shape[0] != self.N:
            self.get_logger().error(f"/range/z length={z.shape[0]} but sensors={self.N}. layout mismatch!")
            return

        first_step = False

        # ----- INIT -----
        if not self.initialized:
            if self.init_from_gt and self.gt_ready:
                # Debug only
                self.x[0, 0] = self.gt_x
                self.x[1, 0] = self.gt_y
                self.x[2, 0] = self.gt_vx
                self.x[3, 0] = self.gt_vy
                self.get_logger().warn("EKF initialized from ground truth (debug/sanity-check).")
            else:
                try:
                    x0, y0 = self.init_xy_from_ranges_ls(z)
                    self.x[0, 0] = x0
                    self.x[1, 0] = y0
                    self.x[2, 0] = 0.0
                    self.x[3, 0] = 0.0
                    self.get_logger().warn(f"EKF initialized from ranges (LS): x={x0:.2f}, y={y0:.2f}")
                except Exception as e:
                    # Last resort fallback (should be rare with 5x5 sensors)
                    self.x[:, 0] = 0.0
                    self.get_logger().error(f"LS init failed, fallback zeros: {e}")

            # Reset covariance to prior at init time
            self.P = self.P0.copy()
            self.initialized = True
            first_step = True

        # ----- PREDICT -----
        if first_step:
            # LS init zaten "measurement-based" bir guess; ilk adımda gereksiz Q şişirmesin.
            x_pred = self.x.copy()
            P_pred = self.P.copy()
        else:
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q

        # ----- UPDATE -----
        xpx = float(x_pred[0, 0])
        xpy = float(x_pred[1, 0])

        z_pred = self.h(xpx, xpy)
        H = self.H_jacobian(xpx, xpy)

        innov = (z - z_pred).reshape((self.N, 1))          # (N,1)
        S = H @ P_pred @ H.T + self.R                      # (N,N)
        PHt = P_pred @ H.T                                 # (4,N)

        # K = PHt * inv(S)  (daha stabil solve)
        K = np.linalg.solve(S.T, PHt.T).T                  # (4,N)

        self.x = x_pred + K @ innov
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ P_pred

        # ----- PUBLISH -----
        now = self.get_clock().now().to_msg()

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame
        od.pose.pose.position.x = float(self.x[0, 0])
        od.pose.pose.position.y = float(self.x[1, 0])
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.w = 1.0
        od.twist.twist.linear.x = float(self.x[2, 0])
        od.twist.twist.linear.y = float(self.x[3, 0])
        self.pub_odom.publish(od)

        # push ekf_proxy model pose to gazebo (for visual trail)
        self._push_proxy_to_gazebo(self.x[0, 0], self.x[1, 0])

        # path
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = self.world_frame
        ps.pose = od.pose.pose

        self.path_msg.header = ps.header
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > self.max_path_len:
            self.path_msg.poses = self.path_msg.poses[-self.max_path_len:]
        self.pub_path.publish(self.path_msg)

        # metrics (if GT available)
        if self.gt_ready:
            ex = float(self.x[0, 0]) - self.gt_x
            ey = float(self.x[1, 0]) - self.gt_y
            e = math.hypot(ex, ey)

            self.pub_err.publish(Float32(data=float(e)))

            self.err2_sum += e * e
            self.err_count += 1
            rmse = math.sqrt(self.err2_sum / max(1, self.err_count))
            self.pub_rmse.publish(Float32(data=float(rmse)))

            self._e2_win.append(e * e)
            rmse_w = math.sqrt(sum(self._e2_win) / max(1, len(self._e2_win)))
            self.pub_rmse_win.publish(Float32(data=float(rmse_w)))

    # -------------------- Gazebo proxy --------------------
    def _push_proxy_to_gazebo(self, x: float, y: float):
        if self._gz_pending:
            return
        if not self._gz_cli.service_is_ready():
            # servis henüz gelmediyse bekleme yok, sonraki callback'te tekrar dener
            return

        req = SetEntityPose.Request()
        req.entity.name = self.gz_entity
        if hasattr(req.entity, "type"):
            req.entity.type = Entity.MODEL

        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.gz_z)
        req.pose.orientation.w = 1.0

        self._gz_pending = True
        fut = self._gz_cli.call_async(req)
        fut.add_done_callback(self._on_gz_done)

    def _on_gz_done(self, fut):
        self._gz_pending = False
        try:
            fut.result()
        except Exception as e:
            self.get_logger().warn(f"set_pose failed: {e}")


def main():
    rclpy.init()
    rclpy.spin(EKFRange())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
