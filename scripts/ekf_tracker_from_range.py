#!/usr/bin/env python3
"""
ekf_tracker_from_range.py  (REFRACTOR: minimal EKF node)

Goal:
  - EKF node does ONE job only: /tracking/estimated publish
  - No RMSE, no Path, no Gazebo proxy, no GT path inside this node.

Sub:
  - z_topic (Float32MultiArray) default: /range/z

Pub:
  - est_topic (Odometry) default: /tracking/estimated

Service:
  - reset_srv (Empty) default: /tracking/reset

Init:
  - Default: linear LS init from first valid range measurement
  - Optional debug: init_from_gt=true + gt_topic subscription (disabled by default)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

from ekf_range_core import ConstantVelocityModel, RangeMeasurementModel, RangeEKF, ls_init_xy_from_ranges


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


class EKFRangeNode(Node):
    def __init__(self):
        super().__init__("ekf_tracker_from_range")

        # ---------------- Params ----------------
        self.layout_file = self.declare_parameter("layout_file", "").value
        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")

        # I/O topics
        self.z_topic = self.declare_parameter("z_topic", "/range/z").value
        self.est_topic = self.declare_parameter("est_topic", "/tracking/estimated").value

        # Frames
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.child_frame = self.declare_parameter("child_frame", "ekf_base").value

        # Noise/model
        self.sigma = float(self.declare_parameter("sigma", 0.10).value)
        self.delta = float(self.declare_parameter("delta", 0.1).value)
        self.tau = float(self.declare_parameter("tau", 1.0).value)

        # Initial covariance prior
        self.init_pos_std = float(self.declare_parameter("init_pos_std", 5.0).value)
        self.init_vel_std = float(self.declare_parameter("init_vel_std", 2.0).value)

        # Optional debug init from GT (OFF by default)
        self.init_from_gt = bool(self.declare_parameter("init_from_gt", False).value)
        self.gt_topic = self.declare_parameter("gt_topic", "/ground_truth/odom").value

        # Reset service name
        self.reset_srv = self.declare_parameter("reset_srv", "/tracking/reset").value

        # ---------------- Layout / models ----------------
        sensors_xy = load_layout_csv(self.layout_file)
        if not sensors_xy:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        self._meas = RangeMeasurementModel(sensors_xy)
        self._proc = ConstantVelocityModel(self.delta, self.tau)

        R = (self.sigma ** 2) * np.eye(self._meas.N, dtype=float)
        P0 = np.diag(
            [
                self.init_pos_std**2,
                self.init_pos_std**2,
                self.init_vel_std**2,
                self.init_vel_std**2,
            ]
        ).astype(float)

        self._ekf = RangeEKF(self._proc, self._meas, R=R, P0=P0)

        # GT cache (only used if init_from_gt is enabled)
        self._gt_ready = False
        self._gt_x = self._gt_y = 0.0
        self._gt_vx = self._gt_vy = 0.0

        # ---------------- ROS I/O ----------------
        self._pub_est = self.create_publisher(Odometry, self.est_topic, 10)

        self.create_subscription(Float32MultiArray, self.z_topic, self._on_z, qos_profile_sensor_data)

        if self.init_from_gt:
            from nav_msgs.msg import Odometry as OdomMsg  # local import by design
            self.create_subscription(OdomMsg, self.gt_topic, self._on_gt, qos_profile_sensor_data)

        self.create_service(Empty, self.reset_srv, self._on_reset)

        self.get_logger().info(
            f"EKF(range) node up. N_sensors={self._meas.N} | sub: {self.z_topic} | pub: {self.est_topic} | init_from_gt={self.init_from_gt}"
        )

    # ---------------- Callbacks ----------------
    def _on_reset(self, req, resp):
        self._ekf.reset()
        self.get_logger().warn("EKF RESET: state/cov cleared.")
        return resp

    def _on_gt(self, msg):  # type: ignore[no-untyped-def]
        self._gt_x = msg.pose.pose.position.x
        self._gt_y = msg.pose.pose.position.y
        self._gt_vx = msg.twist.twist.linear.x
        self._gt_vy = msg.twist.twist.linear.y
        self._gt_ready = True

    def _try_initialize(self, z: np.ndarray) -> bool:
        """Returns True if filter got initialized (and you should skip_predict on this first step)."""
        if self.init_from_gt:
            if not self._gt_ready:
                return False
            self._ekf.initialize([self._gt_x, self._gt_y, self._gt_vx, self._gt_vy])
            self.get_logger().warn("EKF initialized from GT (debug).")
            return True

        x0, y0 = ls_init_xy_from_ranges(self._meas.sensors_xy, z)
        self._ekf.initialize([x0, y0, 0.0, 0.0])
        self.get_logger().warn(f"EKF initialized from ranges (LS): x={x0:.2f}, y={y0:.2f}")
        return True

    def _on_z(self, msg: Float32MultiArray):
        z = np.asarray(msg.data, dtype=float).reshape((-1,))
        if z.shape[0] != self._meas.N:
            self.get_logger().error(f"z length={z.shape[0]} but sensors={self._meas.N}. layout mismatch!")
            return

        # INIT (once)
        skip_predict = False
        if not self._ekf.initialized:
            try:
                skip_predict = self._try_initialize(z)
            except Exception as e:
                self.get_logger().warn(f"Init failed, will retry: {e}")
                return

        # STEP
        try:
            self._ekf.step(z, skip_predict=skip_predict)
        except Exception as e:
            self.get_logger().error(f"EKF step failed: {e}")
            return

        # Publish estimated Odometry
        now = self.get_clock().now().to_msg()
        x = float(self._ekf.x[0, 0])
        y = float(self._ekf.x[1, 0])
        vx = float(self._ekf.x[2, 0])
        vy = float(self._ekf.x[3, 0])

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.w = 1.0
        od.twist.twist.linear.x = vx
        od.twist.twist.linear.y = vy

        self._pub_est.publish(od)


def main():
    rclpy.init()
    rclpy.spin(EKFRangeNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
