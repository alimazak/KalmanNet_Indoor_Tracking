#!/usr/bin/env python3
"""
ekf_tracker_from_range.py

Minimal EKF ROS2 node (one job only):
  - Subscribes: z_topic (Float32MultiArray)
  - Publishes : est_topic (nav_msgs/Odometry)
  - Service   : reset_srv (std_srvs/Empty)

No RMSE, no Path, no Gazebo proxy, no visualization inside this node.

Namespace-friendly defaults:
  z_topic   = "z"
  est_topic = "estimated"
  reset_srv = "reset"
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

# Works both as:
#  - scripts/ usage (local file)
#  - packaged usage (tracking_filters.core.*)
try:
    from tracking_filters.core.ekf_range_core import (
        ConstantVelocityModel,
        RangeMeasurementModel,
        RangeEKF,
        ls_init_xy_from_ranges,
    )
except Exception:  # pragma: no cover
    from ekf_range_core import (  # type: ignore
        ConstantVelocityModel,
        RangeMeasurementModel,
        RangeEKF,
        ls_init_xy_from_ranges,
    )


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
        def p(name: str, default):
            return self.declare_parameter(name, default).value

        self.layout_file = str(p("layout_file", ""))
        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")
        if not Path(self.layout_file).is_file():
            raise RuntimeError(f"layout_file not found: {self.layout_file}")

        # I/O topics (relative by default => respects namespace)
        self.z_topic = str(p("z_topic", "z"))
        self.est_topic = str(p("est_topic", "estimated"))

        # Frames
        self.world_frame = str(p("world_frame", "world"))
        self.child_frame = str(p("child_frame", "ekf_base"))

        # Noise/model
        self.sigma = float(p("sigma", 0.10))
        self.dt = float(p("delta", 0.1))
        self.tau = float(p("tau", 1.0))

        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if self.dt <= 0.0:
            raise ValueError("delta(dt) must be > 0")

        # Initial covariance prior
        self.init_pos_std = float(p("init_pos_std", 5.0))
        self.init_vel_std = float(p("init_vel_std", 2.0))

        # Optional debug init from GT (OFF by default)
        self.init_from_gt = bool(p("init_from_gt", False))
        self.gt_topic = str(p("gt_topic", "gt/odom"))

        # Reset service name (relative by default)
        self.reset_srv = str(p("reset_srv", "reset"))

        # Optional: publish covariance into Odometry
        self.publish_covariance = bool(p("publish_covariance", False))

        # ---------------- Layout / models ----------------
        sensors_xy = load_layout_csv(self.layout_file)
        if not sensors_xy:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        self._meas = RangeMeasurementModel(sensors_xy)
        self._proc = ConstantVelocityModel(self.dt, self.tau)

        R = (self.sigma**2) * np.eye(self._meas.N, dtype=float)
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
            self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)

        self.create_service(Empty, self.reset_srv, self._on_reset)

        self.get_logger().info(
            "EKF(range) ready | "
            f"N={self._meas.N} | sub='{self.z_topic}' | pub='{self.est_topic}' | "
            f"ns='{self.get_namespace()}' | init_from_gt={self.init_from_gt}"
        )

    # ---------------- Helpers ----------------
    @staticmethod
    def _fill_covariances(od: Odometry, P: np.ndarray) -> None:
        """
        Fill Odometry covariance (pose & twist) from EKF P.
        Only x,y and vx,vy blocks are populated; others stay 0.
        Odometry cov arrays are 6x6 flattened row-major.
        """
        P = np.asarray(P, dtype=float)
        if P.shape != (4, 4):
            return

        # Pose covariance: x,y -> indices (0,0)=0, (0,1)=1, (1,0)=6, (1,1)=7
        od.pose.covariance[0] = float(P[0, 0])
        od.pose.covariance[1] = float(P[0, 1])
        od.pose.covariance[6] = float(P[1, 0])
        od.pose.covariance[7] = float(P[1, 1])

        # Twist covariance: vx,vy -> same index pattern
        od.twist.covariance[0] = float(P[2, 2])
        od.twist.covariance[1] = float(P[2, 3])
        od.twist.covariance[6] = float(P[3, 2])
        od.twist.covariance[7] = float(P[3, 3])

    # ---------------- Callbacks ----------------
    def _on_reset(self, req, resp):
        self._ekf.reset()
        self.get_logger().warn("EKF RESET: state/cov cleared.")
        return resp

    def _on_gt(self, msg: Odometry):
        self._gt_x = msg.pose.pose.position.x
        self._gt_y = msg.pose.pose.position.y
        self._gt_vx = msg.twist.twist.linear.x
        self._gt_vy = msg.twist.twist.linear.y
        self._gt_ready = True

    def _try_initialize(self, z: np.ndarray) -> bool:
        """
        Returns True if filter got initialized.
        Caller should skip_predict on the first step after init.
        """
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
        if not np.all(np.isfinite(z)):
            self.get_logger().warn("z contains NaN/inf, skipping this sample.")
            return

        # INIT (once)
        skip_predict = False
        if not self._ekf.initialized:
            try:
                did_init = self._try_initialize(z)
            except Exception as e:
                self.get_logger().warn(f"Init failed, will retry: {e}")
                return

            # If init_from_gt=True and GT not ready yet => wait
            if not did_init:
                return

            skip_predict = True

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

        if self.publish_covariance:
            self._fill_covariances(od, self._ekf.P)

        self._pub_est.publish(od)


def main():
    rclpy.init()
    rclpy.spin(EKFRangeNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
