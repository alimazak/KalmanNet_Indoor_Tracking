#!/usr/bin/env python3
"""
tracking_metrics_node.py

Computes tracking error metrics using:
  - Ground truth odom
  - Estimated odom

Sub:
  - gt_topic  (nav_msgs/Odometry) default: /ground_truth/odom
  - est_topic (nav_msgs/Odometry) default: /tracking/estimated

Pub:
  - error_topic        (std_msgs/Float32) instantaneous Euclidean error in XY
  - rmse_topic         (std_msgs/Float32) running RMSE
  - rmse_window_topic  (std_msgs/Float32) windowed RMSE over last N samples

Service:
  - reset_srv (std_srvs/Empty) clears metrics

Note:
  - Optional time gating: if |t_est - t_gt| > max_stamp_diff, skip sample.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_srvs.srv import Empty


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class TrackingMetrics(Node):
    def __init__(self) -> None:
        super().__init__("metrics")

        # ---------------- Params ----------------
        self.gt_topic: str = str(self.declare_parameter("gt_topic", "/ground_truth/odom").value)
        self.est_topic: str = str(self.declare_parameter("est_topic", "/tracking/estimated").value)

        self.error_topic: str = str(self.declare_parameter("error_topic", "/tracking/metrics/error").value)
        self.rmse_topic: str = str(self.declare_parameter("rmse_topic", "/tracking/metrics/rmse").value)
        self.rmse_window_topic: str = str(
            self.declare_parameter("rmse_window_topic", "/tracking/metrics/rmse_window").value
        )

        self.rmse_window_N: int = int(self.declare_parameter("rmse_window_N", 200).value)

        # If <=0 : disabled
        self.max_stamp_diff: float = float(self.declare_parameter("max_stamp_diff", 0.25).value)

        # Separate reset service (avoid conflicts with filter reset)
        self.reset_srv: str = str(self.declare_parameter("reset_srv", "/tracking/reset_metrics").value)

        # ---------------- State ----------------
        self._gt_xy_t: Optional[Tuple[float, float, float]] = None  # (x,y,t)
        self._err2_sum: float = 0.0
        self._count: int = 0

        self._win: Deque[float] = deque(maxlen=max(1, self.rmse_window_N))
        self._win_sum: float = 0.0

        # ---------------- ROS I/O ----------------
        self._pub_err = self.create_publisher(Float32, self.error_topic, 10)
        self._pub_rmse = self.create_publisher(Float32, self.rmse_topic, 10)
        self._pub_rmse_w = self.create_publisher(Float32, self.rmse_window_topic, 10)

        self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.est_topic, self._on_est, qos_profile_sensor_data)

        self.create_service(Empty, self.reset_srv, self._on_reset)

        self.get_logger().info(
            f"Metrics up. sub(gt)={self.gt_topic} sub(est)={self.est_topic} "
            f"pub(error)={self.error_topic} windowN={self.rmse_window_N} max_stamp_diff={self.max_stamp_diff}"
        )

    def _on_reset(self, req, resp):
        self._gt_xy_t = None
        self._err2_sum = 0.0
        self._count = 0
        self._win.clear()
        self._win_sum = 0.0
        self.get_logger().warn("METRICS RESET: cleared.")
        return resp

    def _on_gt(self, msg: Odometry) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self._gt_xy_t = (x, y, t)

    def _on_est(self, msg: Odometry) -> None:
        if self._gt_xy_t is None:
            return

        gt_x, gt_y, gt_t = self._gt_xy_t
        est_t = _stamp_to_sec(msg.header.stamp)

        if self.max_stamp_diff > 0.0:
            if abs(est_t - gt_t) > self.max_stamp_diff:
                return

        ex = float(msg.pose.pose.position.x) - gt_x
        ey = float(msg.pose.pose.position.y) - gt_y
        e = math.hypot(ex, ey)
        e2 = e * e

        # running RMSE
        self._err2_sum += e2
        self._count += 1
        rmse = math.sqrt(self._err2_sum / max(1, self._count))

        # windowed RMSE
        if len(self._win) == self._win.maxlen:
            self._win_sum -= self._win[0]
        self._win.append(e2)
        self._win_sum += e2
        rmse_w = math.sqrt(self._win_sum / max(1, len(self._win)))

        self._pub_err.publish(Float32(data=float(e)))
        self._pub_rmse.publish(Float32(data=float(rmse)))
        self._pub_rmse_w.publish(Float32(data=float(rmse_w)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackingMetrics()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
