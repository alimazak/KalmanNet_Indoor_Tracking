#!/usr/bin/env python3
"""
gz_proxy_node.py

Moves a Gazebo model (proxy) to the estimated pose so you can see estimator trail in Gazebo.

Sub:
  - est_topic (nav_msgs/Odometry) default: /tracking/estimated (veya namespace içinde "estimated")

Service client:
  - /world/<gz_world>/set_pose  (ros_gz_interfaces/srv/SetEntityPose)

Notes:
  - Subscription sadece "son pozu" cache'ler.
  - Timer (rate Hz) ile set_pose çağrısı yapılır -> flood olmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose


@dataclass
class _Pose:
    x: float
    y: float
    qx: float
    qy: float
    qz: float
    qw: float


class GazeboProxyNode(Node):
    def __init__(self) -> None:
        super().__init__("gz_proxy")

        # --- Params ---
        self.est_topic: str = str(self.declare_parameter("est_topic", "/tracking/estimated").value)
        self.gz_world: str = str(self.declare_parameter("gz_world", "empty_world").value)
        self.gz_entity: str = str(self.declare_parameter("gz_entity", "ekf_proxy").value)
        self.gz_z: float = float(self.declare_parameter("gz_z", 0.01).value)

        # set_pose call rate (Hz) - limiter
        self.rate_hz: float = float(self.declare_parameter("rate_hz", 10.0).value)

        # If true, use odom orientation; else force identity quaternion
        self.use_est_orientation: bool = bool(self.declare_parameter("use_est_orientation", False).value)

        self._srv_name = f"/world/{self.gz_world}/set_pose"
        self._cli = self.create_client(SetEntityPose, self._srv_name)

        self._pending: bool = False
        self._last_pose: Optional[_Pose] = None

        # simple log throttles (sim-time aware)
        self._last_wait_log_ns: int = 0
        self._last_fail_log_ns: int = 0

        # --- ROS I/O ---
        self.create_subscription(Odometry, self.est_topic, self._on_est, qos_profile_sensor_data)

        period = 1.0 / max(self.rate_hz, 1e-6)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"GZ proxy up: sub={self.est_topic} -> srv={self._srv_name} entity={self.gz_entity} "
            f"(rate={self.rate_hz:.1f}Hz, use_est_orientation={self.use_est_orientation})"
        )

    def _throttle_ok(self, last_ns: int, every_sec: float) -> bool:
        now_ns = int(self.get_clock().now().nanoseconds)
        if now_ns - last_ns >= int(every_sec * 1e9):
            return True
        return False

    def _on_est(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        if self.use_est_orientation:
            self._last_pose = _Pose(
                x=float(p.x),
                y=float(p.y),
                qx=float(q.x),
                qy=float(q.y),
                qz=float(q.z),
                qw=float(q.w),
            )
        else:
            self._last_pose = _Pose(
                x=float(p.x),
                y=float(p.y),
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
            )

    def _on_timer(self) -> None:
        if self._last_pose is None:
            return

        if self._pending:
            return

        if not self._cli.service_is_ready():
            now_ns = int(self.get_clock().now().nanoseconds)
            if self._throttle_ok(self._last_wait_log_ns, every_sec=5.0):
                self._last_wait_log_ns = now_ns
                self.get_logger().warn(f"Waiting for service: {self._srv_name}")
            return

        pose = self._last_pose

        req = SetEntityPose.Request()
        req.entity.name = self.gz_entity
        # type field exists in newer ros_gz_interfaces; safe-guard anyway
        try:
            req.entity.type = Entity.MODEL
        except Exception:
            pass

        req.pose.position.x = float(pose.x)
        req.pose.position.y = float(pose.y)
        req.pose.position.z = float(self.gz_z)

        req.pose.orientation.x = float(pose.qx)
        req.pose.orientation.y = float(pose.qy)
        req.pose.orientation.z = float(pose.qz)
        req.pose.orientation.w = float(pose.qw)

        self._pending = True
        fut = self._cli.call_async(req)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        self._pending = False
        try:
            fut.result()
        except Exception as e:
            now_ns = int(self.get_clock().now().nanoseconds)
            if self._throttle_ok(self._last_fail_log_ns, every_sec=2.0):
                self._last_fail_log_ns = now_ns
                self.get_logger().warn(f"set_pose failed: {e}")


def main() -> None:
    rclpy.init()
    rclpy.spin(GazeboProxyNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
