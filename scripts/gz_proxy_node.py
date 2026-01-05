#!/usr/bin/env python3
"""
gz_proxy_node.py

Moves a Gazebo model (proxy) to the estimated pose so you can see estimator trail in Gazebo.

Sub:
  - est_topic (nav_msgs/Odometry) default: /tracking/estimated

Srv client:
  - /world/<gz_world>/set_pose  (ros_gz_interfaces/srv/SetEntityPose)

Notes:
  - If a request is in-flight, we keep only the latest pose and send it next
    (so we don't flood the service but we also don't lag too much).
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
class _Pose2D:
    x: float
    y: float
    qx: float
    qy: float
    qz: float
    qw: float


class GazeboProxyNode(Node):
    def __init__(self) -> None:
        super().__init__("gz_proxy")

        # ---------------- Params ----------------
        self.est_topic: str = str(self.declare_parameter("est_topic", "/tracking/estimated").value)
        self.gz_world: str = str(self.declare_parameter("gz_world", "empty_world").value)
        self.gz_entity: str = str(self.declare_parameter("gz_entity", "ekf_proxy").value)
        self.gz_z: float = float(self.declare_parameter("gz_z", 0.01).value)

        self._srv_name = f"/world/{self.gz_world}/set_pose"
        self._cli = self.create_client(SetEntityPose, self._srv_name)

        # ---------------- State ----------------
        self._pending: bool = False
        self._dirty: bool = False
        self._latest: Optional[_Pose2D] = None

        # ---------------- ROS I/O ----------------
        self.create_subscription(Odometry, self.est_topic, self._on_est, qos_profile_sensor_data)

        self.get_logger().info(
            f"GZ proxy up. sub={self.est_topic} -> srv={self._srv_name} (entity={self.gz_entity}, z={self.gz_z})"
        )

    def _on_est(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self._latest = _Pose2D(
            x=float(p.x),
            y=float(p.y),
            qx=float(q.x),
            qy=float(q.y),
            qz=float(q.z),
            qw=float(q.w) if (q.w != 0.0 or q.x != 0.0 or q.y != 0.0 or q.z != 0.0) else 1.0,
        )

        if self._pending:
            # request in-flight -> mark dirty and return
            self._dirty = True
            return

        # no pending request -> try send immediately
        self._try_send_latest()

    def _try_send_latest(self) -> None:
        if self._latest is None:
            return
        if not self._cli.service_is_ready():
            return
        if self._pending:
            return

        pose = self._latest

        req = SetEntityPose.Request()
        req.entity.name = self.gz_entity
        # Some distros expose entity.type
        if hasattr(req.entity, "type"):
            req.entity.type = Entity.MODEL

        req.pose.position.x = pose.x
        req.pose.position.y = pose.y
        req.pose.position.z = float(self.gz_z)

        # Forward orientation (even if estimator always outputs identity, this is safe)
        req.pose.orientation.x = pose.qx
        req.pose.orientation.y = pose.qy
        req.pose.orientation.z = pose.qz
        req.pose.orientation.w = pose.qw

        self._pending = True
        self._dirty = False
        fut = self._cli.call_async(req)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut) -> None:
        self._pending = False
        try:
            fut.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"set_pose failed: {e}")

        # If new pose arrived while pending, send latest now.
        if self._dirty:
            self._try_send_latest()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GazeboProxyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
