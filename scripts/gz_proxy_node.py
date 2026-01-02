#!/usr/bin/env python3
"""
gz_proxy_node.py

Moves a Gazebo model (proxy) to the estimated pose so you can see estimator trail in Gazebo.

Sub:
  - est_topic (Odometry) default: /tracking/estimated

Srv client:
  - /world/<gz_world>/set_pose
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity


class GazeboProxyNode(Node):
    def __init__(self):
        super().__init__("gz_proxy_node")

        self.est_topic = self.declare_parameter("est_topic", "/tracking/estimated").value
        self.gz_world = self.declare_parameter("gz_world", "empty_world").value
        self.gz_entity = self.declare_parameter("gz_entity", "ekf_proxy").value
        self.gz_z = float(self.declare_parameter("gz_z", 0.01).value)

        self._cli = self.create_client(SetEntityPose, f"/world/{self.gz_world}/set_pose")
        self._pending = False

        self.create_subscription(Odometry, self.est_topic, self._on_est, qos_profile_sensor_data)

        self.get_logger().info(
            f"GZ proxy up. sub={self.est_topic} -> /world/{self.gz_world}/set_pose (entity={self.gz_entity})"
        )

    def _on_est(self, msg: Odometry):
        if self._pending:
            return
        if not self._cli.service_is_ready():
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        req = SetEntityPose.Request()
        req.entity.name = self.gz_entity
        if hasattr(req.entity, "type"):
            req.entity.type = Entity.MODEL

        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(self.gz_z)
        req.pose.orientation.w = 1.0

        self._pending = True
        fut = self._cli.call_async(req)
        fut.add_done_callback(self._on_done)

    def _on_done(self, fut):
        self._pending = False
        try:
            fut.result()
        except Exception as e:
            self.get_logger().warn(f"set_pose failed: {e}")


def main():
    rclpy.init()
    rclpy.spin(GazeboProxyNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()