#!/usr/bin/env python3
"""
gt_posearray_to_odom.py

Gazebo/ros_gz_bridge PoseArray -> nav_msgs/Odometry ground-truth.

Sub:
  - pose_topic (geometry_msgs/PoseArray) default: /gz/dynamic_poses

Pub:
  - odom_topic (nav_msgs/Odometry) default: /ground_truth/odom

PoseArray entity adı taşımadığı için index seçer.

Modes:
  - auto_pick=True : lock_after_msgs kadar mesajdan sonra "en çok hareket eden" index'i kilitler.
  - auto_pick=False: index parametresiyle manuel seçim.

Twist (vx,vy,wz) finite-difference ile hesaplanır (GT metrics / dataset için faydalı).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, Quaternion
from nav_msgs.msg import Odometry


def yaw_from_quat(q: Quaternion) -> float:
    """Yaw (Z) from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def stamp_to_float_sec(stamp) -> float:
    """builtin_interfaces/Time -> float seconds."""
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class PoseArrayToOdom(Node):
    """
    Converts PoseArray to Odometry.
    """

    def __init__(self) -> None:
        super().__init__("gt_posearray_to_odom")

        # ---------------- Params ----------------
        self.pose_topic: str = str(self.declare_parameter("pose_topic", "/gz/dynamic_poses").value)
        self.odom_topic: str = str(self.declare_parameter("odom_topic", "/ground_truth/odom").value)

        self.world_frame: str = str(self.declare_parameter("world_frame", "world").value)
        self.child_frame: str = str(self.declare_parameter("child_frame", "base_link").value)

        self.auto_pick: bool = bool(self.declare_parameter("auto_pick", True).value)
        self.manual_index: int = int(self.declare_parameter("index", 0).value)
        self.lock_after_msgs: int = int(self.declare_parameter("lock_after_msgs", 25).value)

        # ---------------- State ----------------
        self._locked_index: Optional[int] = None

        # For auto_pick scoring
        self._msg_count: int = 0
        self._prev_positions: Optional[List[Tuple[float, float]]] = None
        self._move_score: Optional[List[float]] = None

        # For finite-diff twist
        self._prev_t: Optional[float] = None
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._prev_yaw: float = 0.0

        # ---------------- ROS I/O ----------------
        self._pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.create_subscription(PoseArray, self.pose_topic, self._on_posearray, qos_profile_sensor_data)

        self.get_logger().info(
            f"GT PoseArray->Odom: sub={self.pose_topic} pub={self.odom_topic} "
            f"(auto_pick={self.auto_pick}, index={self.manual_index}, lock_after_msgs={self.lock_after_msgs})"
        )

    # ---------------- Helpers ----------------
    @staticmethod
    def _clamp_index(i: int, n: int) -> int:
        if n <= 0:
            return 0
        return max(0, min(int(i), n - 1))

    def _reset_autopick(self, poses) -> None:
        self._msg_count = 0
        self._locked_index = None
        self._prev_positions = [(p.position.x, p.position.y) for p in poses]
        self._move_score = [0.0] * len(poses)

    def _update_autopick(self, poses) -> None:
        """Update movement scores until lock."""
        if self._prev_positions is None or self._move_score is None:
            self._reset_autopick(poses)
            return

        # PoseArray length changed => reset tracker (index mapping may have changed)
        if len(self._prev_positions) != len(poses) or len(self._move_score) != len(poses):
            self._reset_autopick(poses)
            return

        m = len(poses)
        for i in range(m):
            x0, y0 = self._prev_positions[i]
            x1, y1 = poses[i].position.x, poses[i].position.y
            dx, dy = (x1 - x0), (y1 - y0)
            self._move_score[i] += dx * dx + dy * dy
            self._prev_positions[i] = (x1, y1)

        # Lock if enough msgs collected
        if self._locked_index is None and self._msg_count >= self.lock_after_msgs:
            self._locked_index = int(max(range(m), key=lambda k: self._move_score[k]))
            self.get_logger().warn(f"AUTO PICK locked index = {self._locked_index} (max movement score)")

    def _select_index(self, poses) -> int:
        n = len(poses)
        if n == 0:
            return 0

        if self.auto_pick and (self._locked_index is not None):
            return self._clamp_index(self._locked_index, n)

        return self._clamp_index(self.manual_index, n)

    # ---------------- Callback ----------------
    def _on_posearray(self, msg: PoseArray) -> None:
        poses = msg.poses
        if not poses:
            return

        # Time (prefer incoming stamp; fallback to node clock if stamp is 0)
        t = stamp_to_float_sec(msg.header.stamp)
        stamp_msg = msg.header.stamp
        if t <= 0.0:
            now = self.get_clock().now()
            t = now.nanoseconds * 1e-9
            stamp_msg = now.to_msg()

        # Auto pick update
        if self.auto_pick and self._locked_index is None:
            if self._prev_positions is None:
                self._reset_autopick(poses)
            self._msg_count += 1
            self._update_autopick(poses)

        idx = self._select_index(poses)
        p = poses[idx]

        x = float(p.position.x)
        y = float(p.position.y)
        q = p.orientation
        yaw = yaw_from_quat(q)

        vx = vy = wz = 0.0
        if self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-6:
                vx = (x - self._prev_x) / dt
                vy = (y - self._prev_y) / dt
                wz = wrap_pi(yaw - self._prev_yaw) / dt
            else:
                # dt <= 0 => don't produce garbage twist
                vx = vy = wz = 0.0

        self._prev_t = t
        self._prev_x = x
        self._prev_y = y
        self._prev_yaw = yaw

        od = Odometry()
        od.header.stamp = stamp_msg
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame

        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = float(p.position.z)
        od.pose.pose.orientation = q

        od.twist.twist.linear.x = float(vx)
        od.twist.twist.linear.y = float(vy)
        od.twist.twist.angular.z = float(wz)

        self._pub.publish(od)


def main() -> None:
    rclpy.init()
    rclpy.spin(PoseArrayToOdom())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
