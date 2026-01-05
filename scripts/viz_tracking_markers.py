#!/usr/bin/env python3
"""
viz_tracking_markers.py

RViz MarkerArray visualization for:
  - sensors (from layout CSV)
  - ground-truth trail (Odometry)
  - estimator trail (Odometry)

Sub:
  - gt_topic  (nav_msgs/Odometry) default: /ground_truth/odom
  - est_topic (nav_msgs/Odometry) default: /tracking/estimated

Pub:
  - marker_topic (visualization_msgs/MarkerArray) default: /viz/markers

Params:
  - layout_file (required)
  - world_frame
  - max_points, pub_rate, min_step

Compatibility:
  - If you used an older param name "out_topic", it is still accepted as fallback.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


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


class TrackingViz(Node):
    def __init__(self) -> None:
        super().__init__("viz")

        # ---------------- Params ----------------
        self.world_frame: str = str(self.declare_parameter("world_frame", "world").value)

        self.layout_file: str = str(self.declare_parameter("layout_file", "").value)
        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")

        self.gt_topic: str = str(self.declare_parameter("gt_topic", "/ground_truth/odom").value)
        self.est_topic: str = str(self.declare_parameter("est_topic", "/tracking/estimated").value)

        # Canonical param:
        marker_topic_param = self.declare_parameter("marker_topic", "").value
        # Backward-compat alias:
        out_topic_param = self.declare_parameter("out_topic", "/viz/markers").value

        self.marker_topic: str = str(marker_topic_param).strip() if str(marker_topic_param).strip() else str(out_topic_param)

        self.max_points: int = int(self.declare_parameter("max_points", 3000).value)
        self.pub_rate: float = float(self.declare_parameter("pub_rate", 10.0).value)
        self.min_step: float = float(self.declare_parameter("min_step", 0.02).value)  # meters

        # ---------------- Layout ----------------
        self.sensors: List[Tuple[float, float]] = load_layout_csv(self.layout_file)
        if not self.sensors:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        # ---------------- State ----------------
        self.gt_xy: Optional[Tuple[float, float]] = None
        self.est_xy: Optional[Tuple[float, float]] = None
        self.gt_trail: Deque[Tuple[float, float]] = deque(maxlen=self.max_points)
        self.est_trail: Deque[Tuple[float, float]] = deque(maxlen=self.max_points)

        # ---------------- ROS I/O ----------------
        qos_markers = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(MarkerArray, self.marker_topic, qos_markers)

        self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.est_topic, self._on_est, qos_profile_sensor_data)

        self._timer = self.create_timer(1.0 / max(1e-6, self.pub_rate), self._on_timer)

        self.get_logger().info(
            f"Viz up. pub={self.marker_topic} sub(gt)={self.gt_topic} sub(est)={self.est_topic} "
            f"sensors={len(self.sensors)} max_points={self.max_points}"
        )

    def _append_if_moved(self, trail: Deque[Tuple[float, float]], x: float, y: float) -> None:
        if not trail:
            trail.append((x, y))
            return
        x0, y0 = trail[-1]
        if math.hypot(x - x0, y - y0) >= self.min_step:
            trail.append((x, y))

    def _on_gt(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self.gt_xy = (x, y)
        self._append_if_moved(self.gt_trail, x, y)

    def _on_est(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        self.est_xy = (x, y)
        self._append_if_moved(self.est_trail, x, y)

    def _sensor_markers(self, stamp) -> List[Marker]:
        arr: List[Marker] = []
        for i, (x, y) in enumerate(self.sensors):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.world_frame
            m.ns = "sensors"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.25
            m.pose.orientation.w = 1.0
            m.scale.x = 0.06
            m.scale.y = 0.06
            m.scale.z = 0.5
            m.color.r = 0.7
            m.color.g = 0.7
            m.color.b = 0.7
            m.color.a = 0.8
            arr.append(m)
        return arr

    def _sphere(self, stamp, ns: str, mid: int, xy: Tuple[float, float], r: float, g: float, b: float) -> Marker:
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.position.z = 0.10
        m.pose.orientation.w = 1.0
        m.scale.x = 0.30
        m.scale.y = 0.30
        m.scale.z = 0.30
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 1.0
        return m

    def _trail(self, stamp, ns: str, mid: int, trail: Deque[Tuple[float, float]], r: float, g: float, b: float) -> Marker:
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 0.95

        pts: List[Point] = []
        for (x, y) in trail:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            pts.append(p)
        m.points = pts
        return m

    def _on_timer(self) -> None:
        stamp = self.get_clock().now().to_msg()

        out = MarkerArray()
        out.markers.extend(self._sensor_markers(stamp))

        # GT (green)
        if self.gt_xy is not None:
            out.markers.append(self._sphere(stamp, "gt", 1000, self.gt_xy, 0.1, 0.9, 0.1))
            out.markers.append(self._trail(stamp, "gt_trail", 2000, self.gt_trail, 0.1, 0.9, 0.1))

        # EST (blue)
        if self.est_xy is not None:
            out.markers.append(self._sphere(stamp, "est", 1001, self.est_xy, 0.1, 0.3, 0.95))
            out.markers.append(self._trail(stamp, "est_trail", 2001, self.est_trail, 0.1, 0.3, 0.95))

        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackingViz()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
