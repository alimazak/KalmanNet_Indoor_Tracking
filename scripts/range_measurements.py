#!/usr/bin/env python3
"""
range_measurements.py

Range measurement generator.

Model (hocanın istediği):
  z_i,k = d_i,k + v_i,k
  d_i,k = sqrt((x-x_i)^2 + (y-y_i)^2)
  v_i,k ~ N(0, sigma^2)

Publishes:
  z_topic   (Float32MultiArray): range list (N)
  min_topic (Float32): min true distance to anchors (debug)

Namespace-friendly defaults (relative names):
  gt_topic  = "gt/odom"
  z_topic   = "z"
  min_topic = "range/min"
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray


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


class RangeMeasurementGenerator(Node):
    def __init__(self):
        super().__init__("range_measurement_generator")

        def p(name: str, default):
            return self.declare_parameter(name, default).value

        # Topics (relative by default => respects namespace)
        self.gt_topic = str(p("gt_topic", "gt/odom"))
        self.z_topic = str(p("z_topic", "z"))
        self.min_topic = str(p("min_topic", "range/min"))

        # Layout + noise
        self.layout_file = str(p("layout_file", ""))
        self.sigma = float(p("sigma", 0.10))  # meters
        self.rate = float(p("rate", 10.0))    # Hz

        # Reproducibility
        seed = int(p("seed", 0))
        self.rng = random.Random(seed)

        # Clamp negative ranges (recommended)
        self.clip_ranges = bool(p("clip_ranges", True))
        self.min_range = float(p("min_range", 1e-3))

        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")
        if not Path(self.layout_file).is_file():
            raise RuntimeError(f"layout_file not found: {self.layout_file}")

        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if self.rate <= 0.0:
            raise ValueError("rate must be > 0")
        if self.min_range <= 0.0:
            raise ValueError("min_range must be > 0")

        self.sensors = load_layout_csv(self.layout_file)
        if not self.sensors:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        self.N = len(self.sensors)
        self.get_logger().info(f"Loaded {self.N} sensors from {self.layout_file} | sigma={self.sigma} | rate={self.rate}Hz")

        # GT cache
        self.gt_ready = False
        self.gt_x = 0.0
        self.gt_y = 0.0

        # ROS I/O
        self.create_subscription(Odometry, self.gt_topic, self._on_gt, qos_profile_sensor_data)
        self.pub_z = self.create_publisher(Float32MultiArray, self.z_topic, 10)
        self.pub_min = self.create_publisher(Float32, self.min_topic, 10)

        self.timer = self.create_timer(1.0 / self.rate, self._on_timer)

    def _on_gt(self, msg: Odometry):
        self.gt_x = msg.pose.pose.position.x
        self.gt_y = msg.pose.pose.position.y
        self.gt_ready = True

    def _on_timer(self):
        if not self.gt_ready:
            return

        x = self.gt_x
        y = self.gt_y

        z_list: List[float] = []
        dmin = float("inf")

        for (sx, sy) in self.sensors:
            dx = x - sx
            dy = y - sy
            d = math.hypot(dx, dy)  # true distance
            dmin = min(dmin, d)

            zn = d + self.rng.gauss(0.0, self.sigma)
            if self.clip_ranges:
                zn = max(float(self.min_range), float(zn))

            z_list.append(float(zn))

        msg = Float32MultiArray()
        msg.data = z_list
        self.pub_z.publish(msg)

        if math.isfinite(dmin):
            self.pub_min.publish(Float32(data=float(dmin)))


def main():
    rclpy.init()
    rclpy.spin(RangeMeasurementGenerator())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
