import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Float32


def load_layout_csv(path: str):
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


class RangeMeasurementGenerator(Node):
    """
    Range model (hocanın istediği):
      z_i,k = d_i,k + v_i,k
      d_i,k = sqrt((x-x_i)^2 + (y-y_i)^2)
      v_i,k ~ N(0, sigma^2)
    Publishes:
      /range/z : Float32MultiArray (N distances)
      /range/min : Float32 (debug)
    """

    def __init__(self):
        super().__init__("range_measurement_generator")

        self.gt_topic = self.declare_parameter("gt_topic", "/ground_truth/odom").value
        self.z_topic = self.declare_parameter("z_topic", "/range/z").value
        self.min_topic = self.declare_parameter("min_topic", "/range/min").value

        self.layout_file = self.declare_parameter("layout_file", "").value
        self.sigma = float(self.declare_parameter("sigma", 0.10).value)  # meters
        self.rate = float(self.declare_parameter("rate", 10.0).value)    # Hz

        seed = int(self.declare_parameter("seed", 0).value)
        self.rng = random.Random(seed)

        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")

        self.sensors = load_layout_csv(self.layout_file)
        if not self.sensors:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        self.N = len(self.sensors)
        self.get_logger().info(f"Loaded {self.N} sensors from {self.layout_file}")

        self.gt_ready = False
        self.gt_x = 0.0
        self.gt_y = 0.0

        self.sub = self.create_subscription(Odometry, self.gt_topic, self.on_gt, qos_profile_sensor_data)
        self.pub_z = self.create_publisher(Float32MultiArray, self.z_topic, 10)
        self.pub_min = self.create_publisher(Float32, self.min_topic, 10)

        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

    def on_gt(self, msg: Odometry):
        self.gt_x = msg.pose.pose.position.x
        self.gt_y = msg.pose.pose.position.y
        self.gt_ready = True

    def on_timer(self):
        if not self.gt_ready:
            return

        z = []
        dmin = None
        x = self.gt_x
        y = self.gt_y

        for (sx, sy) in self.sensors:
            dx = x - sx
            dy = y - sy
            d = math.sqrt(dx*dx + dy*dy)
            zn = d + self.rng.gauss(0.0, self.sigma)
            z.append(float(zn))
            if dmin is None or d < dmin:
                dmin = d

        msg = Float32MultiArray()
        msg.data = z
        self.pub_z.publish(msg)

        if dmin is not None:
            self.pub_min.publish(Float32(data=float(dmin)))


def main():
    rclpy.init()
    rclpy.spin(RangeMeasurementGenerator())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
