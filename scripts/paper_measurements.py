import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, Int32


class PaperMeasurementGenerator(Node):
    def __init__(self):
        super().__init__("paper_measurement_generator")

        self.gt_topic = self.declare_parameter("gt_topic", "/ground_truth/odom").value
        self.z_topic = self.declare_parameter("z_topic", "/paper/z").value
        self.y_topic = self.declare_parameter("y_topic", "/paper/y").value
        self.sum_topic = self.declare_parameter("sum_topic", "/paper/y_sum").value

        self.P0 = float(self.declare_parameter("P0", 1.0).value)
        self.sigma = float(self.declare_parameter("sigma", 0.05).value)
        self.gamma = float(self.declare_parameter("gamma", 0.5).value)

        self.N_side = int(self.declare_parameter("N_side", 5).value)  # 5 -> 25 sensors
        self.b = float(self.declare_parameter("b", 8.0).value)        # ROI: b x b

        self.rate = float(self.declare_parameter("rate", 10.0).value)  # Hz (Δ = 1/rate)

        seed = int(self.declare_parameter("seed", 0).value)
        self.rng = random.Random(seed)

        self.sensors = self._make_grid(self.N_side, self.b)
        self.N = len(self.sensors)
        self.get_logger().info(f"{self.N} sensors ready (grid {self.N_side}x{self.N_side})")

        self.gt_ready = False
        self.gt_x = 0.0
        self.gt_y = 0.0

        self.sub = self.create_subscription(Odometry, self.gt_topic, self.on_gt, qos_profile_sensor_data)
        self.pub_z = self.create_publisher(Float32MultiArray, self.z_topic, 10)
        self.pub_y = self.create_publisher(UInt8MultiArray, self.y_topic, 10)
        self.pub_sum = self.create_publisher(Int32, self.sum_topic, 10)

        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

    def _make_grid(self, n_side: int, b: float):
        if n_side <= 1:
            return [(0.0, 0.0)]
        half = 0.5 * b
        step = b / float(n_side - 1)
        xs = [-half + i * step for i in range(n_side)]
        ys = [-half + j * step for j in range(n_side)]
        return [(x, y) for y in ys for x in xs]  # row-major

    def on_gt(self, msg: Odometry):
        self.gt_x = msg.pose.pose.position.x
        self.gt_y = msg.pose.pose.position.y
        self.gt_ready = True

    def on_timer(self):
        if not self.gt_ready:
            return

        z = []
        y = []
        for (sx, sy) in self.sensors:
            dx = sx - self.gt_x
            dy = sy - self.gt_y
            d2 = dx * dx + dy * dy

            a = math.sqrt(self.P0 / (1.0 + d2))                 # Eq (4)
            zn = a + self.rng.gauss(0.0, self.sigma)            # Eq (5)
            yn = 1 if zn >= self.gamma else 0                   # Eq (6)

            z.append(float(zn))
            y.append(int(yn))

        msg_z = Float32MultiArray()
        msg_z.data = z
        self.pub_z.publish(msg_z)

        msg_y = UInt8MultiArray()
        msg_y.data = y
        self.pub_y.publish(msg_y)

        msg_s = Int32()
        msg_s.data = int(sum(y))
        self.pub_sum.publish(msg_s)


def main():
    rclpy.init()
    rclpy.spin(PaperMeasurementGenerator())
    rclpy.shutdown()


if __name__ == "__main__":
    main()