from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


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


class TrackingViz(Node):
    def __init__(self):
        super().__init__("tracking_viz_markers")

        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.layout_file = self.declare_parameter("layout_file", "").value

        self.gt_topic = self.declare_parameter("gt_topic", "/ground_truth/odom").value
        self.est_topic = self.declare_parameter("est_topic", "/tracking/odom").value

        self.out_topic = self.declare_parameter("out_topic", "/viz/markers").value

        self.max_points = int(self.declare_parameter("max_points", 2000).value)
        self.pub_rate = float(self.declare_parameter("pub_rate", 10.0).value)

        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu.")
        self.sensors = load_layout_csv(self.layout_file)
        if not self.sensors:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        self.gt_xy = None
        self.est_xy = None

        self.gt_trail = deque(maxlen=self.max_points)
        self.est_trail = deque(maxlen=self.max_points)

        qos_mark = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(MarkerArray, self.out_topic, qos_mark)

        self.sub_gt = self.create_subscription(Odometry, self.gt_topic, self.on_gt, 10)
        self.sub_est = self.create_subscription(Odometry, self.est_topic, self.on_est, 10)

        self.timer = self.create_timer(1.0 / self.pub_rate, self.on_timer)

        self.get_logger().info(f"Publishing markers: {self.out_topic}")

    def on_gt(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.gt_xy = (x, y)
        self.gt_trail.append((x, y))

    def on_est(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.est_xy = (x, y)
        self.est_trail.append((x, y))

    def _sensor_markers(self, stamp):
        arr = []
        for i, (x, y) in enumerate(self.sensors):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.world_frame
            m.ns = "sensors"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
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

    def _sphere(self, stamp, ns, mid, xy, r, g, b):
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
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        return m

    def _trail(self, stamp, ns, mid, trail, r, g, b):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 0.95
        pts = []
        for (x, y) in trail:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            pts.append(p)
        m.points = pts
        return m

    def on_timer(self):
        stamp = self.get_clock().now().to_msg()
        out = MarkerArray()
        out.markers.extend(self._sensor_markers(stamp))

        if self.gt_xy is not None:
            out.markers.append(self._sphere(stamp, "gt", 1000, self.gt_xy, 0.1, 0.9, 0.1))
            out.markers.append(self._trail(stamp, "gt_trail", 2000, self.gt_trail, 0.1, 0.9, 0.1))

        if self.est_xy is not None:
            out.markers.append(self._sphere(stamp, "ekf", 1001, self.est_xy, 0.95, 0.1, 0.1))
            out.markers.append(self._trail(stamp, "ekf_trail", 2001, self.est_trail, 0.95, 0.1, 0.1))

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(TrackingViz())
    rclpy.shutdown()


if __name__ == "__main__":
    main()