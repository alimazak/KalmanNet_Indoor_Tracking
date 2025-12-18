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
        self.ekf_topic = self.declare_parameter("ekf_topic", "/tracking/odom").value

        self.out_topic = self.declare_parameter("out_topic", "/viz/markers").value

        # sensor marker geometry
        self.sensor_radius = float(self.declare_parameter("sensor_radius", 0.03).value)
        self.sensor_height = float(self.declare_parameter("sensor_height", 0.5).value)

        # trails
        self.max_points = int(self.declare_parameter("max_points", 2000).value)
        self.pub_rate = float(self.declare_parameter("pub_rate", 10.0).value)

        if not self.layout_file:
            raise RuntimeError("layout_file zorunlu. Örn: -p layout_file:=.../paper_sensors_5x5_b20.csv")

        self.sensors = load_layout_csv(self.layout_file)
        if not self.sensors:
            raise RuntimeError(f"layout_file okunamadı/boş: {self.layout_file}")

        # cache latest poses
        self.gt_xy = None
        self.ekf_xy = None

        # trail buffers
        self.gt_trail = deque(maxlen=self.max_points)
        self.ekf_trail = deque(maxlen=self.max_points)

        # QoS: markers latch gibi davransın (RViz sonradan açılırsa da görsün)
        qos_mark = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(MarkerArray, self.out_topic, qos_mark)

        self.sub_gt = self.create_subscription(Odometry, self.gt_topic, self.on_gt, 10)
        self.sub_ekf = self.create_subscription(Odometry, self.ekf_topic, self.on_ekf, 10)

        self.timer = self.create_timer(1.0 / self.pub_rate, self.on_timer)

        self.get_logger().info(f"Publishing MarkerArray on {self.out_topic}")
        self.get_logger().info(f"Sensors: {len(self.sensors)} from {self.layout_file}")

    def on_gt(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.gt_xy = (x, y)
        self.gt_trail.append((x, y))

    def on_ekf(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ekf_xy = (x, y)
        self.ekf_trail.append((x, y))

    def _make_sensor_markers(self, stamp):
        arr = []
        z = self.sensor_height / 2.0

        for i, (x, y) in enumerate(self.sensors):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.world_frame
            m.ns = "paper_sensors"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = 2.0 * self.sensor_radius
            m.scale.y = 2.0 * self.sensor_radius
            m.scale.z = self.sensor_height
            # grey
            m.color.r = 0.7
            m.color.g = 0.7
            m.color.b = 0.7
            m.color.a = 0.8
            arr.append(m)
        return arr

    def _make_point_marker(self, stamp, ns, mid, xy, r, g, b, scale=0.25, z=0.10):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        return m

    def _make_trail_marker(self, stamp, ns, mid, trail, r, g, b, width=0.06, z=0.05):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 0.95
        pts = []
        for (x, y) in trail:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            pts.append(p)
        m.points = pts
        return m

    def on_timer(self):
        stamp = self.get_clock().now().to_msg()
        out = MarkerArray()

        # sensors
        out.markers.extend(self._make_sensor_markers(stamp))

        # GT + EKF imleç
        if self.gt_xy is not None:
            out.markers.append(self._make_point_marker(stamp, "gt", 1000, self.gt_xy, 0.1, 0.9, 0.1, scale=0.28))
            out.markers.append(self._make_trail_marker(stamp, "gt_trail", 2000, self.gt_trail, 0.1, 0.9, 0.1))

        if self.ekf_xy is not None:
            out.markers.append(self._make_point_marker(stamp, "ekf", 1001, self.ekf_xy, 0.95, 0.1, 0.1, scale=0.28))
            out.markers.append(self._make_trail_marker(stamp, "ekf_trail", 2001, self.ekf_trail, 0.95, 0.1, 0.1))

        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(TrackingViz())
    rclpy.shutdown()


if __name__ == "__main__":
    main()