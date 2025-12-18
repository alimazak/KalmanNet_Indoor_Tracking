import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class PoseArrayToOdom(Node):
    """
    /gz/dynamic_poses (PoseArray) -> /ground_truth/odom (Odometry)

    PoseArray isim taşımadığı için index seçiyoruz.
    auto_pick=True ise en çok hareket eden index'i otomatik seçer (genelde robot gövdesi index=0).
    """

    def __init__(self):
        super().__init__("gt_posearray_to_odom")

        self.pose_topic = self.declare_parameter("pose_topic", "/gz/dynamic_poses").value
        self.odom_topic = self.declare_parameter("odom_topic", "/ground_truth/odom").value
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.child_frame = self.declare_parameter("child_frame", "base_link").value

        self.auto_pick = bool(self.declare_parameter("auto_pick", True).value)
        self.manual_index = int(self.declare_parameter("index", 0).value)
        self.lock_after = int(self.declare_parameter("lock_after_msgs", 25).value)

        self.locked_index = None
        self._msg_count = 0
        self._prev_positions = None
        self._move_score = None

        self.prev_stamp = None
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None

        self.pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.sub = self.create_subscription(
            PoseArray, self.pose_topic, self.cb, qos_profile_sensor_data
        )

        self.get_logger().info(f"Listening {self.pose_topic} -> publishing {self.odom_topic}")

    def cb(self, msg: PoseArray):
        poses = msg.poses
        if not poses:
            return

        self._msg_count += 1

        if self.auto_pick:
            if self._prev_positions is None:
                self._prev_positions = [(p.position.x, p.position.y) for p in poses]
                self._move_score = [0.0] * len(poses)
            else:
                m = min(len(self._prev_positions), len(poses))
                for i in range(m):
                    x0, y0 = self._prev_positions[i]
                    x1, y1 = poses[i].position.x, poses[i].position.y
                    dx, dy = (x1 - x0), (y1 - y0)
                    self._move_score[i] += dx * dx + dy * dy
                    self._prev_positions[i] = (x1, y1)

            if self.locked_index is None and self._msg_count >= self.lock_after:
                self.locked_index = int(max(range(len(self._move_score)), key=lambda i: self._move_score[i]))
                self.get_logger().warn(f"AUTO PICK locked index = {self.locked_index} (movement score max)")

        idx = self.locked_index if (self.auto_pick and self.locked_index is not None) else self.manual_index
        idx = max(0, min(idx, len(poses) - 1))
        p = poses[idx]

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        x = p.position.x
        y = p.position.y
        q = p.orientation
        yaw = yaw_from_quat(q)

        vx = vy = wz = 0.0
        if self.prev_stamp is not None:
            dt = stamp - self.prev_stamp
            if dt > 1e-6:
                vx = (x - self.prev_x) / dt
                vy = (y - self.prev_y) / dt
                wz = wrap_pi(yaw - self.prev_yaw) / dt

        self.prev_stamp, self.prev_x, self.prev_y, self.prev_yaw = stamp, x, y, yaw

        od = Odometry()
        od.header = msg.header
        od.header.frame_id = self.world_frame
        od.child_frame_id = self.child_frame
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = p.position.z
        od.pose.pose.orientation = q
        od.twist.twist.linear.x = vx
        od.twist.twist.linear.y = vy
        od.twist.twist.angular.z = wz

        self.pub.publish(od)


def main():
    rclpy.init()
    rclpy.spin(PoseArrayToOdom())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
