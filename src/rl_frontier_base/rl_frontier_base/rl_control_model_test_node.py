import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from .model_exploration import RLFrontierBaseModel
import threading
import torch


from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from .world_controller import WorldManager
import threading
import numpy as np
from skimage.transform import resize
import torch
import math


class RLFrontierBaseTrain(Node):
    def __init__(self):
        super().__init__("rl_control_test_node")
        self.create_subscription(OccupancyGrid, "map", self.map_callback, 10)
        self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)

        self.scan_data = None
        self.map_data = None
        self.odom_data = None

        self.previous_twist = None

        # Start exploration in a new thread
        thread = threading.Thread(target=self.exploration)
        thread.start()

    def exploration(self):
        self.model = RLFrontierBaseModel()
        self.model.load_state_dict(
            torch.load("/home/berkay/Desktop/rl_frontier_base/src/rl_frontier_base/rl_frontier_base/model/model.pth")
        )
        while rclpy.ok():
            if self.map_data is None or self.odom_data is None or self.scan_data is None:
                self.get_logger().info("Waiting for data...")
                continue

            # run the robot
            final_move = [0, 0, 0, 0]
            print("AI~")
            state = self.getState()

            state0 = torch.tensor(state[0], dtype=torch.float32)
            state1 = torch.tensor(state[1], dtype=torch.float32)
            prediction = self.model(state0, state1)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

            self.move_robot(final_move)

    def getState(self):
        originX = self.get_originX()
        originY = self.get_originY()
        xPosition = self.get_Xposition()
        yPosition = self.get_Yposition()
        scan_ranges = self.get_scan_ranges()  # yaw and others above also should be normalized into some numbers
        data = self.get_MapData()  # should be normalized and 128x128

        return [
            np.array([originX, originY, xPosition, yPosition, scan_ranges[0], scan_ranges[1]], dtype=int),
            np.array([data], dtype=int),
        ]

    def preprocess_map_data(self, map_data, n, m):
        # 1D diziyi n x m boyutunda bir matrise dönüştür
        map_2d = np.array(map_data).reshape((n, m))

        # Harita verisini 128x128 boyutuna yeniden boyutlandır
        map_resized = resize(np.array(map_2d), [128, 128], mode="reflect", anti_aliasing=True)

        # Map verilerini 0-1 arasında normalize et
        map_normalized = (map_resized - np.min(map_resized)) / (np.max(map_resized) - np.min(map_resized))

        return map_normalized

    def get_originY(self):
        if self.map_data is not None:
            return self.map_data.info.origin.position.y

        return None

    def get_originX(self):
        if self.map_data is not None:
            return self.map_data.info.origin.position.x

        return None

    def get_Xposition(self):
        if self.odom_data is not None:
            return self.odom_data.pose.pose.position.x

        return None

    def get_resolution(self):
        if self.map_data is not None:
            return self.map_data.info.resolution

        return None

    def get_Yposition(self):
        if self.odom_data is not None:
            return self.odom_data.pose.pose.position.y

        return None

    def get_scan_ranges(self):
        if self.scan_data is not None:
            return [max(min(self.scan_data.ranges), 0), min(max(self.scan_data.ranges), 9999.0)]
            self.euler_from_quaternion(
                self.odom_data.pose.pose.orientation.x,
                self.odom_data.pose.pose.orientation.y,
                self.odom_data.pose.pose.orientation.z,
                self.odom_data.pose.pose.orientation.w,
            )

        return None

    def get_MapData(self):
        if self.map_data is not None:
            return self.preprocess_map_data(self.map_data.data, self.map_data.info.height, self.map_data.info.width)

        return None

    def get_min_scan_distance(self):
        if self.scan_data is not None:
            return min(self.scan_data.ranges)

        return None

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z

    def move_robot(self, action):
        if self.previous_twist != None:
            # ileri veya geri gidiyorsa yavaslatarak durdur
            if self.previous_twist.linear.x > 0:
                while self.previous_twist.linear.x > 0:
                    self.previous_twist.linear.x -= 0.1
                    self.publisher.publish(self.previous_twist)

            if self.previous_twist.linear.x < 0:
                while self.previous_twist.linear.x < 0:
                    self.previous_twist.linear.x += 0.1
                    self.publisher.publish(self.previous_twist)

            if self.previous_twist.angular.z < 0:
                while self.previous_twist.angular.z < 0:
                    self.previous_twist.angular.z += 0.1
                    self.publisher.publish(self.previous_twist)

            if self.previous_twist.angular.z > 0:
                while self.previous_twist.angular.z > 0:
                    self.previous_twist.angular.z -= 0.1
                    self.publisher.publish(self.previous_twist)

        # [straight,back,right,left]
        twist = Twist()
        twist.linear.x += float(action[0])
        twist.linear.x += -1.0 * float(action[1])

        twist.angular.z += float(action[2])
        twist.angular.z += -1.0 * float(action[3])

        print("Twist Message:", twist, "Action:", action)
        self.previous_twist = twist
        self.publisher.publish(twist)

    def scan_callback(self, msg):
        self.scan_data = msg

    def map_callback(self, msg):
        self.map_data = msg

    def odom_callback(self, msg):
        self.odom_data = msg


def main(args=None):
    rclpy.init(args=args)
    navigation_control = RLFrontierBaseTrain()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
