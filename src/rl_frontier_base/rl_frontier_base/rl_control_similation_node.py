import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from .world_controller import WorldManager
import threading
import numpy as np
from skimage.transform import resize
import torch
import math


class RLFrontierBaseSimilation(Node):
    def __init__(self):
        super().__init__("rl_control_similation_node")

        self.scan_data = None
        self.previous_map_data = np.array([])
        self.map_data = None
        self.odom_data = None

        # similasyon ortamini olusturma
        self.world_manager = WorldManager()
        self.start_similation()

        # # Start exploration in a new thread
        # thread = threading.Thread(target=self.exploration)
        # thread.start()

    def calculate_reward(self, originX, originY, resolution, posX, posY, action, prev_map_data):
        reward = 0

        if resolution == 0 or posX == None or posY == None or action == None or len(prev_map_data) < 1:
            return 0

        # Piksel koordinatlarını hesaplama

        x_pixel = abs(int((posX - originX) / resolution))
        y_pixel = abs(int((posY - originY) / resolution))

        try:
            # Keşfedilmemiş bir alana gidildiyse ödül ver
            if prev_map_data[x_pixel, y_pixel] <= 0 and self.map_changed == 1:
                print("NO error assigning reward +10", prev_map_data[x_pixel, y_pixel], x_pixel, y_pixel)
                reward += 10
        except:
            print("ERROR CHECKING INDEX", prev_map_data.shape, x_pixel, y_pixel)
            print("originX:", originX, "originY:", originY, "posX:", posX, "posY:", posY)

        # Aynı yöne gitme cezası
        if action == self.previous_direction:
            self.same_direction_count += 1
            if self.same_direction_count >= 2:  # Aynı yöne gitme sayısı eşiği
                reward -= 15  # Ceza puanı
        else:
            self.same_direction_count = 0  # Yön değişti, sayacı sıfırla

        self.previous_direction = action
        self.map_changed = 0

        return reward

    def move_step(self, action):
        self.similation_iteration += 1
        # 1. move
        self.move_robot(action)

        # 2. check if collision
        reward = 0
        similation_over = False
        min_distance = self.get_min_scan_distance()

        if (min_distance is not None and min_distance < 0.6) or self.similation_iteration > 100:
            similation_over = True
            reward = -15
            self.score = len([x for x in self.map_data.data if x > 0])

            return reward, similation_over, self.score

        # 3. calculate reward
        reward = self.calculate_reward(
            self.get_originX(),
            self.get_originY(),
            self.get_resolution(),
            self.get_Xposition(),
            self.get_Yposition(),
            action,
            self.previous_map_data,
        )

        self.score = len([x for x in self.map_data.data if x >= 0])

        return reward, similation_over, self.score

    def move_robot(self, action):
        if self.previous_twist is not None:
            self.smooth_stop()

        # [straight,back,right,left]
        twist = Twist()
        twist.linear.x += float(action[0])
        twist.linear.x += -1.0 * float(action[1])

        twist.angular.z += float(action[2])
        twist.angular.z += -1.0 * float(action[3])

        print("Twist Message:", twist, "Action:", action)
        self.previous_twist = twist
        self.publisher.publish(twist)

    def smooth_stop(self):
        stop_step = 0.05  # Yavaşlama adımı büyüklüğü
        sleep_time = 0.1  # Her adım arasındaki bekleme süresi

        # Lineer yavaşlatma
        while abs(self.previous_twist.linear.x) > 0:
            if self.previous_twist.linear.x > 0.0:
                self.previous_twist.linear.x = max(0.0, self.previous_twist.linear.x - stop_step)
            elif self.previous_twist.linear.x < 0.0:
                self.previous_twist.linear.x = min(0.0, self.previous_twist.linear.x + stop_step)

            self.publisher.publish(self.previous_twist)
            time.sleep(sleep_time)

        # Açısal yavaşlatma
        while abs(self.previous_twist.angular.z) > 0.0:
            if self.previous_twist.angular.z > 0.0:
                self.previous_twist.angular.z = max(0.0, self.previous_twist.angular.z - stop_step)
            elif self.previous_twist.angular.z < 0.0:
                self.previous_twist.angular.z = min(0.0, self.previous_twist.angular.z + stop_step)

            self.publisher.publish(self.previous_twist)
            time.sleep(sleep_time)

        # Robot tamamen durduğunda
        self.previous_twist.linear.x = 0.0
        self.previous_twist.angular.z = 0.0
        self.publisher.publish(self.previous_twist)

    def reset(self):
        self.previous_direction = None
        self.same_direction_count = 0

        self.score = 0
        self.similation_iteration = 0
        self.kill_similation()
        time.sleep(5.0)  # for now just sleep the thread for 5 sec
        self.start_similation()

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

    def preprocess_map_data(self, map_data, n, m):
        # 1D diziyi n x m boyutunda bir matrise dönüştür
        map_2d = np.array(map_data).reshape((n, m))

        # Harita verisini 128x128 boyutuna yeniden boyutlandır
        map_resized = resize(np.array(map_2d), [128, 128], mode="reflect", anti_aliasing=True)

        # Map verilerini 0-1 arasında normalize et
        map_normalized = (map_resized - np.min(map_resized)) / (np.max(map_resized) - np.min(map_resized))

        return map_normalized

    def start_similation(self):
        self.previous_direction = None
        self.same_direction_count = 0

        self.score = 0
        self.similation_iteration = 0
        self.map_changed = 0
        # start gazebo and rviz with particular nodes
        self.world_manager.launch_world()

        self.previous_twist = None

        self.map_subscription = self.create_subscription(OccupancyGrid, "map", self.map_callback, 10)
        self.odom_subscriptions = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.scan_subscriptions = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)

    def kill_similation(self):
        self.world_manager.kill_process()

        self.destroy_subscription(self.odom_subscriptions)
        self.destroy_subscription(self.scan_subscriptions)
        self.destroy_subscription(self.map_subscription)
        self.destroy_publisher(self.publisher)
        self.scan_data = None
        self.map_data = None
        self.odom_data = None
        self.previous_twist = None

    def scan_callback(self, msg):
        self.scan_data = msg

    def map_callback(self, msg):

        if self.map_data is None:
            self.previous_map_data = msg
        else:
            self.previous_map_data = self.map_data

        self.map_data = msg

        self.previous_map_data = np.reshape(
            self.previous_map_data.data, (self.previous_map_data.info.height, self.previous_map_data.info.width)
        )
        self.map_changed = 1

    def odom_callback(self, msg):
        self.odom_data = msg


def main(args=None):
    rclpy.init(args=args)
    navigation_control = RLFrontierBaseSimilation()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
