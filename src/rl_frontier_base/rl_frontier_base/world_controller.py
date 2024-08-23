import subprocess
import time
import os
import signal


class WorldManager:

    def __init__(self) -> None:
        self.process = None

    def launch_world(self):
        # Launch dosyasını başlatan komutu çalıştırmak
        self.process = subprocess.Popen(
            ["ros2", "launch", "rl_custom_robot_bringup", "rl_custom_robot_gazebo.launch.xml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def kill_process(self):
        if self.process is None:
            print("Process is not running or already terminated.")
            return

        # Process'i durdurmak için
        self.process.terminate()

        # Gerekirse process'i zorla sonlandırmak (kill)
        if self.process.poll() is None:  # Process hala çalışıyorsa
            os.kill(self.process.pid, signal.SIGKILL)

        # Gazebo'yu zorla kapatma
        subprocess.call(["pkill", "-9", "gazebo"])
        subprocess.call(["pkill", "-9", "gzserver"])
        subprocess.call(["pkill", "-9", "gzclient"])

        # RViz'i kapatma
        subprocess.call(["pkill", "-9", "rviz2"])
