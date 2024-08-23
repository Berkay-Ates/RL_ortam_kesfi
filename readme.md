# Reinforcement Learning Custom Robot Simulation Project

## Overview

> This project involves a custom robot simulation in Gazebo, utilizing ROS 2. The robot model is defined using URDF and Xacro files, and the project includes a Lidar sensor for scanning. General purpose of the project is trainng a RL model for environment mapping and frontier exploration.

## Project Structure

The project is structured as follows:

- **rl_custom_robot_bringup**: `rl_custom_robot_gazebo.launch.xml` - Main launch file to start the Gazebo simulation and RViz visualization. This file is automatically handled while training.

- **rl_frontier_base**: `rl_control_agent_node.py` - This file is runner file for starting the train process.

## How to Run

### Prerequisites

- ROS 2 installed.
- Gazebo installed.
- Custom robot packages (`custom_robot_description`, `custom_robot_bringup`) correctly set up.

### Steps

1. **Build the Workspace**:

   ```sh
   colcon build
   ```
2. **Source the Setup Files:**:
   ```sh
   source install/setup.bash
   ```
3. **Launch the Training Simulation:**:
   ```sh
   python3 rl_frontier_base/rl_frontier_base/rl_control_agent_node.py
   ```


## Purpose

> The primary goal of this project is to train a robot using QLearning to enable it for automatically exploring the environment.

## Future Works

>

1.  Add more sensors to the robot model.
2.  Implement advanced reward functions.
3.  Integrate with other ROS 2 packages for extended functionality.

## Notes
 
[!(image.png![alt text](image.png))](https://youtu.be/MbZN7rhkaCM)


This README provides an overview and instructions to get started with the custom robot simulation project in ROS 2.
