#!/usr/bin/env python3
"""
File: main.py
Author: Mitchel Bekink
Date: 25/04/2025
Description: Main file for running the PPO algorithm with various testing
environments. Please not that roscore and a Gazebo simulation with a turtlebot3
model must already be running before running this script!
"""

import rospy
import ppo_alg
import test_environments
import gymnasium as gym
import time
import ros_environments

def setup(version_num, last_mod):
    print()
    print("##########################################################################")
    print("#################### ROS REINFORCEMENT LEARNING " + version_num + " ####################")
    print()
    print("Author: Mitchel Bekink")
    print("Email: xlk501@york.ac.uk")
    print("Last Modified: " + last_mod)
    print()
    print("Please check GitHub README for references and further resources.\n")
    print("##########################################################################")
    print("##########################################################################\n")
    time.sleep(2)

def learn(algorithm_str, environment_str, track_mode, time_steps):
    if algorithm_str == "PPO":
        if(environment_str == "TurtleTest1"):
            current_network = ppo_alg.PPO(test_environments.TurtleTest1(track_mode), 4, 4, 0)
            current_network.learn(time_steps)
        else:
            if environment_str == "CartPole-v1":
                current_network = ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"),2, 4, track_mode)
                current_network.learn(time_steps)
            elif environment_str == "MountainCar-v0":
                current_network = ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"),3, 2, track_mode)
                current_network.learn(time_steps)
            elif environment_str == "Acrobot-v1":
                current_network = ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"),3, 6, track_mode)
                current_network.learn(time_steps)
            elif environment_str == "ROSLidar":
                current_network = ppo_alg.PPO(ros_environments.ROSLidar(track_mode), 6, 363, 0)
                # Note that 362 obs dimension because 360 degrees, then dist from goal, then angle to goal, then bot angle
                current_network.learn(time_steps)
    return current_network

def run(network, time_steps):
    network.learn(time_steps)

if __name__ == '__main__':
    rospy.init_node("main_ppo")

    rospy.loginfo("Hello from main PPO!")

    rospy.sleep(1.0)

    rospy.loginfo("End of starter bit?")

    if(True):

        version_num = "1.0.3"
        last_mod = "25/04/2025"

        setup(version_num, last_mod)

        # Current available test environments are:
        # TurtleTest1
        # CartPole-v1
        # MountainCar-v0
        # Acrobot-v1

        # Current ROS environments are:
        # ROSLidar

        current_network = learn("PPO", "ROSLidar", 4, 1000000)
        run(current_network, 100000)