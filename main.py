#!/usr/bin/env python3
"""
File: main.py
Author: Mitchel Bekink
Date: 26/07/2025
Description: Main file for running the PPO algorithm with various testing
environments. Please note that roscore and a Gazebo simulation with your robot
model must already be running before running this script!
"""

import rospy
import ppo_alg
import test_environments
import gymnasium as gym
import time
import datetime
import ros_environments
import torch
import networks

version_num = "1.0.4"
last_mod = "26/07/2025"

# Variables to modify:
model_file_path = "/home/mitchel/Models/"
model_name = "Main_Lidar_Full"

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

# Note that track_mode is the debug level, 0 being no debug 4 being maximum.
def create(algorithm_str, environment_str, network, track_mode):

    # Current available test environments for algorithms are:
        # TurtleTest1
        # CartPole-v1
        # MountainCar-v0
        # Acrobot-v1

    # Current ROS environments are:
        # ROSLidarFull
        # ROSLidarReduced

    # Current available networks are:
        # FeedForwardNN_64
        # FeedForwardNN_128
        # FeedForwardNN_256
        # FeedForwardNN_512

    if algorithm_str == "PPO":
        if(environment_str == "TurtleTest1"):
            current_network = ppo_alg.PPO(test_environments.TurtleTest1(track_mode), environment_str, network, 4, 4, 0)
        else:
            if environment_str == "CartPole-v1":
                current_network = ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"), environment_str, network, 2, 4, track_mode)
            elif environment_str == "MountainCar-v0":
                current_network = ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"), environment_str,network, 3, 2, track_mode)
            elif environment_str == "Acrobot-v1":
                current_network = ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"), environment_str,network, 3, 6, track_mode)
            elif environment_str == "ROSLidarFull":
                current_network = ppo_alg.PPO(ros_environments.ROSLidarFull(track_mode), environment_str, network, 6, 363, 0)
                # Note that 363 obs dimension because 360 degrees, then dist from goal, then angle to goal, then bot angle
            elif environment_str == "ROSLidarReduced":
                current_network = ppo_alg.PPO(ros_environments.ROSLidarReduced(track_mode), environment_str, network, 6, 39, 0)
                # Note that 39 obs dimension because 36 LIDAR directions, then dist from goal, then angle to goal, then bot angle
            elif environment_str == "ROSTurtleBot3":
                current_network = ppo_alg.PPO(ros_environments.ROSTurtleBot3(track_mode), environment_str, network, 6, 36, 0)
                # Note that 39 obs dimension because 36 LIDAR directions, then dist from goal, then angle to goal, then bot angle

    return current_network

def learn(model, time_steps):
    model.learn(time_steps)

def save(model, name=""):
    file_path = model_file_path

    # Default name is time of save:
    if name == "":
        current_timestamp = str(datetime.datetime.now().strftime("%d.%m.%Y_%H:%M:%S"))
        torch.save(model.get_models()[0], file_path + model.get_env_name() +  "_" + current_timestamp + "_actor.pth")
        torch.save(model.get_models()[1], file_path + model.get_env_name() +  "_" + current_timestamp + "_critic.pth")
    else:
        torch.save(model.get_models()[0], file_path + name + "_actor.pth")
        torch.save(model.get_models()[1], file_path + name + "_critic.pth")
    print("Save Complete!")

# Note that you must pass the full path to each model including .pth!!
def load(model, actor_path, critic_path):
    actor = torch.load(actor_path)
    critic = torch.load(critic_path)
    model.load(actor, critic)
    print("Load Complete!")

if __name__ == '__main__':
    rospy.init_node("main_ppo")

    rospy.sleep(1.0)

    setup(version_num, last_mod)

    # Add your training code in here!
    if(True):

        current_model = create("PPO", "ROSLidarFull", networks.FeedForwardNN_64, 4)
        #learn(current_model, 10000)
        #save(current_model, model_name)
        load(current_model, model_file_path + model_name + "_actor.pth", model_file_path + model_name + "_critic.pth")
        learn(current_model, 100000)
        print(current_model.get_odometer())
        save(current_model, model_name)