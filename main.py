"""
File: main.py
Author: Mitchel Bekink
Date: 11/04/2025
Description: Main file for running the PPO algorithm with various testing
environments.
"""

import ppo_alg
import test_environments
import gymnasium as gym
import time

version_num = "1.0.0"
last_mod = "11/04/2025"

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
    time.sleep(3)

def run(algorithm_str, environment_str, track_mode, time_steps):
    if algorithm_str == "PPO":
        if(environment_str == "TurtleTest1"):
            ppo_alg.PPO(test_environments.TurtleTest1(), 4, 4, track_mode).learn(time_steps)
        else:
            match environment_str:
                case "CartPole-v1":
                    ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"),2, 4, track_mode).learn(time_steps)
                case "MountainCar-v0":
                    ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"),3, 2, track_mode).learn(time_steps)
                case "Acrobot-v1":
                    ppo_alg.PPO(gym.make(environment_str,  render_mode="rgb_array"),3, 6, track_mode).learn(time_steps)

setup(version_num, last_mod)

# Current available gym test environments are:
# CartPole-v1
# MountainCar-v0
# Acrobot-v1
run("PPO", "TurtleTest1", 3, 100200)