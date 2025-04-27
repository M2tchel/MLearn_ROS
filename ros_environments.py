"""
File: ros_environments.py
Author: Mitchel Bekink
Date: 25/04/2025
Description: A selection of different ROS environment presetes to be used with
the PPO algorithm, each contains different combinations of input devices.
"""

import numpy as np
import ppo_ros_controller
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import rospy
import math
import matplotlib.pyplot as plt

class ROSLidar():
    def __init__(self, log_level):
        self.current_goal_x = 0
        self.current_goal_y = 0
        self.current_timestep = 0
        self.complete = False
        self.success_rew = 10000
        self.collision_rew = -100
        self.fail_rew = -1000
        self.success_dist = 0.15
        self.timesteps_per_ep = 2500
        self.object_dist = 0.2
        self.controller = ppo_ros_controller.Controller()
        self.base_rate = 30 # In Hz, need to fix as it actually runs much faster than this value even with rtf
        self.times_goal_reached = 0
        self.times_goal_not_reached = 0
        self.log_level = log_level
        self.total_timesteps = 0
    
    def reset(self):
        bot_coords = self.controller.get_pos()

        while(True):
            self.current_goal_x = np.random.randint(-13, 14) / 10
            self.current_goal_y = np.random.randint(-18, 18) / 10
            if(self.current_goal_x > bot_coords.x + 2*self.success_dist) or (self.current_goal_x < bot_coords.x - 2*self.success_dist):
                if(self.current_goal_y > bot_coords.y + 2*self.success_dist) or (self.current_goal_y < bot_coords.y - 2*self.success_dist):
                    break
        
        self.complete = False
        self.current_timestep = 0

        output = self.controller.get_obs()
        output.append(np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y))
        output.append(math.atan2(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.x) * 180 / math.pi)
        output.append(self.controller.get_angle()[2])

        return output

    def step(self, action):
        self.total_timesteps += 1

        self.controller.perform_action(action)

        current_rate = rospy.Rate(self.base_rate * self.controller.get_rtf())

        current_rate.sleep()

        self.current_timestep += 1

        obs = self.controller.get_obs()
        bot_coords = self.controller.get_pos()

        dist = np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y)
        goal_angle = math.atan2(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.x) * 180 / math.pi
        bot_angle = self.controller.get_angle()[2]

        if self.current_timestep >= self.timesteps_per_ep:
            rew = self.fail_rew
            self.complete = True
            self.times_goal_not_reached += 1
            self.log(rew, dist, goal_angle, bot_angle, bot_coords)
            obs.append(dist)
            obs.append(goal_angle)
            obs.append(bot_angle)
            return obs, rew, self.complete

        if (dist <= self.success_dist):
            rew = self.success_rew
            self.complete = True
            self.times_goal_reached += 1
            self.log(rew, dist, goal_angle, bot_angle, bot_coords)
            obs.append(dist)
            obs.append(goal_angle)
            obs.append(bot_angle)
            return obs, rew, self.complete

        dist_rew = (6 - dist) / 6 * 10

        proximity_punishment = 0

        min_dist = min(obs)

        if min_dist <= 0.15:
            proximity_punishment += self.collision_rew
        elif min_dist <= self.object_dist:
            proximity_punishment += self.collision_rew * 0.2
        
        rew = dist_rew + proximity_punishment

        if(rew > 0):
            rew = rew * ((self.timesteps_per_ep - self.current_timestep) / self.timesteps_per_ep)

        self.log(rew, dist, goal_angle, bot_angle, bot_coords)
        obs.append(dist)
        obs.append(goal_angle)
        obs.append(bot_angle)

        return obs, rew, self.complete

    def render(self, bot_coords):
            plt.clf()
            plt.xlabel("Current Timestep: " + str(self.total_timesteps))
            plt.plot([self.current_goal_x], [self.current_goal_y], "g*")
            plt.plot([bot_coords.x], [bot_coords.y], "r^")
            plt.axis((-5, 5, -5, 5))
            plt.pause(0.0001)
            plt.draw()
    
    def log(self, rew, dist, goal_angle, bot_angle, bot_coords):
        if self.log_level > 0 and self.current_timestep % 100 == 0:
            self.render(bot_coords)
        if self.log_level == 1 and self.total_timesteps % self.timesteps_per_ep == 0:
            print("No times reached goal: " + str(self.times_goal_reached))
            print("No times not reached goal: " + str(self.times_goal_not_reached))
        if self.log_level > 2:
            print("Current reward is: " + str(rew))
            print("No times reached goal: " + str(self.times_goal_reached))
            print("No times not reached goal: " + str(self.times_goal_not_reached))
        if self.log_level > 3:
            print("Current distance from goal: " + str(dist))
            print("Current angle to goal: " + str(goal_angle))
            print("Current bot angle: " + str(bot_angle))