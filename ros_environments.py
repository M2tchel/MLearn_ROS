"""
File: ros_environments.py
Author: Mitchel Bekink
Date: 17/04/2025
Description: A selection of different ROS environment presetes to be used with
the PPO algorithm, each contains different combinations of input devices.
"""

import numpy as np
import ppo_ros_controller
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import rospy

class ROSLidar():
    def __init__(self):
        self.current_goal_x = 0
        self.current_goal_y = 0
        self.current_timestep = 0
        self.complete = False
        self.success_rew = 100
        self.collision_rew = -10
        self.fail_rew = -100
        self.success_dist = 0.05
        self.timesteps_per_ep = 1000
        self.object_dist = 0.03
        self.controller = ppo_ros_controller.Controller()
        self.rate = rospy.Rate(30)
    
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
        output.append(self.current_goal_x)
        output.append(self.current_goal_y)

        return output

    def step(self, action):

        self.controller.perform_action(action)

        self.rate.sleep()

        self.current_timestep += 1

        obs = self.controller.get_obs()
        obs.append(self.current_goal_x)
        obs.append(self.current_goal_y)

        if self.current_timestep >= self.timesteps_per_ep:
            rew = self.fail_rew
            self.complete = True
            print(rew)
            return obs, rew, self.complete
        
        bot_coords = self.controller.get_pos()

        if (abs(bot_coords.x - self.current_goal_x) <= self.success_dist) and (abs(bot_coords.y - self.current_goal_y) <= self.success_dist):
            rew = self.success_rew
            self.complete = True
            print(rew)
            return obs, rew, self.complete

        dist_rew = (2.8 - np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y)) / 2.8 * ((self.timesteps_per_ep - self.current_timestep)/self.timesteps_per_ep)

        proximity_punishment = 0

        for distance in obs:
            if distance <= 0.01:
                proximity_punishment += self.collision_rew
            if distance <= self.object_dist:
                proximity_punishment += self.collision_rew * 0.2
        
        rew = dist_rew - proximity_punishment

        print(rew)
        return obs, rew, self.complete

    def render(self):
        return