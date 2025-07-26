"""
File: ros_environments.py
Author: Mitchel Bekink
Date: 25/04/2025
Description: A selection of different ROS environment presets to be used with
the PPO algorithm, each contains different combinations of input devices. This is
where you can add functionality for your own sensors/reward systems etc.
"""

import numpy as np
import ppo_ros_controller
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import rospy
import math
import matplotlib.pyplot as plt
import datetime
from collections import deque

class ROSLidarFull():
    def __init__(self, log_level):

        # Initialise the parameters.
        self.current_goal_x = 0
        self.current_goal_y = 0
        self.current_timestep = 0
        self.complete = False
        self.success_rew = 10000
        self.collision_rew = -100
        self.fail_rew = -1000
        self.success_dist = 0.25
        self.timesteps_per_ep = 2500
        self.object_dist = 0.2
        self.controller = ppo_ros_controller.TurtleBot3Controller()
        self.base_rate = 30 # In Hz.
        self.times_goal_reached = 0
        self.times_goal_not_reached = 0
        self.log_level = log_level
        self.total_timesteps = 0
        self.previous_render_timestamp = datetime.datetime.now().timestamp()
        self.previous_render_timestep = 0
        self.overall_number_of_timesteps = -1
        self.recent_etas = deque()
    
    # "Resets" the scene, setting a new goal for the bot and returning the new observation.
    def reset(self):
        bot_coords = self.controller.get_pos()

        # Loops while attempting to set new goal until one is set that is a good distance away from the bot.
        while(True):
            self.current_goal_x = np.random.randint(-13, 14) / 10
            self.current_goal_y = np.random.randint(-18, 18) / 10
            if(self.current_goal_x > bot_coords.x + 2*self.success_dist) or (self.current_goal_x < bot_coords.x - 2*self.success_dist):
                if(self.current_goal_y > bot_coords.y + 2*self.success_dist) or (self.current_goal_y < bot_coords.y - 2*self.success_dist):
                    break
        
        self.complete = False
        self.current_timestep = 0

        # Gets the current observation for the bot.
        output = self.controller.get_obs()
        output.append(np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y))
        output.append(math.atan2(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.x) * 180 / math.pi)
        output.append(self.controller.get_angle()[2])

        return output

    # Performs one action, calculates the reward and checks if goal has been met.
    def step(self, action):
        self.total_timesteps += 1

        self.controller.perform_action(action)

        # Ensures enough time passes for the action to occur in sim before moving on.
        current_rate = rospy.Rate(self.base_rate * self.controller.get_rtf())
        current_rate.sleep()

        self.current_timestep += 1

        obs = self.controller.get_obs()
        bot_coords = self.controller.get_pos()

        dist = np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y)
        goal_angle = math.atan2(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.x) * 180 / math.pi
        bot_angle = self.controller.get_angle()[2]

        # Bot has taken too long to reach goal.
        if self.current_timestep >= self.timesteps_per_ep:
            rew = self.fail_rew
            self.complete = True
            self.times_goal_not_reached += 1
            self.log(rew, dist, goal_angle, bot_angle, bot_coords)
            obs.append(dist)
            obs.append(goal_angle)
            obs.append(bot_angle)
            return obs, rew, self.complete

        # Bot has reached goal.
        if (dist <= self.success_dist):
            rew = self.success_rew
            self.complete = True
            self.times_goal_reached += 1
            self.log(rew, dist, goal_angle, bot_angle, bot_coords)
            obs.append(dist)
            obs.append(goal_angle)
            obs.append(bot_angle)
            return obs, rew, self.complete

        # Else:
        dist_rew = (6 - dist) / 6 * 10

        proximity_punishment = 0

        min_dist = min(obs)

        # Punishment for being too close to obstacle:
        if min_dist <= 0.15:
            proximity_punishment += self.collision_rew
        elif min_dist <= self.object_dist:
            proximity_punishment += self.collision_rew * 0.2
        
        rew = dist_rew + proximity_punishment

        # Decreases positive reward based on how long bot has been in current episode:
        if(rew > 0):
            rew = rew * ((self.timesteps_per_ep - self.current_timestep) / self.timesteps_per_ep)

        self.log(rew, dist, goal_angle, bot_angle, bot_coords)
        obs.append(dist)
        obs.append(goal_angle)
        obs.append(bot_angle)

        return obs, rew, self.complete

    # Renders the current position of the bot in relation to its goal, and shows training ETA.
    def render(self, bot_coords):
            if(self.overall_number_of_timesteps != -1):
                current_time = datetime.datetime.now().timestamp()
                time_passed = current_time - self.previous_render_timestamp
                timesteps_passed = self.total_timesteps - self.previous_render_timestep
                overall_timesteps_remaining = self.overall_number_of_timesteps - self.total_timesteps
                estimated_time_remaining = overall_timesteps_remaining / (timesteps_passed/time_passed)
                self.previous_render_timestamp = current_time
                self.previous_render_timestep = self.total_timesteps
                self.recent_etas.append(estimated_time_remaining)
                if(len(self.recent_etas) >= 20):
                    self.recent_etas.popleft()
                averaged_time_remaining = np.mean(self.recent_etas)
            
            plt.clf()
            plt.title("Current Timestep: " + str(self.total_timesteps))
            plt.xlabel("Estimated Time Remaining: " + str(datetime.timedelta(seconds=averaged_time_remaining)))
            plt.plot([self.current_goal_x], [self.current_goal_y], "g*")
            plt.plot([bot_coords.x], [bot_coords.y], "r^")
            plt.axis((-5, 5, -5, 5))
            plt.pause(0.0001)
            plt.draw()
    
    # Prints debug information to terminal based on level chosen.
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
    
    def set_timesteps(self, timesteps):
        self.overall_number_of_timesteps = timesteps

class ROSLidarReduced():

    def __init__(self, log_level):
        # Initialise the parameters.
        self.current_goal_x = 0
        self.current_goal_y = 0
        self.current_timestep = 0
        self.complete = False
        self.success_rew = 10000
        self.collision_rew = -100
        self.fail_rew = -1000
        self.success_dist = 0.25
        self.timesteps_per_ep = 2500
        self.object_dist = 0.2
        self.controller = ppo_ros_controller.TurtleBot3Controller()
        self.base_rate = 30 # In Hz.
        self.times_goal_reached = 0
        self.times_goal_not_reached = 0
        self.log_level = log_level
        self.total_timesteps = 0
        self.previous_render_timestamp = datetime.datetime.now().timestamp()
        self.previous_render_timestep = 0
        self.overall_number_of_timesteps = -1
        self.recent_etas = deque()
    
    # "Resets" the scene, setting a new goal for the bot and returning the new observation.
    def reset(self):
        bot_coords = self.controller.get_pos()

        # Loops while attempting to set new goal until one is set that is a good distance away from the bot.
        while(True):
            self.current_goal_x = np.random.randint(-13, 14) / 10
            self.current_goal_y = np.random.randint(-18, 18) / 10
            if(self.current_goal_x > bot_coords.x + 2*self.success_dist) or (self.current_goal_x < bot_coords.x - 2*self.success_dist):
                if(self.current_goal_y > bot_coords.y + 2*self.success_dist) or (self.current_goal_y < bot_coords.y - 2*self.success_dist):
                    break
        
        self.complete = False
        self.current_timestep = 0

        # Gets the current observation for the bot.
        output = self.controller.get_reduced_obs()
        output.append(np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y))
        output.append(math.atan2(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.x) * 180 / math.pi)
        output.append(self.controller.get_angle()[2])

        return output

    # Performs one action, calculates the reward and checks if goal has been met.
    def step(self, action):
        self.total_timesteps += 1

        self.controller.perform_action(action)

        # Ensures enough time passes for the action to occur in sim before moving on.
        current_rate = rospy.Rate(self.base_rate * self.controller.get_rtf())
        current_rate.sleep()

        self.current_timestep += 1

        obs = self.controller.get_reduced_obs()
        bot_coords = self.controller.get_pos()

        dist = np.hypot(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.y)
        goal_angle = math.atan2(self.current_goal_x - bot_coords.x, self.current_goal_y - bot_coords.x) * 180 / math.pi
        bot_angle = self.controller.get_angle()[2]

        # Bot has taken too long to reach goal.
        if self.current_timestep >= self.timesteps_per_ep:
            rew = self.fail_rew
            self.complete = True
            self.times_goal_not_reached += 1
            self.log(rew, dist, goal_angle, bot_angle, bot_coords)
            obs.append(dist)
            obs.append(goal_angle)
            obs.append(bot_angle)
            return obs, rew, self.complete

        # Bot has reached goal.
        if (dist <= self.success_dist):
            rew = self.success_rew
            self.complete = True
            self.times_goal_reached += 1
            self.log(rew, dist, goal_angle, bot_angle, bot_coords)
            obs.append(dist)
            obs.append(goal_angle)
            obs.append(bot_angle)
            return obs, rew, self.complete

        # Else:
        dist_rew = (6 - dist) / 6 * 10

        proximity_punishment = 0

        min_dist = min(obs)

        # Punishment for being too close to obstacle:
        if min_dist <= 0.15:
            proximity_punishment += self.collision_rew
        elif min_dist <= self.object_dist:
            proximity_punishment += self.collision_rew * 0.2
        
        rew = dist_rew + proximity_punishment

        # Decreases positive reward based on how long bot has been in current episode:
        if(rew > 0):
            rew = rew * ((self.timesteps_per_ep - self.current_timestep) / self.timesteps_per_ep)

        self.log(rew, dist, goal_angle, bot_angle, bot_coords)
        obs.append(dist)
        obs.append(goal_angle)
        obs.append(bot_angle)

        return obs, rew, self.complete

    # Renders the current position of the bot in relation to its goal, and shows training ETA.
    def render(self, bot_coords):
            if(self.overall_number_of_timesteps != -1):
                current_time = datetime.datetime.now().timestamp()
                time_passed = current_time - self.previous_render_timestamp
                timesteps_passed = self.total_timesteps - self.previous_render_timestep
                overall_timesteps_remaining = self.overall_number_of_timesteps - self.total_timesteps
                estimated_time_remaining = overall_timesteps_remaining / (timesteps_passed/time_passed)
                self.previous_render_timestamp = current_time
                self.previous_render_timestep = self.total_timesteps
                self.recent_etas.append(estimated_time_remaining)
                if(len(self.recent_etas) >= 20):
                    self.recent_etas.popleft()
                averaged_time_remaining = np.mean(self.recent_etas)

            plt.clf()
            plt.title("Current Timestep: " + str(self.total_timesteps))
            plt.xlabel("Estimated Time Remaining: " + str(datetime.timedelta(seconds=averaged_time_remaining)))
            plt.plot([self.current_goal_x], [self.current_goal_y], "g*")
            plt.plot([bot_coords.x], [bot_coords.y], "r^")
            plt.axis((-5, 5, -5, 5))
            plt.pause(0.0001)
            plt.draw()
    
    # Prints debug information to terminal based on level chosen.
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
    
    def set_timesteps(self, timesteps):
        self.overall_number_of_timesteps = timesteps

# NOT YET IMPLEMENTED
class ROSTurtleBot3():
    def __init__(self, log_level):
        self.current_timestep = 0
        self.complete = False
        self.success_rew = 10000
        self.collision_rew = -100
        self.fail_rew = -1000
        self.success_dist = 0.25
        self.timesteps_per_ep = 2500
        self.object_dist = 0.2
        self.controller = ppo_ros_controller.TurtleBot3Controller()
        self.base_rate = 2 # In Hz, need to fix as it actually runs much faster than this value even with rtf
        self.times_goal_reached = 0
        self.times_goal_not_reached = 0
        self.log_level = log_level
        self.total_timesteps = 0
        self.previous_render_timestamp = datetime.datetime.now().timestamp()
        self.previous_render_timestep = 0
        self.overall_number_of_timesteps = -1
        self.recent_etas = deque()
    
    def reset(self):
        
        self.complete = False
        self.current_timestep = 0

        output = self.controller.get_reduced_obs()

        return output

    def step(self, action):
        self.total_timesteps += 1

        self.controller.perform_action(action)

        current_rate = rospy.Rate(self.base_rate * self.controller.get_rtf())

        current_rate.sleep()

        self.current_timestep += 1

        obs = self.controller.get_reduced_obs()

        if self.current_timestep >= self.timesteps_per_ep:
            rew = self.fail_rew
            self.complete = True
            self.times_goal_not_reached += 1
            self.log(rew)
            return obs, rew, self.complete

        proximity_punishment = 0

        min_dist = min(obs)

        if min_dist <= 0.15:
            proximity_punishment += self.collision_rew
        elif min_dist <= self.object_dist:
            proximity_punishment += self.collision_rew * 0.2
        
        rew = proximity_punishment

        if(rew > 0):
            rew = rew * ((self.timesteps_per_ep - self.current_timestep) / self.timesteps_per_ep)

        self.log(rew)

        return obs, rew, self.complete

    def render(self):
            None
    
    def log(self, rew):
        if self.log_level == 1 and self.total_timesteps % self.timesteps_per_ep == 0:
            print("No times reached goal: " + str(self.times_goal_reached))
            print("No times not reached goal: " + str(self.times_goal_not_reached))
            print("Current Timestep: " + str(self.total_timesteps))
        if self.log_level > 2:
            print("Current reward is: " + str(rew))
            print("No times reached goal: " + str(self.times_goal_reached))
            print("No times not reached goal: " + str(self.times_goal_not_reached))
            print("Current Timestep: " + str(self.total_timesteps))
    
    def set_timesteps(self, timesteps):
        self.overall_number_of_timesteps = timesteps