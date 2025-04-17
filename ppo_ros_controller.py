#!/usr/bin/env python3
"""
File: ppo_ros_controller.py
Author: Mitchel Bekink
Date: 17/04/2025
Description: Controls the publishing and subscribing needed to connect the PPO algorithm
to the various ROS nodes and topics.
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates

class Controller():

    def __init__(self):
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.sub_lidar = rospy.Subscriber("/scan", LaserScan, callback=self.lidar_callback)
        self.sub_pos = rospy.Subscriber("/gazebo/model_states", ModelStates, callback=self.pos_callback)

        self.current_throttle = 0.0
        self.current_steer = 0.0
        self.max_throttle = 0.25 # Unsure what happens when you increase this beyond 0.25

        rospy.sleep(1)

    def lidar_callback(self, msg: LaserScan):
        self.current_obs = msg.ranges

    def pos_callback(self, msg: ModelStates):

        index = msg.name.index("turtlebot3_waffle")
        pose = msg.pose[index]
        self.coordinates = pose.position

    def get_obs(self):
        new_obs = []
        for value in self.current_obs:
            if value > 10:
                new_obs.append(10)
            else:
                new_obs.append(value)
        return new_obs

    def get_pos(self):
        return self.coordinates

    def perform_action(self, action):
        if (action == 1) and (self.current_throttle <= self.max_throttle):
            self.current_throttle += 0.01
        elif (action == 2) and (self.current_throttle >= -self.max_throttle):
            self.current_throttle -= 0.01
        elif (action == 3) and (self.current_steer <= 1.7):
            self.current_steer += 0.01
        elif (action == 4) and (self.current_steer >= -1.7):
            self.current_steer -= 0.01
        elif action == 5:
            self.current_steer = 0
            self.current_throttle = 0
    
        msg = Twist()
        msg.linear.x = self.current_throttle
        msg.angular.z = self.current_steer
        self.pub.publish(msg)