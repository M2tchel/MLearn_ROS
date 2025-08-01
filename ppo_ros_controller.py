#!/usr/bin/env python3
"""
File: ppo_ros_controller.py
Author: Mitchel Bekink
Date: 26/07/2025
Description: Controls the publishing and subscribing needed to connect the PPO algorithm
to the various ROS nodes and topics. This is where you can add support for different ROS
capable robots.
"""

import scipy.spatial
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import PerformanceMetrics
import scipy
import numpy as np

class TurtleBot3Controller():

    def __init__(self):
        # Create publishers and subscribers specific to a TurtleBot3 running in Gazebo.
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.sub_lidar = rospy.Subscriber("/scan", LaserScan, callback=self.lidar_callback)
        self.sub_pos = rospy.Subscriber("/gazebo/model_states", ModelStates, callback=self.pos_callback)
        self.sub_rtf = rospy.Subscriber("/gazebo/performance_metrics", PerformanceMetrics, callback=self.rtf_callback)

        self.current_throttle = 0.0
        self.current_steer = 0.0
        self.max_throttle = 0.25 # Do not increase beyond 0.25!
        self.current_rtf = 1.0

        rospy.sleep(1) # Short delay to ensure everything in ROS is initialized before making accesses.

    # Callback for LIDAR readings.
    def lidar_callback(self, msg: LaserScan):
        self.current_obs = msg.ranges

    # Callback to calculate the turtlebot's position and angle (used for reward generation).
    def pos_callback(self, msg: ModelStates):
        try:
            index = msg.name.index("turtlebot3_waffle")
        except:
            index = msg.name.index("turtlebot3")
        pose = msg.pose[index]
        self.coordinates = pose.position
        angle_quat = scipy.spatial.transform.Rotation.from_quat(np.array((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)))
        angle_deg = angle_quat.as_euler('xyz', degrees=True)
        self.angle = angle_deg[0], angle_deg[1], angle_deg[2]
    
    # RTF is used if wanting to increase Gazebo sim speed beyond realtime.
    def rtf_callback(self, msg: PerformanceMetrics):
        self.current_rtf = msg.real_time_factor

    # Gets full LIDAR readings (360 angles).
    def get_obs(self):
        new_obs = []
        for value in self.current_obs:
            if value > 10:
                new_obs.append(10)
            else:
                new_obs.append(value)
        return new_obs
    
    # Gets partial LIDAR readings (36 angles).
    def get_reduced_obs(self):
        new_obs = []
        count = 0
        for value in self.current_obs:
            if(count % 10 == 0):
                if value > 10:
                    new_obs.append(10)
                else:
                    new_obs.append(value)
            count += 1
        return new_obs


    def get_pos(self):
        return self.coordinates
    
    def get_angle(self):
        return self.angle
    
    def get_rtf(self):
        return self.current_rtf

    # Sends action to Gazebo model to perform.
    def perform_action(self, action):
        if (action == 1):
            self.current_throttle = self.max_throttle
            if(self.current_steer > 0.1):
                self.current_steer -= 0.1
            elif(self.current_steer < -0.1):
                self.current_steer += 0.1
            else:
                self.current_steer = 0
        elif (action == 2):
            self.current_throttle = -self.max_throttle
            if(self.current_steer > 0.1):
                self.current_steer -= 0.1
            elif(self.current_steer < -0.1):
                self.current_steer += 0.1
            else:
                self.current_steer = 0
        elif (action == 3):
            self.current_steer = 1.7
        elif (action == 4):
            self.current_steer = -1.7
        elif action == 5:
            self.current_steer = 0
            self.current_throttle = 0
        # Action 6 does nothing, allowing the robot to wait
    
        msg = Twist()
        msg.linear.x = self.current_throttle
        msg.angular.z = self.current_steer
        self.pub.publish(msg)