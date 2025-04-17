"""
File: test_environments.py
Author: Mitchel Bekink
Date: 17/04/2025
Description: Contains a range of testing environments made to mimic different
ROS environments to test learning algorithms without needing to run a full
simulation.
"""

import numpy as np

class TurtleTest1():
    # A simple 5x5 2D coordinate self.board where the agent needs to find the end in
    # as few a moves as possible

    def __init__(self):

        # Initialise the variables
        self.current_x = 1
        self.current_y = 5
        self.goal_x = 5
        self.goal_y = 1
        self.starting_dist = np.hypot(self.current_x - self.goal_x, self.current_y - self.goal_y)
        self.complete = False
        self.fail_rew = -1
        self.timeout_rew = -50
        self.success_rew = 200
        self.initialise_board()
        self.current_vals = ()
        self.current_step = 0
        self.max_steps = 15
    
    def initialise_board(self):
        # Initialise the self.board (-1=fail, 0=neutral, 1=agent, 2=success)
        self.current_x = 1
        self.current_y = 5
        self.board = np.zeros((7,7))
        for i in range(7):
            self.board[i][0] = -1
            self.board[0][i] = -1
            self.board[i][-1] = -1
            self.board[-1][i] = -1
        self.board[self.goal_y][self.goal_x] = 2
        self.board[self.current_y][self.current_x] = 1
        self.board[4][2] = -1
        self.board[2][4] = -1
        self.board[3][4] = -1
        self.board[4][4] = -1
        self.board[1][2] = -1
        self.board[2][2] = -1

    def get_obs_dim(self):
        # Returns the dimensionality of the observations
        return 4

    def get_act_dim(self):
        # Returns the no. of different actions the agent can take each timestep
        return 4

    def reset(self):
        # Resets the environment
        # Restart the board
        self.initialise_board()
        self.complete = False
        self.current_step = 0

        # Calculate and return the starting observation
        obs = []
        obs.append(self.board[self.current_y + 1][self.current_x]) # Looking up
        obs.append(self.board[self.current_y - 1][self.current_x]) # Looking down
        obs.append(self.board[self.current_y][self.current_x - 1]) # Looking left
        obs.append(self.board[self.current_y][self.current_x + 1]) # Looking right

        return obs

    def step(self, action):
        # Progresses to the next timestep, performing the passed action
        if action == 0:
            self.current_y -= 1 # Move up
        elif action == 1:
            self.current_y += 1 # Move down
        elif action == 2:
            self.current_x -= 1 # Move left
        elif action == 3:
            self.current_x += 1 # Move right
        
        self.current_step += 1

        # Case if the agent has failed (moved onto a -1)

        if(self.current_step >= self.max_steps):
            obs = [0, 0, 0, 0]
            rew = self.fail_rew
            self.complete = True
            self.current_vals = obs, rew, self.complete
            return self.current_vals

        if(self.board[self.current_y][self.current_x] == -1):
            obs = [0, 0, 0, 0]
            rew = self.fail_rew
            self.complete = True
            self.current_vals = obs, rew, self.complete
            return self.current_vals

        # Case if the agent has succeeded (moved onto a 2)
        if(self.board[self.current_y][self.current_x] == 2):
            obs = [0, 0, 0, 0]
            rew = self.success_rew
            self.complete = True
            self.current_vals = obs, rew, self.complete
            return self.current_vals
        
        # Case if the agent has moved to a neutral space
        # Update the board with the movement
        if action == 0:
            self.board[self.current_y + 1][self.current_x] = 0
        elif action == 1:
            self.board[self.current_y - 1][self.current_x] = 0
        elif action == 2:
            self.board[self.current_y][self.current_x + 1] = 0
        elif action == 3:
            self.board[self.current_y][self.current_x - 1] = 0
        self.board[self.current_y][self.current_x] = 1

        # Calculate the observation after the move
        obs = []
        obs.append(self.board[self.current_y + 1][self.current_x]) # Looking up
        obs.append(self.board[self.current_y - 1][self.current_x]) # Looking down
        obs.append(self.board[self.current_y][self.current_x - 1]) # Looking left
        obs.append(self.board[self.current_y][self.current_x + 1]) # Looking right

        # The reward value is the percentage of the distance travelled, multiplied by half the success reward
        rew = round((self.starting_dist - np.hypot(self.current_x - self.goal_x, self.current_y - self.goal_y)) / self.starting_dist * (self.success_rew/10) / self.current_step)

        self.current_vals = obs, rew, self.complete
        return self.current_vals
    
    def manual(self):
        # Allows manual navigation of the environment, for testing and debugging
        while True:
            print(self.board)
            print()
            key = input("AWSD for next move, C to exit: ").upper()
            print()
            if key == "C":
                print("Exiting...\n")
                self.reset()
                return
            elif key == "A":
                obs, rew, done = self.step(2)
            elif key =="W":
                obs, rew, done = self.step(0)
            elif key =="S":
                obs, rew, done = self.step(1)
            elif key =="D":
                obs, rew, done = self.step(3)
            else:
                print("Invalid input, please try again")
                print()
                
            print("Current observations: " + str(obs))
            print("Current reward: " + str(rew))
            if(done):
                print("Episode complete, restarting...\n")
                self.reset()
    
    def render(self):
        obs = self.current_vals[0]
        rew = self.current_vals[1]
        done = self.current_vals[2]

        output = ""
        output += "Current observations: " + str(obs) + "\n"
        output += "Current reward: " + str(rew) + "\n"
        if(done):
            output += "Episode complete, restarting...\n"
        output += str(self.board)
        output += "\n"
        return output