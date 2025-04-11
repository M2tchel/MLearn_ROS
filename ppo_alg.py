"""
File: ppo_alg.py
Author: Mitchel Bekink
Date: 11/04/2025
Description: An implementation of the PPO learning algorithm, designed to be used
within a ROS environment.
"""

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import networks
import test_environments
from matplotlib import pyplot as plt

class PPO:
    def __init__(self, environment, act_dim, obs_dim, track_mode):

        # Hyperparameter Initialisation:
        self.timesteps_per_batch = 1000
        self.timesteps_per_episode = 200
        self.gamma = 0.95 # The discount rate for future rewards
        self.updates_per_iteration = 3 # The number of nn updates per iteration
        self.clip = 0.2 # Used in the clipping function
        self.lr = 0.001
        self.nn_design = networks.FeedForwardNN_3 # Neural network design
        self.nn_no_nodes = 64
        self.optimiser = Adam
        self.epsilon = 0.1

        # Environment information for nn initialisation
        self.env = environment
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Initialise the neural networks
        self.actor = self.nn_design(self.obs_dim, self.act_dim, self.nn_no_nodes)
        self.critic = self.nn_design(self.obs_dim, 1, self.nn_no_nodes)

        # Initialise actor/critic optimiser
        self.actor_optim = self.optimiser(self.actor.parameters(), lr=self.lr)
        self.critic_optim = self.optimiser(self.critic.parameters(), lr=self.lr)

        # Peformance Tracking
        self.max_reward = -np.inf
        self.times_at_max = 0
        self.current_timestep = 0
        self.playback_count = 0
        self.track_mode = track_mode

    def run_batch(self):
        # Initialise data for this batch
        batch_obs = [] # batch observations
        batch_acts = [] # batch actions
        batch_log_probs = [] # log probabilities of each action
        batch_rews = [] # batch rewards-to-go
        batch_lens = [] # episodic lengths in batch
        batch_t = 0 # Number of timesteps ran in this batch so far

        # Outer loop for batch control
        while batch_t < self.timesteps_per_batch:
            ep_rews = [] # The rewards from this episode

            obs = self.env.reset() # Reset the environment
            ep_complete = False

            for ep_t in range(self.timesteps_per_episode):
                # Increment batch timestep counter
                batch_t+=1
                self.current_timestep += 1

                # Save the current observation
                if(isinstance(obs,tuple)):
                    batch_obs.append(torch.tensor(obs[0], dtype=torch.float32))
                else:
                    batch_obs.append(torch.tensor(obs, dtype=torch.float32))

                # Query the actor for the next action, then run this action
                if(isinstance(obs,tuple)):
                    obs_tensor = torch.tensor(obs[0], dtype=torch.float32)
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)

                action, log_prob = self.get_action(obs_tensor)
                values = self.env.step(action)
                obs = values[0]
                rew = values[1]
                ep_terminated = values[2]

                # Some enviornments differentiate between truncated and terminated,
                # we don't care so combine the two
                if(len(values) > 3):
                    ep_truncated = values[3]
                    if(ep_terminated or ep_truncated):
                        ep_complete = True
                    else:
                        ep_complete = False
                else:
                    ep_complete = ep_terminated

                # Performance Tracking
                # Log the maximum reward achieved, and print every n timesteps
                if(self.track_mode >= 1):
                    if(self.current_timestep % 1000 == 0):
                        print("Current Timestep: " + str(self.current_timestep) + ", current max value: " + str(self.max_reward))
                    if(rew > self.max_reward):
                        self.max_reward = rew
                        self.times_at_max = 0
                    elif(rew == self.max_reward):
                        self.times_at_max += 1

                # For environments that support rendering, render 200 timesteps every n timesteps
                if(self.track_mode >= 2 ) and not (isinstance(self.env,test_environments.TurtleTest1)):
                    if(self.current_timestep % 10000 == 0):
                        self.playback_count = 200
                
                    if self.playback_count > 0:
                        plt.clf()
                        img = self.env.render()
                        plt.imshow(img)
                        plt.pause(0.01)
                        plt.title("Performance at Timestep: " + str(self.current_timestep))
                        plt.axis('off')
                        plt.draw()
                        self.playback_count -= 1
                    elif self.playback_count == 0:
                        plt.close("all")
                        self.playback_count -= 1
                elif(self.track_mode >= 2):
                    if(self.current_timestep % 10000 == 0):
                        self.playback_count = 200
                    if self.playback_count > 0:
                        print(self.env.render())
                        self.playback_count -= 1
                    elif self.playback_count == 0:
                        self.playback_count -= 1

                # Prints each time a new max reward is achieved or achieved again
                if(self.track_mode >= 3):
                    if(rew > self.max_reward):
                        print(self.env.show_info(obs, rew, ep_complete))
                        print("Max Reward Beaten! Currently at: " + str(self.max_reward) + ", at iteration: " + str(self.current_timestep))
                    elif(rew == self.max_reward):
                        print("Reached Max of: " + str(self.max_reward) + " Again! Count at: " + str(self.times_at_max) + ", at iteration: " + str(self.current_timestep))

                # Save all the data obtained from this timestep
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # Ends loop early if the episode has ended
                if ep_complete:
                    break
        
                # Save the remaining data after episode has ended
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

            # Calculate the rewards-to-go for all batch rewards
            batch_rtgs = self.compute_rtgs(batch_rews)

            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):

        # Ensures correct typing and dimensionality
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        # Query the actor network for action logits
        logits = self.actor(obs)

        # Compute both the log and regular probabilities of each action
        probabilities = F.softmax(logits, dim=1)
        log_probabilities = F.log_softmax(logits, dim=1)

        if (self.epsilon > np.random.rand()): # Explore:
            action = torch.multinomial(probabilities, num_samples=1)
        else: # Greedy:
            action = torch.argmax(log_probabilities, dim=1, keepdim=True)
        
        # Get the probability of the chosen action
        log_probability = log_probabilities.gather(1, action)

        return action.item(), log_probability.squeeze()

    def compute_rtgs(self, batch_rews):
        # The shape will be the number of timesteps per episode
        batch_rtgs = []

        # Iterate through each episode backwards as we are working from back to front
        for ep_rews in reversed(batch_rews):

            total = 0

            for rew in reversed(ep_rews):
                total = rew + total * self.gamma
                batch_rtgs.insert(0, total)
        
        # Convert to a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        
        # Ensure correct typing and dimensionality
        if not isinstance(batch_obs, torch.Tensor):
            batch_obs = torch.stack(batch_obs)
        if not isinstance(batch_acts, torch.Tensor):
            batch_acts = torch.tensor(batch_acts, dtype=torch.long)

        # Query critic network for a value for each observation
        V = self.critic(batch_obs).squeeze()

        # Calculating the log probabilities of each action that occured
        logits = self.actor(batch_obs)
        log_probs = F.log_softmax(logits, dim=1)

        action_log_probs = log_probs[torch.arange(len(batch_acts)), batch_acts]

        return V, action_log_probs

    def learn(self, total_timesteps):
        t_so_far = 0 # The overall number of timesteps simulated thus far

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.run_batch()

            # Update the overall number of timesteps with how many occured in this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate the advantage
            A_k = batch_rtgs - V

            A_k = (A_k - A_k.mean()) / (A_k.std(unbiased=False) + 1e-9) # The extra addition is to prevent / by 0

            # Updating the actor and critic nn
            for _ in range(1):
                # Calculate V_phi and pi_theta
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs[0])

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k # The clipping function
                min_func = -torch.min(surr1, surr2)

                actor_loss = min_func.mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backwards propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                #Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
        print("Max reward achieved: " + str(self.max_reward) + ", percentage of timesteps where this reward was achieved: " + str(self.times_at_max / self.current_timestep * 100) + "%")

# Uncomment to run on GPU (not recommended)
#torch.cuda.set_device(0)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')