"""
File: networks.py
Author: Mitchel Bekink
Date: 25/04/2025
Description: A selection of nn designs, created to allow easy experimentation
with different designs within learning algorithms. This is where you can add your
own NN architectures.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Basic FF-NN with 512 hidden nodes
class FeedForwardNN_512(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN_512, self).__init__()
        no_nodes = 512
        self.odometer = 0

        self.layer1 = nn.Linear(in_dim, no_nodes)
        self.layer2 = nn.Linear(no_nodes, no_nodes)
        self.layer3 = nn.Linear(no_nodes, out_dim)

    def forward(self, obs): 

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        x_1 = F.relu(self.layer1(obs))
        x_2 = F.relu(self.layer2(x_1))
        output = self.layer3(x_2)
        self.odometer += 1
        return output

    def get_odometer(self):
        return self.odometer

# Basic FF-NN with 256 hidden nodes
class FeedForwardNN_256(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN_256, self).__init__()
        no_nodes = 256
        self.odometer = 0

        self.layer1 = nn.Linear(in_dim, no_nodes)
        self.layer2 = nn.Linear(no_nodes, no_nodes)
        self.layer3 = nn.Linear(no_nodes, out_dim)

    def forward(self, obs): 

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        x_1 = F.relu(self.layer1(obs))
        x_2 = F.relu(self.layer2(x_1))
        output = self.layer3(x_2)
        self.odometer += 1
        return output
    
    def get_odometer(self):
        return self.odometer
    
# Basic FF-NN with 128 hidden nodes
class FeedForwardNN_128(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN_128, self).__init__()
        no_nodes = 128
        self.odometer = 0

        self.layer1 = nn.Linear(in_dim, no_nodes)
        self.layer2 = nn.Linear(no_nodes, no_nodes)
        self.layer3 = nn.Linear(no_nodes, out_dim)

    def forward(self, obs): 

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        x_1 = F.relu(self.layer1(obs))
        x_2 = F.relu(self.layer2(x_1))
        output = self.layer3(x_2)
        self.odometer += 1
        return output

    def get_odometer(self):
        return self.odometer

# Basic FF-NN with 64 hidden nodes
class FeedForwardNN_64(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN_64, self).__init__()
        no_nodes = 64
        self.odometer = 0

        self.layer1 = nn.Linear(in_dim, no_nodes)
        self.layer2 = nn.Linear(no_nodes, no_nodes)
        self.layer3 = nn.Linear(no_nodes, out_dim)

    def forward(self, obs): 

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        x_1 = F.relu(self.layer1(obs))
        x_2 = F.relu(self.layer2(x_1))
        output = self.layer3(x_2)
        self.odometer += 1
        return output

    def get_odometer(self):
        return self.odometer