"""
File: networks.py
Author: Mitchel Bekink
Date: 25/04/2025
Description: A selection of nn designs, created to allow easy experimentation
with different designs within learning algorithms (specifically within a ROS
environment)
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_3(nn.Module):
    def __init__(self, in_dim, out_dim, no_nodes):
        super(FeedForwardNN_3, self).__init__()

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
        return output