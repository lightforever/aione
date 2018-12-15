import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenetv2 import MobileNetV2

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, action_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cnn = MobileNetV2(input_size=input_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, action_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
