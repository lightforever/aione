import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenetv2 import MobileNetV2

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class BaseModel:s
    def __init__(self, input_size, action_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cnn = MobileNetV2(input_size=input_size)
        self.cnn.load_state_dict(torch.load(open('mobilenet_v2.pth.tar', 'rb')))
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.cnn(state))
        x = F.relu(self.fc1(x))
        return F.tanh(x)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    pass


class Critic(nn.Module):
    """Critic (Value) Model."""

    pass
