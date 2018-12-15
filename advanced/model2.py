import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenetv2 import MobileNetV2
from utils import *
from torchvision import transforms

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, action_size, seed, cnn_out_size=1280, fc1_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cnn = MobileNetV2(input_size=input_size)
        self.cnn.load_state_dict(torch.load(open('mobilenet_v2.pth.tar', 'rb'), map_location=map_location))
        self.fc1 = nn.Linear(cnn_out_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, state):
        if len(state.shape)<4:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        for i in range(len(state)):
            state[i] = normalize(state[i])
        state = state.permute(0, 3, 1, 2)

        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.cnn.features(state).view(batch_size, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, action_size, seed, cnn_out_size=1280, fc1_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cnn = MobileNetV2(input_size=input_size)
        self.cnn.load_state_dict(torch.load(open('mobilenet_v2.pth.tar', 'rb'), map_location=map_location))
        self.fc1 = nn.Linear(cnn_out_size+action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, state, action):
        if len(state.shape)<4:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        for i in range(len(state)):
            state[i] = normalize(state[i])
        state = state.permute(0, 3, 1, 2)

        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.cnn.features(state).view(batch_size, -1))
        x = F.relu(self.fc1(torch.cat((x, action), dim=1)))

        return self.fc2(x)
