from feature_proc import get_features
import torch
import random
import numpy as np
import math


class RLAgent:
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

    def __init__(self, net, device):
        self.steps_done = 0
        self.n_actions = 4
        self.prev_heads = [-1, -1, -1, -1]
        self.net = net
        self.device = device

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.net(state.cuda().unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def __call__(self, observation, configuration):
        if observation['step'] == 0:
            self.prev_heads = [-1, -1, -1, -1]
        state = get_features(observation, configuration, self.prev_heads)
        action = self.select_action(state)
        self.prev_heads = [goose[0] if len(goose) > 0 else -1 for goose in observation['geese']]
        return ['NORTH', 'EAST', 'SOUTH', 'WEST'][action]
