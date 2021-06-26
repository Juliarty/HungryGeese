from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, GreedyAgent as kGreedyAgent
from feature_proc import get_action_rotated_north_head
import torch
import random
import math


class AbstractAgent:
    def __init__(self, net, get_state, device):
        self.n_actions = 4
        self.net = net
        self.device = device
        self.get_state = get_state

    def __call__(self, observation: Observation, configuration):
        pass

    def get_action_index(self, observation: Observation, configuration: Configuration):
        action = self.__call__(observation, configuration)
        if action == Action(1).name:
            return 0
        elif action == Action(2).name:
            return 1
        elif action == Action(3).name:
            return 2
        elif action == Action(4).name:
            return 3


class GreedyAgent(AbstractAgent):
    def __init__(self, net, get_state, device, eps):
        super().__init__(net, get_state, device)
        self.eps = eps

    def __call__(self, observation: Observation,  configuration):
        configuration = Configuration(configuration)
        observation = Observation(observation)

        self.greedy_agent = kGreedyAgent(configuration)
        return self.greedy_agent(observation)

    def get_eps_greedy(self, observation: Observation,  configuration: Configuration):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        else:
            return self.get_action_index(observation, configuration)


class RLAgent(AbstractAgent):
    def __init__(self, net, get_state, device, eps, adjust_action):
        super().__init__(net, get_state, device)
        self.steps_done = 0
        self.prev_heads = [-1, -1, -1, -1]
        self.eps = eps
        self.adjust_action = adjust_action

    def get_eps_greedy_action(self, state):
        sample = random.random()

        self.steps_done += 1
        if sample > self.eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = int(self.net(state.unsqueeze(0)).max(1)[1].view(1)[0])
        else:
            action = random.randrange(self.n_actions)

        # в Enum нумерация начинается с 1
        return Action(action + 1)

    def __call__(self, observation,  configuration):
        configuration = Configuration(configuration)
        observation = Observation(observation)
        if observation.step == 0:
            self.prev_heads = [-1, -1, -1, -1]
        state = self.get_state(observation,  self.prev_heads, configuration)
        action = self.get_eps_greedy_action(state)
        if self.adjust_action is not None:
            action = self.adjust_action(observation,
                                               self.prev_heads,
                                               action,
                                               configuration.rows,
                                               configuration.columns)

        self.prev_heads = [goose[0] if len(goose) > 0 else -1 for goose in observation.geese]
        return action.name
