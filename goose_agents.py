from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Observation, Configuration, Action, GreedyAgent as kGreedyAgent
import torch
import random


class AbstractAgent:

    def __init__(self, net, get_state, eps_greedy):
        self.n_actions = 4
        self.net = net
        if net is not None:
            self.net.eval()

        self.eps_greedy = eps_greedy
        self.get_state = get_state

    def __call__(self, observation: Observation, configuration):
        pass

    def get_action_index(self, observation: Observation, configuration: Configuration):
        action = self.__call__(observation, configuration)
        i = 0
        while i < self.n_actions:
            if action == Action(i + 1).name:
                return i
            i += 1


class AgentFactory:
    def __init__(self, initial_net,
                 abstract_agent_class,
                 get_state):
        self.abstract_agent_class = abstract_agent_class
        self.get_state = get_state
        self.net = initial_net

    def create(self, eps_greedy, updated_net=None):
        if updated_net is not None:
            self.net = updated_net

        return self.abstract_agent_class(self.net, self.get_state, eps_greedy)


class GreedyAgent(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.eps = eps_greedy

    def __call__(self, observation, configuration):
        configuration = Configuration(configuration)
        observation = Observation(observation)
        self.greedy_agent = kGreedyAgent(configuration)

        if random.random() < self.eps:
            return Action(random.randrange(self.n_actions) + 1).name

        return self.greedy_agent(observation)


class RLAgent(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.prev_observation = None
        self.eps = eps_greedy

    def _get_eps_greedy_action(self, state):
        if random.random() > self.eps:
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
        self.net.eval()
        observation = Observation(observation)

        state = self.get_state(observation,  self.prev_observation)
        action = self._get_eps_greedy_action(state)
        self.prev_observation = observation

        return action.name


class RLAgentWithRules(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.prev_observation = None
        self.eps = eps_greedy
        self.prev_action = 0

    def __call__(self, observation,  configuration):
        self.net.eval()
        observation = Observation(observation)
        state = self.get_state(observation,  self.prev_observation)
        q_values = self.net(state.unsqueeze(0)).squeeze()
        q_values[(self.prev_action + 2) % 4] = -10000
        action_index = int(q_values.max(0)[1])
        self.prev_action = action_index
        return Action(action_index + 1).name


class EnemyFactoryRandom:
    def __init__(self, agents,
                 probabilities):
        self.agents = agents
        self.probabilities = probabilities

    def create(self):
        return random.choices(self.agents, self.probabilities, k=1)[0]


