from replay_memory import Transition, ReplayMemory, update_priorities, get_priority_weight
from qestimator import QEstimator, QEstimatorFactory, AlexNetQEstimator
from goose_agents import RLAgent, GreedyAgent, AgentFactory
from feature_transform import SimpleFeatureTransform, SquareWorld7x7FeatureTransform
from goose_tools import get_eps

from tqdm import tqdm
from kaggle_environments import make, evaluate
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Observation, Configuration
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class TrainGeese:
    _debug = False
    _win_rates = {'Score': [], 'Rank': []}
    _EPS_START = 0.85
    _EPS_END = 0.05
    _BATCH_SIZE = 64
    _GAMMA = 0.9

    _TARGET_UPDATE = 50
    _MEMORY_THRESHOLD = 10000
    _MEMORY_GREEDY_EXAMPLES = _MEMORY_THRESHOLD // 2
    _COLLECT_FROM_POLICY_NUM = 3
    _EVALUATE_FROM_POLICY_NUM = 3
    _MAXIMAL_REPLAY_PRIORITY = 1000

    def __init__(self,
                 replay_memory,
                 nn_factory,
                 agent_factory,
                 abstract_feature_transform_class,
                 device,
                 starting_net
                 ):
        """
        :param replay_memory:
        :param nn_factory:
        :param device: Устройство, на котором проводятся вычисления
        """
        self._env = make("hungry_geese", debug=TrainGeese._debug)
        self.result_net_path = ""

        self._device = device

        self._replay_memory = replay_memory
        self._MEMORY_THRESHOLD = replay_memory.capacity

        self._get_state = abstract_feature_transform_class.get_state
        self._get_reward = abstract_feature_transform_class.get_reward
        self._adjust_action = abstract_feature_transform_class.adjust_action
        self.feature_c = abstract_feature_transform_class.FEATURES_CHANNELS_IN
        self.feature_h = abstract_feature_transform_class.FEATURES_HEIGHT
        self.feature_w = abstract_feature_transform_class.FEATURES_WIDTH

        self._policy_net = nn_factory.create()
        if starting_net is not None:
            self._policy_net.load_state_dict(starting_net.state_dict())
        self._target_net = nn_factory.create()
        self._result_net = nn_factory.create()
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._result_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._result_net.eval()
        self._optimizer = optim.RMSprop(self._policy_net.parameters())
        # self._optimizer = torch.optim.Adam(policy_net.parameters(), 3e-4)

        self._agent_factory = agent_factory
        self._agent_factory.net = self._policy_net

        # self._enemy_factory = AgentFactory(self._policy_net, RLAgent, abstract_feature_transform_class, self._device)
        self._enemy_factory = AgentFactory(None, GreedyAgent, abstract_feature_transform_class, self._device)
        self._rollout_agent_factory = AgentFactory(None, GreedyAgent, abstract_feature_transform_class, self._device)

        self._last_saved_score = 0
        self.n_episodes = 0

    def train(self, n_episodes):
        self.n_episodes = n_episodes
        self._reset_statistics()
        self.collect_initial_replay_memory()

        for episode in tqdm(range(n_episodes), desc='Training', leave=False):
            eps_greedy = get_eps(self._EPS_START, self._EPS_END, episode, n_episodes)
            self.collect_data(self._agent_factory, self._enemy_factory, self._COLLECT_FROM_POLICY_NUM, eps_greedy)
            self.optimize_model(episode, n_episodes)

            if episode % self._TARGET_UPDATE == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

            self._update_result_net()

    def optimize_model(self, step, max_step):
        if len(self._replay_memory) < self._BATCH_SIZE:
            return
        priority_sample = self._replay_memory.get(self._BATCH_SIZE)

        idx = list(map(lambda x: x[0], priority_sample))
        transitions = list(map(lambda x: x[1], priority_sample))
        priorities = list(map(lambda x: x[2], priority_sample))
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self._device, dtype=torch.bool)

        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        # torch.stack в данном случае не будет работать
        if len(non_final_next_states_list) == 0:
            return

        non_final_next_states = torch.stack(non_final_next_states_list)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        self._policy_net.train()
        state_action_values = self._policy_net(torch.stack(batch.state, dim=0, out=None)).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._BATCH_SIZE, device=self._device)
        next_state_values[non_final_mask] = self._target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._GAMMA) + reward_batch.squeeze(1)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data *= get_priority_weight(priorities, self._replay_memory, step, max_step)
            param.grad.data.clamp_(-1, 1)

        self._optimizer.step()

        with torch.no_grad():
            update_priorities(idx, torch.abs(state_action_values.squeeze(1) - expected_state_action_values).squeeze(),
                              self._replay_memory)

        self._policy_net.eval()

    def collect_initial_replay_memory(self):
        self.collect_data(self._rollout_agent_factory,
                          self._rollout_agent_factory,
                          self._MEMORY_GREEDY_EXAMPLES,
                          eps_greedy=0.1)

    def collect_data(self, agent_factory, enemy_factory, n_episodes, eps_greedy):
        configuration = Configuration(self._env.configuration)
        collect_range = tqdm(range(n_episodes), desc='Collecting', leave=False) if n_episodes > 100 \
            else range(n_episodes)

        for _ in collect_range:
            done = False
            agent = agent_factory.create(eps_greedy=eps_greedy)
            enemies = [enemy_factory.create(eps_greedy=0) for _ in range(3)]
            trainer = self._env.train([None] + enemies)

            prev_obs = None

            observation = Observation(trainer.reset())
            state = self._get_state(observation, prev_obs)
            # start rollout
            while not done:
                # make a move
                action_index = agent.get_action_index(observation, configuration)
                action_name = Action(action_index + 1).name
                action = torch.tensor([[action_index]], device=self._device, dtype=torch.long)

                prev_obs = observation
                prev_state = state
                # observe
                observation, reward, done, _ = trainer.step(action_name)
                observation = Observation(observation)

                if not done:
                    state = self._get_state(observation, prev_obs)
                else:
                    state = None

                reward = self._get_reward(observation, prev_obs, configuration)
                reward = torch.tensor([[reward]], device=self._device, dtype=torch.float)

                priority = self._MAXIMAL_REPLAY_PRIORITY
                self._replay_memory.push(priority, prev_state, action, state, reward)

    def save_result(self):
        feature_str = "c{0}_h{1}_w{2}".format(self.feature_c, self.feature_h, self.feature_w)
        self.result_net_path = './results/{0}_{1}_{2}_DQN.net'.format(self.n_episodes,
                                                                      str(self._policy_net),
                                                                      feature_str)
        torch.save(self._result_net.state_dict(), self.result_net_path)

    def print_win_rates(self):
        t = np.arange(len(self._win_rates['Score']))
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('TimeSteps')
        ax1.set_ylabel('Score', color=color)
        ax1.plot(t, self._win_rates['Score'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Rank', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, self._win_rates['Rank'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Performance of DQN-net vs 3 Greedy Agents')
        plt.show()

    def _evaluate(self, save_statistics=True):
        self._policy_net.eval()
        with torch.no_grad():
            agent = self._agent_factory.create(0, self._policy_net)
            enemies = [self._enemy_factory.create(0.1, None) for _ in range(3)]
            scores = evaluate("hungry_geese",
                              [agent] + enemies,
                              num_episodes=self._EVALUATE_FROM_POLICY_NUM, debug=TrainGeese._debug)
            evaluated_score = np.mean([r[0] for r in scores])

        if save_statistics:
            self._win_rates['Score'].append(evaluated_score)
            self._win_rates['Rank'].append(np.mean([sum(r[0] <= r_ for r_ in r if r_ is not None) for r in scores]))

        return evaluated_score

    def _reset_statistics(self):
        self._win_rates['Score'] = []
        self._win_rates['Rank'] = []

    def _update_result_net(self):
        evaluated_score = self._evaluate()
        # save agent if he outperforms our previous games
        if self._last_saved_score * 1.2 < evaluated_score:
            # check again whether it plays well
            second_evaluation_score = self._evaluate(save_statistics=False)
            if self._last_saved_score < second_evaluation_score:
                self._result_net.load_state_dict(self._policy_net.state_dict())
                self._last_saved_score = second_evaluation_score


def train(feature_transform_class, n_episodes, kernel_sizes, layer_channels, starting_net):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    replay_memory = ReplayMemory(10000)
    agent_factory = AgentFactory(None, RLAgent, feature_transform_class, device)
    nn_factory = QEstimatorFactory(estimator_class=AlexNetQEstimator,
                                   channel_in=feature_transform_class.FEATURES_CHANNELS_IN,
                                   height=feature_transform_class.FEATURES_HEIGHT,
                                   width=feature_transform_class.FEATURES_WIDTH,
                                   kernel_sizes=kernel_sizes,
                                   layer_channels=layer_channels,
                                   device=device)

    train_geese = TrainGeese(replay_memory=replay_memory,
                             nn_factory=nn_factory,
                             agent_factory=agent_factory,
                             abstract_feature_transform_class=feature_transform_class,
                             device=device,
                             starting_net=starting_net)
    train_geese.train(n_episodes)
    train_geese.print_win_rates()
    train_geese.save_result()

    record_game(feature_transform_class,
                net_path=train_geese.result_net_path,
                kernel_sizes=kernel_sizes,
                layer_channels=layer_channels)


def record_game(feature_transform_class, net_path, kernel_sizes, layer_channels):
    env = make("hungry_geese", debug=True)
    env.reset()
    device = torch.device("cpu")

    net = AlexNetQEstimator(c=feature_transform_class.FEATURES_CHANNELS_IN,
                     h=feature_transform_class.FEATURES_HEIGHT,
                     w=feature_transform_class.FEATURES_WIDTH,
                     kernel_sizes=kernel_sizes,
                     layer_channels=layer_channels,
                     outputs=4,
                     device=device).to(device)

    net.load_state_dict(torch.load(net_path))
    net.eval()

    agent_factory = AgentFactory(net, RLAgent, feature_transform_class, device)
    player = agent_factory.create(eps_greedy=0)
    players = [agent_factory.create(eps_greedy=0) for _ in range(2)] + \
              [AgentFactory(None, GreedyAgent, feature_transform_class, device).create(0)]

    env.run([player] + players)
    out = env.render(mode="html", width=500, height=400)

    f = open("game.html", "w")
    f.write(out)
    f.close()


if __name__ == '__main__':
    device='cpu'
    feature_transform_class = SimpleFeatureTransform
    kernel_sizes = [(3, 3), (3, 3), (3, 3)]
    layer_channels = [16, 16, 16]

    # net_path = "./results/35000_QEstimator_ks22_33_22_ch64_32_16_c12_h7_w11_DQN.net"
    # net = QEstimator(c=12,
    #                  h=7,
    #                  w=11,
    #                  kernel_sizes=kernel_sizes,
    #                  layer_channels=layer_channels,
    #                  outputs=4,
    #                  device=device).to(device)

    # net.load_state_dict(torch.load(net_path))

    train(feature_transform_class, 35000, kernel_sizes, layer_channels, None)
    # record_game(feature_transform_class, net_path, kernel_sizes, layer_channels)