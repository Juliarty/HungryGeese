import torch
from kaggle_environments import evaluate
import numpy as np
import seaborn as sns
from feature_transform import SimpleFeatureTransform

from goose_agents import RLAgent, GreedyAgent, AgentFactory
import matplotlib.pyplot as plt
from qestimator import OneLayerNetQEstimator, GooseNet, AlexNetQEstimator


class GooseWar:
    def __init__(self, name_to_agent_factories: dict):
        self.warrior_names = name_to_agent_factories.keys()
        self.warriors = [name_to_agent_factories[name] for name in name_to_agent_factories.keys()]
        self.n_armies = len(self.warriors)
        self.scores = np.zeros(shape=(self.n_armies, self.n_armies))

    def begin_each_one_vs_all(self, n_rounds):
        n = self.n_armies
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self.battle(i, j, n_rounds)

    def battle(self, i, j, n_rounds):
        with torch.no_grad():
            hero = self.warriors[i].create(0)
            enemies = [self.warriors[j].create(0) for _ in range(3)]

            scores = evaluate("hungry_geese",
                              [hero] + enemies,
                              num_episodes=n_rounds, debug=False)

            self.scores[i][j] = np.mean([sum(r[0] <= r_ for r_ in r if r_ is not None) for r in scores])

    def show_scores(self):
        fig, ax = plt.subplots(figsize=(20, 30))
        ax = sns.heatmap(self.scores, cmap="YlGnBu")
        plt.yticks(range(self.n_armies), self.warrior_names)
        plt.xticks(range(self.n_armies), self.warrior_names)
        plt.show()


if __name__ == "__main__":
    device = 'cpu'
    factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform)

    name_to_factory = {"greedy1": factory}

    #####
    # net_path = "./results/35k_7x11_ks_432_24_48_simple_DQN"
    # net = QEstimator(c=12,
    #                  h=7,
    #                  w=11,
    #                  kernel_sizes=[(4, 4), (3, 3), (2, 2)],
    #                  layer_channels=[32, 48, 48],
    #                  outputs=4,
    #                  device=device).to(device)
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform, device)
    # name_to_factory["35k_7x11_ks_432_24_48"] = agent_factory
    #######
    #

    #####

    net_path = "./results/40000_OneLayerNetQEstimator_ks33_33_00_ch32_64_256_c12_h7_w11_DQN.net"
    net = OneLayerNetQEstimator()

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform)
    name_to_factory["40000_OneLayerNetQEstimator"] = agent_factory
    #######

    #####

    net_path = "./results/5000_OneLayerNetQEstimator_ks33_33_00_ch32_64_256_c12_h7_w11_DQN.net"
    net = OneLayerNetQEstimator()

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform)
    name_to_factory["5000_OneLayerNetQEstimator"] = agent_factory
    #######
    #####

    # net_path = "./results/40000_AlexNetQEstimator_ks33_33_00_ch32_64_256_c12_h7_w11_DQN.net"
    # net = AlexNetQEstimator(c=12,
    #                             h=7,
    #                             w=11,
    #                             kernel_sizes=[(3, 3), (3, 3), (4, 4)],
    #                             layer_channels=[32, 64, 48],
    #                             outputs=4,
    #                             device=device).to(device)
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform, device)
    # name_to_factory["40000_AlexNetQEstimator"] = agent_factory

    #####

    net_path = "./results/20000_YarEstimator_c12_h7_w11_DQN.net"
    net = GooseNet()

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform)
    name_to_factory["20000_YarEstimator"] = agent_factory
    #######
    # #####
    # net_path = "./results/20k_7x11_ks232_l64l32l16_simple.net"
    # net = QEstimator(c=12,
    #                  h=7,
    #                  w=11,
    #                  kernel_sizes=[(2, 2), (3, 3), (2, 2)],
    #                  layer_channels=[64, 32, 16],
    #                  outputs=4,
    #                  device=device).to(device)
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform, device)
    # name_to_factory["20k_7x11_ks232_l64l32l16"] = agent_factory
    # #######

    # #####
    # net_path = "./results/35000_QEstimator_ks22_33_22_ch64_32_16_c12_h7_w11_DQN_oldrew.net"
    # net = QEstimator(c=12,
    #                  h=7,
    #                  w=11,
    #                  kernel_sizes=[(2, 2), (3, 3), (2, 2)],
    #                  layer_channels=[64, 32, 16],
    #                  outputs=4,
    #                  device=device).to(device)
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgent, SimpleFeatureTransform, device)
    # name_to_factory["35000_QEstimator_ks22_33_22_ch64_32_16_c12_h7_w11_DQN"] = agent_factory
    # #######

    war = GooseWar(name_to_factory)
    war.begin_each_one_vs_all(100)

    war.show_scores()