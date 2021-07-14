import torch
from kaggle_environments import evaluate
import numpy as np
import seaborn as sns
from feature_transform import SimpleFeatureTransform

from goose_agents import GreedyAgent, AgentFactory, RLAgentWithRules, SmartGoose
import matplotlib.pyplot as plt
from qestimator import OneLayerNetQEstimator, AlexNetQEstimator, TwoLayerNetQEstimator, GooseNet4, GooseNet2, GooseNet3


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
    factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform.get_state)

    # name_to_factory = {"greedy1": factory}
    name_to_factory = {}
    # ####
    # net_path = "./Champions/30000_OneLayerNetQEstimator_c12_h7_w11_DQN.net"
    # net = OneLayerNetQEstimator()
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    # name_to_factory["30000_OneLayerNetQEstimator"] = agent_factory
    ####
    # net_path = "Champions/30002_TwoLayerNetQEstimator_c12_h7_w11_DQN.net"
    # net = TwoLayerNetQEstimator()
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    # name_to_factory["30002_TwoLayerNetQEstimator"] = agent_factory
    #######
    #
    # net_path = "Champions/10000_YarEstimator2Layer_c12_h7_w11_DQN.net"
    # net = GooseNet2()
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    # name_to_factory["2Layers"] = agent_factory

    ###
    # net_path = "Champions/30k_YarEstimator3Layer.net"
    # net = GooseNet3()
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    # name_to_factory["3Layers"] = agent_factory
    # ###

    ###
    net_path = "./Champions/10000_YarEstimator4Layer_c12_h7_w11_DQN.net"
    net = GooseNet4()

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    name_to_factory["4layers"] = agent_factory
    ###
    ###
    net_path = "./Champions/10000_YarEstimator4Layer_c12_h7_w11_DQN.net"
    net = GooseNet4()

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, SmartGoose, SimpleFeatureTransform.get_state)
    name_to_factory["Smart"] = agent_factory
    ###
    ###
    war = GooseWar(name_to_factory)
    war.begin_each_one_vs_all(100)

    war.show_scores()