import torch
from kaggle_environments import evaluate
import numpy as np
import seaborn as sns
from tqdm import tqdm

from feature_transform import SimpleFeatureTransform, AnnFeatureTransform, YarFeatureTransform

from goose_agents import GreedyAgent, AgentFactory, RLAgentWithRules, SmartGoose, GooseWarlock, GoosePaladin
import matplotlib.pyplot as plt
from qestimator import OneLayerNetQEstimator, AlexNetQEstimator, TwoLayerNetQEstimator, GooseNet4, GooseNet2, GooseNet3, \
    GooseNetResidual4, GooseNetGoogle


class GooseWar:
    def __init__(self, name_to_agent_factories: dict):
        self.warrior_names = list(name_to_agent_factories.keys())
        self.warriors = [name_to_agent_factories[name] for name in name_to_agent_factories.keys()]
        self.n_armies = len(self.warriors)
        self.scores = np.zeros(shape=(self.n_armies, self.n_armies))

    def begin_each_one_vs_all(self, n_rounds):
        n = self.n_armies
        fight_range = tqdm(range(n ** 2), desc='War has been sparked', leave=True)
        for i in fight_range:
            attacker = i // n
            defender = i % n
            fight_range.set_description(desc='{0} is fighting. {1} is under attack'
                                 .format(self.warrior_names[attacker], self.warrior_names[defender]))
            if attacker == defender:
                continue
            self.battle(attacker, defender, n_rounds)

    def battle(self, i, j, n_rounds):
        with torch.no_grad():
            hero = self.warriors[i].create(0)
            enemies = [self.warriors[j].create(0) for _ in range(3)]

            scores = evaluate("hungry_geese",
                              [hero] + enemies,
                              num_episodes=n_rounds, debug=True)

            self.scores[i][j] = np.mean([get_points(sum(r[0] <= r_ for r_ in r if r_ is not None)) for r in scores])

    def show_scores(self):
        fig, ax = plt.subplots(figsize=(20, 30))
        ax = sns.heatmap(self.scores, cmap="YlGnBu")
        plt.yticks(range(self.n_armies), self.warrior_names)
        plt.xticks(range(self.n_armies), self.warrior_names)
        print(self.scores)
        plt.show()


def get_points(place):
    place = int(place)
    if place == 1:
        return 2
    elif place == 2:
        return 1
    elif place == 3:
        return 0.25
    else:
        return 0


if __name__ == "__main__":
    device = 'cpu'
    factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform.get_state)

    name_to_factory = {"greedy1": factory}
    # name_to_factory = {}
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

    # ###
    # net_path = "./Champions/7500_GooseNetGoogle_c14_h7_w11_DQN.net"
    # net = GooseNetGoogle(n_channels=14)
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, GooseWarlock, AnnFeatureTransform.get_state)
    # name_to_factory["Google_7.5k"] = agent_factory
    ###
    ###
    # net_path = "Champions/150000_YarEstimator4LayerResNet_14ch_c14_h7_w11_DQN.net"
    # net = GooseNetResidual4(n_channels=14)
    #
    # net.load_state_dict(torch.load(net_path))
    # net.eval()
    # agent_factory = AgentFactory(net, SmartGoose, AnnFeatureTransform.get_state)
    # name_to_factory["Residual4L"] = agent_factory
    ###
    ###
    net_path = "./Champions/10000_YarEstimator4Layer_c12_h7_w11_DQN.net"
    net = GooseNet4(n_channels=12)

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, SmartGoose, YarFeatureTransform.get_state)
    name_to_factory["GooseNet smart"] = agent_factory
    ###
    ###
    net_path = "./Champions/10000_YarEstimator4Layer_c12_h7_w11_DQN.net"
    net = GooseNet4(n_channels=12)

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, GooseWarlock, YarFeatureTransform.get_state)
    name_to_factory["GooseNet Warlock"] = agent_factory
    ###
    ###
    net_path = "./Champions/75000_GooseNetGoogle_c14_h7_w11_DQN.net"
    net = GooseNetGoogle(n_channels=14)

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, GooseWarlock, AnnFeatureTransform.get_state)
    name_to_factory["Google Warlock"] = agent_factory
    ###
    ###
    net_path = "./Champions/75000_GooseNetGoogle_c14_h7_w11_DQN.net"
    net = GooseNetGoogle(n_channels=14)

    net.load_state_dict(torch.load(net_path))
    net.eval()
    agent_factory = AgentFactory(net, GoosePaladin, AnnFeatureTransform.get_state)
    name_to_factory["Google Paladin"] = agent_factory
    ###
    ###
    war = GooseWar(name_to_factory)
    war.begin_each_one_vs_all(1)

    war.show_scores()